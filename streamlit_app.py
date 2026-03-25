from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import streamlit as st
from PIL import Image

from data_loader import load_padchest_metadata, load_retrieval_metadata, save_retrieval_metadata
from embedding import extract_embeddings, get_device
from faiss_index import build_index_flat_l2, load_faiss_index, save_faiss_index
from pca import fit_pca, print_pca_variance, save_pca_model
from rag_explainer import build_rag_context, generate_explanation
from retrieval import search_similar_cases
from smoke_test import run_query_smoke, validate_artifacts


DEFAULT_CSV = r"C:\Users\Shiva\OneDrive\Pictures\Documents\Code\minor\chest_x_ray_images_labels_sample.csv"
DEFAULT_IMAGES_ROOT = r"C:\Users\Shiva\OneDrive\Pictures\Documents\Code\minor\sample"
DEFAULT_OUTPUT_DIR = r"C:\Users\Shiva\OneDrive\Pictures\Documents\Code\minor\artifacts"


def prepare_xray_for_display(image_path: Path) -> Image.Image:
    """Convert medical grayscale images (including 16-bit) to viewable 8-bit grayscale."""
    with Image.open(image_path) as img:
        arr = np.array(img)

    return prepare_xray_array_for_display(arr)


def prepare_xray_array_for_display(arr: np.ndarray) -> Image.Image:
    """Convert a loaded image array to viewable 8-bit output."""
    if arr.ndim == 2:
        arr = arr.astype(np.float32)
        lo = float(np.percentile(arr, 1))
        hi = float(np.percentile(arr, 99))
        if hi <= lo:
            lo = float(arr.min())
            hi = float(arr.max())
        if hi > lo:
            arr = np.clip((arr - lo) / (hi - lo), 0.0, 1.0)
        else:
            arr = np.zeros_like(arr, dtype=np.float32)
        return Image.fromarray((arr * 255.0).astype(np.uint8), mode="L")

    return Image.fromarray(arr).convert("RGB")


def derive_image_locator(row: Dict[str, Any]) -> Dict[str, str]:
    train_folder = str(row.get("train_folder", "")).strip()
    if not train_folder:
        source_zip = str(row.get("source_zip", "")).strip()
        if source_zip.lower().endswith(".zip"):
            train_folder = source_zip[:-4]

    image_filename = str(row.get("image_filename", "")).strip()
    if not image_filename:
        image_filename = Path(str(row.get("image_id", "")).strip()).name

    return {
        "train_folder": train_folder,
        "image_filename": image_filename,
    }


def resolve_local_result_image(row: Dict[str, Any], retrieval_images_root: str) -> Optional[Path]:
    # If metadata already has a concrete local path, use it first.
    direct_path = Path(str(row.get("image_path", "")).strip())
    if str(direct_path) and direct_path.exists():
        return direct_path

    root_text = str(retrieval_images_root).strip()
    if not root_text:
        return None

    root = Path(root_text)
    locator = derive_image_locator(row)
    train_folder = locator["train_folder"]
    image_filename = locator["image_filename"]
    image_id = str(row.get("image_id", "")).strip()

    candidates: List[Path] = []
    if train_folder and image_filename:
        candidates.append(root / train_folder / image_filename)
    if image_filename:
        candidates.append(root / image_filename)
    if image_id:
        candidates.append(root / image_id)

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def artifact_paths(output_dir: str) -> Dict[str, str]:
    out = Path(output_dir)
    return {
        "embeddings": str(out / "final_embeddings.npy"),
        "metadata": str(out / "final_metadata.json"),
        "pca_model": str(out / "pca_model.pkl"),
        "faiss_index": str(out / "faiss.index"),
    }


def run_build_pipeline(config: Dict[str, Any]) -> Dict[str, Any]:
    out_paths = artifact_paths(config["output_dir"])
    Path(config["output_dir"]).mkdir(parents=True, exist_ok=True)

    metadata_df = load_padchest_metadata(
        csv_path=config["dataset_csv"],
        images_root=config["images_root"],
        image_id_col=config["image_id_col"],
        image_dir_col=config["image_dir_col"],
        labels_col=config["labels_col"],
        locations_col=config["locations_col"],
        translate_reports_to_english=config["translate_reports"],
        enable_report_translation_api_fallback=config["translation_api_fallback"],
        report_translation_model=config["translation_model"],
    )

    embeddings, valid_metadata_df, stats = extract_embeddings(
        metadata_df=metadata_df,
        batch_size=config["batch_size"],
        num_workers=config["num_workers"],
        device=config["device"],
    )

    pca, reduced_embeddings = fit_pca(embeddings=embeddings, n_components=config["pca_dim"])
    print_pca_variance(pca)
    save_pca_model(pca, out_paths["pca_model"])

    np.save(out_paths["embeddings"], reduced_embeddings)
    save_retrieval_metadata(
        metadata_df=valid_metadata_df,
        output_json=out_paths["metadata"],
    )

    index = build_index_flat_l2(reduced_embeddings)
    save_faiss_index(index, out_paths["faiss_index"])

    return {
        "stats": stats,
        "embedding_shape": tuple(reduced_embeddings.shape),
        "output_paths": out_paths,
        "faiss_ntotal": int(index.ntotal),
    }


def run_query_pipeline(
    output_dir: str,
    query_image: str,
    top_k: int,
    device: str,
    use_openai: bool,
    openai_model: str,
) -> Dict[str, Any]:
    out_paths = artifact_paths(output_dir)
    metadata_df = load_retrieval_metadata(out_paths["metadata"])
    index = load_faiss_index(out_paths["faiss_index"])

    results = search_similar_cases(
        query_image_path=query_image,
        faiss_index=index,
        retrieval_metadata=metadata_df,
        pca_model_path=out_paths["pca_model"],
        top_k=top_k,
        device=device,
    )

    context = build_rag_context(results)
    explanation = generate_explanation(
        context=context,
        use_openai_if_available=use_openai,
        openai_model=openai_model,
    )

    return {
        "results": results,
        "context": context,
        "explanation": explanation,
    }


def show_results(
    results: List[Dict[str, Any]],
    retrieval_images_root: str = "",
) -> None:
    st.subheader("Top Similar Cases")
    if not results:
        st.info("No similar cases returned.")
        return

    grid_cols = st.columns(3)
    for idx, row in enumerate(results):
        with grid_cols[idx % 3]:
            st.markdown(
                f"**Rank {row['rank']}**  \n"
                f"image_id: {row['image_id']}  \n"
                f"distance: {row['distance']:.4f}"
            )

            result_image = resolve_local_result_image(row=row, retrieval_images_root=retrieval_images_root)
            if result_image is not None:
                try:
                    preview = prepare_xray_for_display(result_image)
                    st.image(preview, caption=row["image_id"], use_column_width=True)
                except Exception as exc:
                    st.warning(f"Could not render retrieved image: {exc}")
            else:
                if str(retrieval_images_root).strip():
                    locator = derive_image_locator(row)
                    st.warning(
                        "Retrieved image file not found under local root. "
                        f"Expected train_folder={locator['train_folder']}, image_filename={locator['image_filename']}."
                    )
                else:
                    st.warning(
                        "Retrieved image file not found on disk. "
                        "Set 'Retrieved images root' in the sidebar."
                    )

            labels = row.get("labels", [])
            locations = row.get("locations", [])
            st.write("Labels:", ", ".join(labels) if labels else "None")
            st.write("Locations:", ", ".join(locations) if locations else "None")


def main() -> None:
    st.set_page_config(page_title="Chest X-ray Explainable RAG", layout="wide")
    st.title("Explainable Chest X-ray Retrieval (PadChest)")
    st.caption("Build embeddings + FAISS index, retrieve similar cases, and generate patient-friendly explanations.")

    with st.sidebar:
        st.header("Configuration")
        dataset_csv = st.text_input("Dataset CSV", value=DEFAULT_CSV)
        images_root = st.text_input("Images folder", value=DEFAULT_IMAGES_ROOT)
        output_dir = st.text_input("Artifacts output folder", value=DEFAULT_OUTPUT_DIR)
        retrieval_images_root = st.text_input(
            "Retrieved images root",
            value=DEFAULT_IMAGES_ROOT,
            help="Local folder containing image files or train subfolders like 0, 1, 2.",
        )

        st.markdown("---")
        st.subheader("CSV Columns")
        image_id_col = st.text_input("Image ID column", value="ImageID")
        image_dir_col = st.text_input("Image directory column", value="ImageDir")
        labels_col = st.text_input("Labels column", value="Labels")
        locations_col = st.text_input("Locations column", value="Localizations")
        translate_reports = st.checkbox("Translate report text to English", value=True)
        translation_api_fallback = st.checkbox("Use API fallback translation", value=False)
        translation_model = st.text_input("Translation model", value="gpt-4o-mini")

        st.markdown("---")
        st.subheader("Runtime")
        device = st.selectbox(
            "Device",
            options=["auto", "cpu", "cuda"],
            index=0,
            help="auto selects cuda if available.",
        )
        batch_size = st.number_input("Batch size", min_value=1, max_value=256, value=32, step=1)
        num_workers = st.number_input("Num workers", min_value=0, max_value=16, value=0, step=1)
        pca_dim = st.number_input("PCA dimensions", min_value=2, max_value=1024, value=256, step=1)

    runtime_device = None if device == "auto" else device
    if device == "auto":
        st.info(f"Auto device selected: {get_device(None)}")

    build_col, validate_col = st.columns(2)
    with build_col:
        if st.button("Build Artifacts", type="primary"):
            try:
                with st.spinner("Building embeddings, PCA, and FAISS index..."):
                    build_result = run_build_pipeline(
                        {
                            "dataset_csv": dataset_csv,
                            "images_root": images_root,
                            "output_dir": output_dir,
                            "image_id_col": image_id_col,
                            "image_dir_col": image_dir_col,
                            "labels_col": labels_col,
                            "locations_col": locations_col,
                            "translate_reports": bool(translate_reports),
                            "translation_api_fallback": bool(translation_api_fallback),
                            "translation_model": str(translation_model).strip() or "gpt-4o-mini",
                            "batch_size": int(batch_size),
                            "num_workers": int(num_workers),
                            "pca_dim": int(pca_dim),
                            "device": runtime_device,
                        }
                    )
                st.success("Build completed successfully.")
                st.json(build_result)
                st.session_state["last_build"] = build_result
            except Exception as exc:
                st.error(f"Build failed: {exc}")

    with validate_col:
        if st.button("Validate Artifacts"):
            try:
                out_paths = artifact_paths(output_dir)
                stats = validate_artifacts(
                    embeddings_path=out_paths["embeddings"],
                    metadata_path=out_paths["metadata"],
                    pca_model_path=out_paths["pca_model"],
                    faiss_index_path=out_paths["faiss_index"],
                )
                st.success("Artifact validation passed.")
                st.json(stats)
            except Exception as exc:
                st.error(f"Validation failed: {exc}")

    st.markdown("---")
    st.subheader("Query")
    query_col1, query_col2, query_col3 = st.columns([3, 1, 1])
    with query_col1:
        query_image = st.text_input("Query image path", value="")
    with query_col2:
        top_k = st.number_input("Top K", min_value=1, max_value=20, value=5, step=1)
    with query_col3:
        disable_openai = st.checkbox("Disable OpenAI", value=False)

    openai_model = st.text_input("OpenAI model", value="gpt-4o-mini")

    run_query_btn = st.button("Retrieve Similar Cases", type="primary")
    smoke_btn = st.button("Run Query Smoke Test")

    if run_query_btn:
        try:
            with st.spinner("Retrieving similar cases..."):
                payload = run_query_pipeline(
                    output_dir=output_dir,
                    query_image=query_image,
                    top_k=int(top_k),
                    device=runtime_device,
                    use_openai=(not disable_openai),
                    openai_model=openai_model,
                )
            show_results(payload["results"], retrieval_images_root=retrieval_images_root)
            st.subheader("RAG Context Summary")
            st.write(payload["context"]["summary_text"])
            label_notes = payload["context"].get("label_explanations", [])
            if label_notes:
                st.subheader("Label-wise Explanation")
                for item in label_notes:
                    st.markdown(f"- **{item['label']}**: {item['explanation']}")
            st.subheader("Final Explanation")
            st.write(payload["explanation"])
            if Path(query_image).exists():
                st.subheader("Query Image")
                try:
                    query_preview = prepare_xray_for_display(Path(query_image))
                    st.image(query_preview, use_column_width=True)
                except Exception as exc:
                    st.warning(f"Could not render query image: {exc}")
        except Exception as exc:
            st.error(f"Query failed: {exc}")

    if smoke_btn:
        try:
            out_paths = artifact_paths(output_dir)
            smoke = run_query_smoke(
                query_image=query_image,
                metadata_path=out_paths["metadata"],
                pca_model_path=out_paths["pca_model"],
                faiss_index_path=out_paths["faiss_index"],
                top_k=min(int(top_k), 3),
                device=runtime_device,
            )
            st.success("Smoke test completed.")
            st.json(smoke)
        except Exception as exc:
            st.error(f"Smoke test failed: {exc}")


if __name__ == "__main__":
    main()
