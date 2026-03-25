import argparse
import time
from pathlib import Path

import numpy as np

from data_loader import load_padchest_metadata, load_retrieval_metadata, save_retrieval_metadata
from embedding import extract_embeddings, get_device
from faiss_index import build_index_flat_l2, load_faiss_index, save_faiss_index
from pca import fit_pca, print_pca_variance, save_pca_model
from rag_explainer import build_rag_context, generate_explanation
from retrieval import search_similar_cases
from smoke_test import run_query_smoke, validate_artifacts


def default_artifact_paths(output_dir: str) -> dict:
    out = Path(output_dir)
    return {
        "embeddings": str(out / "final_embeddings.npy"),
        "metadata": str(out / "final_metadata.json"),
        "pca_model": str(out / "pca_model.pkl"),
        "faiss_index": str(out / "faiss.index"),
    }


def run_build(args: argparse.Namespace) -> None:
    start = time.time()
    paths = default_artifact_paths(args.output_dir)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    print("[1/5] Loading PadChest metadata...")
    metadata_df = load_padchest_metadata(
        csv_path=args.dataset_csv,
        images_root=args.images_root,
        image_id_col=args.image_id_col,
        image_dir_col=args.image_dir_col,
        labels_col=args.labels_col,
        locations_col=args.locations_col,
        translate_reports_to_english=args.translate_reports,
        enable_report_translation_api_fallback=args.translation_api_fallback,
        report_translation_model=args.translation_model,
    )
    print(f"Loaded rows: {len(metadata_df)}")

    print("[2/5] Extracting DenseNet121 embeddings...")
    device = get_device(args.device)
    print(f"Runtime device: {device}")
    embeddings, valid_metadata_df, stats = extract_embeddings(
        metadata_df=metadata_df,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=args.device,
    )
    print(f"Embedding shape: {embeddings.shape}")
    print(
        "Rows summary: "
        f"total={stats['total_rows']}, valid={stats['valid_rows']}, skipped={stats['skipped_rows']}"
    )

    print("[3/5] Fitting PCA and reducing dimensions...")
    pca, reduced_embeddings = fit_pca(embeddings=embeddings, n_components=args.pca_dim)
    print_pca_variance(pca)
    save_pca_model(pca, paths["pca_model"])

    print("[4/5] Saving embeddings and metadata...")
    np.save(paths["embeddings"], reduced_embeddings)
    save_retrieval_metadata(
        metadata_df=valid_metadata_df,
        output_json=paths["metadata"],
    )

    print("[5/5] Building and saving FAISS index...")
    index = build_index_flat_l2(reduced_embeddings)
    save_faiss_index(index, paths["faiss_index"])

    elapsed = time.time() - start
    print("Build completed.")
    print(f"Saved embeddings: {paths['embeddings']}")
    print(f"Saved metadata JSON: {paths['metadata']}")
    print(f"Saved PCA model: {paths['pca_model']}")
    print(f"Saved FAISS index: {paths['faiss_index']}")
    print(f"Total runtime: {elapsed:.2f}s")


def run_build_index(args: argparse.Namespace) -> None:
    paths = default_artifact_paths(args.output_dir)

    embeddings_path = Path(paths["embeddings"])
    if not embeddings_path.exists():
        raise FileNotFoundError(
            f"Missing merged embeddings file: {paths['embeddings']}. "
            "Run merge_artifacts.py first."
        )

    embeddings = np.load(embeddings_path)
    if embeddings.ndim != 2:
        raise ValueError(f"Merged embeddings must be 2D, got shape={embeddings.shape}")

    metadata_df = load_retrieval_metadata(paths["metadata"])
    if len(metadata_df) != int(embeddings.shape[0]):
        raise ValueError(
            "Merged metadata row count does not match merged embeddings rows: "
            f"metadata={len(metadata_df)}, embeddings={int(embeddings.shape[0])}"
        )

    index = build_index_flat_l2(embeddings)
    save_faiss_index(index, paths["faiss_index"])

    print("FAISS index build completed.")
    print(f"Embeddings source: {paths['embeddings']}")
    print(f"Metadata source: {paths['metadata']}")
    print(f"Saved FAISS index: {paths['faiss_index']}")
    print(f"Index vectors: {index.ntotal}")


def run_query(args: argparse.Namespace) -> None:
    paths = default_artifact_paths(args.output_dir)

    print("Loading retrieval artifacts...")
    metadata_df = load_retrieval_metadata(paths["metadata"])
    faiss_index = load_faiss_index(paths["faiss_index"])

    if faiss_index.ntotal != len(metadata_df):
        raise RuntimeError(
            f"FAISS index size mismatch: ntotal={faiss_index.ntotal}, metadata_rows={len(metadata_df)}"
        )

    print("Running query retrieval...")
    results = search_similar_cases(
        query_image_path=args.query_image,
        faiss_index=faiss_index,
        retrieval_metadata=metadata_df,
        pca_model_path=paths["pca_model"],
        top_k=args.top_k,
        device=args.device,
    )

    print("\nTop similar image IDs and metadata:")
    for item in results:
        print(f"- Rank {item['rank']} | image_id={item['image_id']} | distance={item['distance']:.4f}")
        print(f"  labels: {item['labels']}")
        print(f"  locations: {item['locations']}")

    context = build_rag_context(results)
    explanation = generate_explanation(
        context=context,
        use_openai_if_available=(not args.disable_openai),
        openai_model=args.openai_model,
    )

    print("\nRAG context summary:")
    print(context["summary_text"])
    print("\nFinal explanation:")
    print(explanation)


def run_validate(args: argparse.Namespace) -> None:
    paths = default_artifact_paths(args.output_dir)
    print("Running artifact validation...")
    stats = validate_artifacts(
        embeddings_path=paths["embeddings"],
        metadata_path=paths["metadata"],
        pca_model_path=paths["pca_model"],
        faiss_index_path=paths["faiss_index"],
    )
    print(
        "Validation passed: "
        f"rows={stats['rows']}, dim={stats['dim']}, faiss_ntotal={stats['faiss_ntotal']}"
    )

    if args.query_image:
        print("Running optional query smoke test...")
        smoke = run_query_smoke(
            query_image=args.query_image,
            metadata_path=paths["metadata"],
            pca_model_path=paths["pca_model"],
            faiss_index_path=paths["faiss_index"],
            top_k=args.top_k,
            device=args.device,
        )
        print(f"Smoke results count: {smoke['result_count']}")
        print(f"Smoke top IDs: {smoke['top_ids']}")
        print(f"Smoke summary: {smoke['summary']}")
        print(f"Smoke explanation: {smoke['explanation']}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Explainable Chest X-ray Retrieval-Augmented Generation (PadChest)."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_cmd = subparsers.add_parser("build", help="Build retrieval artifacts from PadChest metadata.")
    build_cmd.add_argument("--dataset-csv", type=str, required=True, help="Path to PadChest metadata CSV.")
    build_cmd.add_argument("--images-root", type=str, default="", help="Root directory for image files.")
    build_cmd.add_argument("--image-id-col", type=str, default="ImageID")
    build_cmd.add_argument("--image-dir-col", type=str, default="ImageDir")
    build_cmd.add_argument("--labels-col", type=str, default="Labels")
    build_cmd.add_argument("--locations-col", type=str, default="Localizations")
    build_cmd.add_argument(
        "--translate-reports",
        dest="translate_reports",
        action="store_true",
        default=True,
        help="Translate report text to English during build.",
    )
    build_cmd.add_argument(
        "--no-translate-reports",
        dest="translate_reports",
        action="store_false",
        help="Disable Spanish-to-English report translation during build.",
    )
    build_cmd.add_argument(
        "--translation-api-fallback",
        action="store_true",
        default=False,
        help="Use OpenAI API fallback to translate unresolved Spanish report text.",
    )
    build_cmd.add_argument(
        "--translation-model",
        type=str,
        default="gpt-4o-mini",
        help="OpenAI model used for translation API fallback.",
    )
    build_cmd.add_argument("--batch-size", type=int, default=32)
    build_cmd.add_argument("--num-workers", type=int, default=0)
    build_cmd.add_argument("--pca-dim", type=int, default=256)
    build_cmd.add_argument("--device", type=str, default=None, help="Optional device override, e.g. cuda or cpu.")
    build_cmd.add_argument("--output-dir", type=str, default="artifacts")
    build_cmd.set_defaults(func=run_build)

    build_index_cmd = subparsers.add_parser(
        "build-index",
        help="Build FAISS index from merged final embeddings and metadata.",
    )
    build_index_cmd.add_argument("--output-dir", type=str, default="artifacts")
    build_index_cmd.set_defaults(func=run_build_index)

    query_cmd = subparsers.add_parser("query", help="Query similar X-rays and generate explanation.")
    query_cmd.add_argument("--query-image", type=str, required=True, help="Path to query X-ray image.")
    query_cmd.add_argument("--top-k", type=int, default=5)
    query_cmd.add_argument("--device", type=str, default=None, help="Optional device override, e.g. cuda or cpu.")
    query_cmd.add_argument("--output-dir", type=str, default="artifacts")
    query_cmd.add_argument("--disable-openai", action="store_true", help="Force placeholder explanation.")
    query_cmd.add_argument("--openai-model", type=str, default="gpt-4o-mini")
    query_cmd.set_defaults(func=run_query)

    validate_cmd = subparsers.add_parser("validate", help="Validate artifacts and run optional smoke query.")
    validate_cmd.add_argument("--output-dir", type=str, default="artifacts")
    validate_cmd.add_argument("--query-image", type=str, default="", help="Optional query image for smoke retrieval.")
    validate_cmd.add_argument("--top-k", type=int, default=3)
    validate_cmd.add_argument("--device", type=str, default=None, help="Optional device override, e.g. cuda or cpu.")
    validate_cmd.set_defaults(func=run_validate)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
