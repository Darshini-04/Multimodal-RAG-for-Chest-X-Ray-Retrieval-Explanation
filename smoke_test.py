from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from data_loader import load_retrieval_metadata
from faiss_index import load_faiss_index
from pca import load_pca_model
from rag_explainer import build_rag_context, generate_explanation
from retrieval import search_similar_cases


def _assert_file_exists(path: str, label: str) -> None:
    if not Path(path).exists():
        raise FileNotFoundError(f"Missing {label}: {path}")


def validate_artifacts(
    embeddings_path: str,
    metadata_path: str,
    pca_model_path: str,
    faiss_index_path: str,
) -> Dict[str, Any]:
    """Validate build artifacts and return summary stats."""
    _assert_file_exists(embeddings_path, "embeddings file")
    _assert_file_exists(metadata_path, "metadata file")
    _assert_file_exists(pca_model_path, "PCA model")
    _assert_file_exists(faiss_index_path, "FAISS index")

    embeddings = np.load(embeddings_path)
    if embeddings.ndim != 2:
        raise ValueError(f"Embeddings must be 2D, got shape={embeddings.shape}")

    metadata_df = load_retrieval_metadata(metadata_path)
    pca = load_pca_model(pca_model_path)
    index = load_faiss_index(faiss_index_path)

    if len(metadata_df) != embeddings.shape[0]:
        raise ValueError(
            "Metadata row count does not match embedding rows: "
            f"metadata={len(metadata_df)}, embeddings={embeddings.shape[0]}"
        )

    if index.ntotal != embeddings.shape[0]:
        raise ValueError(
            "FAISS ntotal does not match embedding rows: "
            f"ntotal={index.ntotal}, embeddings={embeddings.shape[0]}"
        )

    if index.d != embeddings.shape[1]:
        raise ValueError(
            "FAISS dimension does not match embedding dimension: "
            f"faiss_d={index.d}, embeddings_d={embeddings.shape[1]}"
        )

    if hasattr(pca, "n_components_") and int(pca.n_components_) != embeddings.shape[1]:
        raise ValueError(
            "PCA components do not match reduced embedding dimension: "
            f"pca={int(pca.n_components_)}, embeddings_d={embeddings.shape[1]}"
        )

    return {
        "rows": int(embeddings.shape[0]),
        "dim": int(embeddings.shape[1]),
        "faiss_ntotal": int(index.ntotal),
    }


def run_query_smoke(
    query_image: str,
    metadata_path: str,
    pca_model_path: str,
    faiss_index_path: str,
    top_k: int = 3,
    device: Optional[str] = None,
) -> Dict[str, Any]:
    """Run a lightweight retrieval and explanation smoke test."""
    metadata_df = load_retrieval_metadata(metadata_path)
    index = load_faiss_index(faiss_index_path)

    results = search_similar_cases(
        query_image_path=query_image,
        faiss_index=index,
        retrieval_metadata=metadata_df,
        pca_model_path=pca_model_path,
        top_k=top_k,
        device=device,
    )
    context = build_rag_context(results)
    explanation = generate_explanation(context=context, use_openai_if_available=False)

    return {
        "result_count": len(results),
        "top_ids": [row["image_id"] for row in results],
        "summary": context["summary_text"],
        "explanation": explanation,
    }
