from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from embedding import build_feature_extractor, embed_single_image, get_image_transform
from pca import load_pca_model


def search_similar_cases(
    query_image_path: str,
    faiss_index: Any,
    retrieval_metadata: pd.DataFrame,
    pca_model_path: str,
    top_k: int = 5,
    device: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Embed query image, apply PCA, and search FAISS for nearest neighbors."""
    if top_k <= 0:
        raise ValueError("top_k must be > 0")

    if retrieval_metadata.empty:
        raise ValueError("Retrieval metadata is empty.")

    model = build_feature_extractor(device=device)
    transform = get_image_transform()
    query_embedding = embed_single_image(
        image_path=query_image_path,
        model=model,
        transform=transform,
        device=device,
    )

    pca = load_pca_model(pca_model_path)
    query_reduced = pca.transform(query_embedding).astype(np.float32)

    k = min(top_k, len(retrieval_metadata))
    distances, indices = faiss_index.search(query_reduced, k)

    results: List[Dict[str, Any]] = []
    for rank, (dist, idx) in enumerate(zip(distances[0], indices[0]), start=1):
        row = retrieval_metadata.iloc[int(idx)]
        results.append(
            {
                "rank": rank,
                "distance": float(dist),
                "image_id": str(row["image_id"]),
                "image_path": str(row["image_path"]),
                "report": str(row.get("report", "")),
                "train_folder": str(row.get("train_folder", "")),
                "image_filename": str(row.get("image_filename", "")),
                "source_zip": str(row.get("source_zip", "")),
                "labels": row["labels"] if isinstance(row["labels"], list) else [],
                "locations": row["locations"] if isinstance(row["locations"], list) else [],
            }
        )

    return results
