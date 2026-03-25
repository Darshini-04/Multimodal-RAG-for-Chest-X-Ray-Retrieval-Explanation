from pathlib import Path

import faiss
import numpy as np


def build_index_flat_l2(embeddings: np.ndarray) -> faiss.IndexFlatL2:
    if embeddings.ndim != 2:
        raise ValueError(f"Embeddings must be 2D, got shape {embeddings.shape}")

    vectors = np.ascontiguousarray(embeddings.astype(np.float32))
    dim = vectors.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(vectors)
    return index


def save_faiss_index(index: faiss.Index, index_path: str) -> None:
    out_file = Path(index_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(out_file))


def load_faiss_index(index_path: str) -> faiss.Index:
    in_file = Path(index_path)
    if not in_file.exists():
        raise FileNotFoundError(f"FAISS index not found: {index_path}")
    return faiss.read_index(str(in_file))
