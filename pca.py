from pathlib import Path
from typing import Tuple
import warnings

import joblib
import numpy as np
from sklearn.decomposition import PCA


def fit_pca(embeddings: np.ndarray, n_components: int = 256, random_state: int = 42) -> Tuple[PCA, np.ndarray]:
    if embeddings.ndim != 2:
        raise ValueError(f"Embeddings must be 2D, got shape {embeddings.shape}")

    n_samples, n_features = embeddings.shape
    max_components = min(n_samples, n_features)
    if max_components < 1:
        raise ValueError(
            f"Cannot fit PCA for embeddings with shape {embeddings.shape}. "
            "Need at least one sample and one feature."
        )

    if n_components > max_components:
        warnings.warn(
            f"Requested PCA n_components={n_components} exceeds maximum {max_components} "
            f"for this data (n_samples={n_samples}, n_features={n_features}). "
            f"Using n_components={max_components}.",
            RuntimeWarning,
            stacklevel=2,
        )
        n_components = max_components

    pca = PCA(n_components=n_components, random_state=random_state)
    reduced = pca.fit_transform(embeddings).astype(np.float32)
    return pca, reduced


def transform_with_pca(pca: PCA, embeddings: np.ndarray) -> np.ndarray:
    reduced = pca.transform(embeddings)
    return reduced.astype(np.float32)


def print_pca_variance(pca: PCA) -> None:
    explained = pca.explained_variance_ratio_
    cumulative = explained.cumsum()
    print(f"PCA components: {pca.n_components_}")
    print(f"Explained variance ratio (first 10): {explained[:10]}")
    print(f"Cumulative explained variance: {float(cumulative[-1]):.6f}")


def save_pca_model(pca: PCA, model_path: str) -> None:
    out_file = Path(model_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pca, out_file)


def load_pca_model(model_path: str) -> PCA:
    in_file = Path(model_path)
    if not in_file.exists():
        raise FileNotFoundError(f"PCA model not found: {model_path}")
    pca = joblib.load(in_file)
    return pca
