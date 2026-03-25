from __future__ import annotations

import argparse
import gc
from pathlib import Path
from typing import Iterable, Tuple

import joblib
import numpy as np
from sklearn.decomposition import IncrementalPCA, PCA

from artifact_utils import list_part_dirs, parse_part_index


def _print_progress(message: str, percent: float) -> None:
    bounded = max(0.0, min(100.0, percent))
    print(f"[{bounded:6.2f}%] {message}")


def _iter_batches(num_rows: int, batch_size: int) -> Iterable[Tuple[int, int]]:
    for start in range(0, num_rows, batch_size):
        yield start, min(start + batch_size, num_rows)


def _collect_shape_stats(part_dirs: list[Path]) -> tuple[int, int]:
    total_rows = 0
    feature_dim = -1

    for part_dir in part_dirs:
        emb_path = part_dir / "embeddings.npy"
        emb = np.load(emb_path, mmap_mode="r")
        if emb.ndim != 2:
            raise ValueError(f"Embeddings must be 2D: {emb_path} shape={emb.shape}")

        if feature_dim < 0:
            feature_dim = int(emb.shape[1])
        elif feature_dim != int(emb.shape[1]):
            raise ValueError(
                f"Embedding feature dimension mismatch in {emb_path}. "
                f"Expected {feature_dim}, got {int(emb.shape[1])}."
            )

        total_rows += int(emb.shape[0])

    if total_rows <= 0 or feature_dim <= 0:
        raise ValueError("No embeddings found for PCA fitting.")

    return total_rows, feature_dim


def _fit_incremental_pca(
    part_dirs: list[Path],
    n_components: int,
    batch_size: int,
) -> IncrementalPCA:
    model = IncrementalPCA(n_components=n_components, batch_size=batch_size)

    total_parts = len(part_dirs)
    for idx, part_dir in enumerate(part_dirs, start=1):
        emb_path = part_dir / "embeddings.npy"
        emb = np.load(emb_path, mmap_mode="r")

        for start, end in _iter_batches(int(emb.shape[0]), batch_size):
            batch = np.asarray(emb[start:end], dtype=np.float32)
            model.partial_fit(batch)

        pct = 10.0 + 40.0 * (idx / total_parts)
        _print_progress(
            f"Incremental PCA partial_fit complete for {part_dir.name} ({idx}/{total_parts})",
            pct,
        )

    return model


def _fit_full_pca(part_dirs: list[Path], n_components: int) -> PCA:
    blocks: list[np.ndarray] = []
    total_parts = len(part_dirs)

    for idx, part_dir in enumerate(part_dirs, start=1):
        emb_path = part_dir / "embeddings.npy"
        block = np.load(emb_path).astype(np.float32)
        blocks.append(block)

        pct = 10.0 + 20.0 * (idx / total_parts)
        _print_progress(f"Loaded embeddings for full PCA from {part_dir.name}", pct)

    all_embeddings = np.concatenate(blocks, axis=0)
    model = PCA(n_components=n_components, random_state=42)
    model.fit(all_embeddings)

    del all_embeddings
    del blocks
    gc.collect()

    _print_progress("Full PCA fit complete", 45.0)
    return model


def _transform_and_save_parts(
    part_dirs: list[Path],
    model: PCA | IncrementalPCA,
    n_components: int,
    batch_size: int,
) -> None:
    total_parts = len(part_dirs)

    for idx, part_dir in enumerate(part_dirs, start=1):
        emb_path = part_dir / "embeddings.npy"
        emb = np.load(emb_path, mmap_mode="r")

        reduced = np.empty((int(emb.shape[0]), n_components), dtype=np.float32)
        for start, end in _iter_batches(int(emb.shape[0]), batch_size):
            batch = np.asarray(emb[start:end], dtype=np.float32)
            reduced[start:end] = model.transform(batch).astype(np.float32)

        np.save(part_dir / "embeddings_pca.npy", reduced)

        pct = 60.0 + 40.0 * (idx / total_parts)
        _print_progress(
            f"Saved embeddings_pca.npy for {part_dir.name} ({idx}/{total_parts})",
            pct,
        )

        del reduced
        gc.collect()


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit global PCA across part_* embeddings and transform each part.")
    parser.add_argument("--output_dir", type=str, default="artifacts", help="Root artifacts directory")
    parser.add_argument("--n_components", type=int, default=256, help="PCA output dimension")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["incremental", "full"],
        default="incremental",
        help="PCA fitting mode: incremental for low memory, full for in-memory fit.",
    )
    parser.add_argument("--batch_size", type=int, default=2048, help="Batch size for incremental fit/transform")
    args = parser.parse_args()

    part_dirs = list_part_dirs(args.output_dir, required_file="embeddings.npy")
    if not part_dirs:
        raise FileNotFoundError(
            f"No part_* directories with embeddings.npy found under output_dir={args.output_dir}"
        )

    part_dirs.sort(key=lambda p: parse_part_index(p.name))

    total_rows, feature_dim = _collect_shape_stats(part_dirs)
    n_components = min(int(args.n_components), total_rows, feature_dim)
    if n_components < 1:
        raise ValueError("Effective PCA n_components must be >= 1")

    print(f"Parts discovered: {len(part_dirs)}")
    print(f"Total embeddings rows: {total_rows}")
    print(f"Input dimension: {feature_dim}")
    print(f"Requested n_components: {args.n_components}")
    print(f"Effective n_components: {n_components}")

    _print_progress("Starting global PCA fit", 5.0)
    if args.mode == "incremental":
        model: PCA | IncrementalPCA = _fit_incremental_pca(
            part_dirs=part_dirs,
            n_components=n_components,
            batch_size=args.batch_size,
        )
    else:
        model = _fit_full_pca(part_dirs=part_dirs, n_components=n_components)

    model_path = Path(args.output_dir) / "pca_model.pkl"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    _print_progress(f"Saved PCA model to {model_path}", 55.0)

    _transform_and_save_parts(
        part_dirs=part_dirs,
        model=model,
        n_components=n_components,
        batch_size=args.batch_size,
    )

    print("Global PCA fit/transform complete.")


if __name__ == "__main__":
    main()
