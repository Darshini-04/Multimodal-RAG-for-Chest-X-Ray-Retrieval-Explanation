from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from artifact_utils import list_part_dirs, load_json_records, parse_part_index, save_json_records


def _print_progress(message: str, percent: float) -> None:
    bounded = max(0.0, min(100.0, percent))
    print(f"[{bounded:6.2f}%] {message}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge PCA-reduced part artifacts into final retrieval artifacts.")
    parser.add_argument("--output_dir", type=str, default="artifacts", help="Root artifacts directory")
    args = parser.parse_args()

    part_dirs = list_part_dirs(args.output_dir, required_file="embeddings_pca.npy")
    if not part_dirs:
        raise FileNotFoundError(
            f"No part_* directories with embeddings_pca.npy found under output_dir={args.output_dir}"
        )

    part_dirs.sort(key=lambda p: parse_part_index(p.name))

    embedding_blocks: List[np.ndarray] = []
    final_metadata: List[Dict[str, Any]] = []
    expected_dim = -1

    total_parts = len(part_dirs)
    for idx, part_dir in enumerate(part_dirs, start=1):
        emb_path = part_dir / "embeddings_pca.npy"
        metadata_path = part_dir / "metadata.json"

        if not metadata_path.exists():
            raise FileNotFoundError(f"Missing metadata.json for {part_dir.name}")

        embeddings = np.load(emb_path).astype(np.float32)
        if embeddings.ndim != 2:
            raise ValueError(f"embeddings_pca.npy must be 2D in {part_dir.name}, got {embeddings.shape}")

        if expected_dim < 0:
            expected_dim = int(embeddings.shape[1])
        elif expected_dim != int(embeddings.shape[1]):
            raise ValueError(
                f"PCA embedding dimension mismatch in {part_dir.name}: "
                f"expected {expected_dim}, got {int(embeddings.shape[1])}"
            )

        metadata_records = load_json_records(metadata_path)
        if len(metadata_records) != int(embeddings.shape[0]):
            raise ValueError(
                f"Row mismatch in {part_dir.name}: embeddings_rows={int(embeddings.shape[0])}, "
                f"metadata_rows={len(metadata_records)}"
            )

        embedding_blocks.append(embeddings)
        final_metadata.extend(metadata_records)

        pct = 10.0 + 70.0 * (idx / total_parts)
        _print_progress(
            f"Merged {part_dir.name}: rows={int(embeddings.shape[0])} ({idx}/{total_parts})",
            pct,
        )

    final_embeddings = np.concatenate(embedding_blocks, axis=0)

    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    final_embeddings_path = output_root / "final_embeddings.npy"
    final_metadata_path = output_root / "final_metadata.json"

    np.save(final_embeddings_path, final_embeddings.astype(np.float32))
    save_json_records(final_metadata, final_metadata_path)

    _print_progress(f"Saved merged embeddings to {final_embeddings_path}", 90.0)
    _print_progress(f"Saved merged metadata to {final_metadata_path}", 100.0)

    print(f"Final embedding shape: {final_embeddings.shape}")
    print(f"Final metadata rows: {len(final_metadata)}")
    print("Merge complete.")


if __name__ == "__main__":
    main()
