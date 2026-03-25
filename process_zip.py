from __future__ import annotations

import argparse
import gc
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch

from artifact_utils import ensure_dir, find_extracted_images, parse_zip_index, part_dir_for_zip, save_json_records
from data_loader import parse_list_field
from embedding import extract_embeddings, get_device


def _print_progress(message: str, percent: float) -> None:
    bounded = max(0.0, min(100.0, percent))
    print(f"[{bounded:6.2f}%] {message}")


def _build_part_metadata(
    metadata_csv: str,
    image_name_to_path: Dict[str, Path],
    image_id_col: str,
    labels_col: str,
    locations_col: str,
    report_col: str,
) -> pd.DataFrame:
    csv_path = Path(metadata_csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"Metadata CSV not found: {metadata_csv}")

    source_df = pd.read_csv(csv_path)
    required_cols = [image_id_col, labels_col, locations_col]
    missing = [col for col in required_cols if col not in source_df.columns]
    if missing:
        raise ValueError(f"Missing required metadata columns: {missing}")

    if report_col not in source_df.columns:
        source_df[report_col] = ""

    image_ids = source_df[image_id_col].fillna("").astype(str).str.strip()
    filtered_df = source_df.loc[image_ids.isin(image_name_to_path.keys())].copy()

    if filtered_df.empty:
        raise ValueError("No metadata rows match images in this zip file.")

    rows: List[Dict[str, Any]] = []
    for _, row in filtered_df.iterrows():
        image_id = str(row[image_id_col]).strip()
        if not image_id:
            continue
        extracted_path = image_name_to_path.get(image_id)
        if extracted_path is None:
            continue

        report_value = row[report_col]
        report_text = "" if pd.isna(report_value) else str(report_value).strip()

        rows.append(
            {
                "image_id": image_id,
                "image_path": str(extracted_path),
                "labels": parse_list_field(row[labels_col]),
                "locations": parse_list_field(row[locations_col]),
                "report": report_text,
            }
        )

    metadata_df = pd.DataFrame(rows)
    if metadata_df.empty:
        raise ValueError("No usable metadata rows after zip filtering.")
    return metadata_df


def process_single_zip(args: argparse.Namespace) -> None:
    zip_file = Path(args.zip_path)
    if not zip_file.exists():
        raise FileNotFoundError(f"Zip file not found: {args.zip_path}")

    zip_index = parse_zip_index(args.zip_path)
    train_folder = str(zip_index)
    part_dir = part_dir_for_zip(args.zip_path, args.output_dir)
    ensure_dir(part_dir)

    print(f"Current zip: {zip_file.name}")
    print(f"Part output: {part_dir}")
    _print_progress("Starting zip processing", 0.0)

    with tempfile.TemporaryDirectory(prefix="padchest_zip_") as temp_dir:
        extract_root = Path(temp_dir)

        _print_progress("Extracting zip archive", 10.0)
        with zipfile.ZipFile(zip_file, "r") as archive:
            archive.extractall(extract_root)

        image_name_to_path = find_extracted_images(extract_root)
        total_images = len(image_name_to_path)
        if total_images == 0:
            raise ValueError("No supported image files found in zip archive.")

        _print_progress(f"Found {total_images} image files in zip", 25.0)

        metadata_df = _build_part_metadata(
            metadata_csv=args.metadata_csv,
            image_name_to_path=image_name_to_path,
            image_id_col=args.image_id_col,
            labels_col=args.labels_col,
            locations_col=args.locations_col,
            report_col=args.report_col,
        )

        _print_progress(f"Matched {len(metadata_df)} metadata rows", 40.0)
        runtime_device = get_device(args.device)
        print(f"Runtime device: {runtime_device}")

        embeddings, valid_metadata_df, stats = extract_embeddings(
            metadata_df=metadata_df,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=args.device,
        )

        valid_count = int(stats["valid_rows"])
        total_count = int(stats["total_rows"])
        skipped_count = int(stats["skipped_rows"])
        valid_pct = (100.0 * valid_count / total_count) if total_count else 0.0

        _print_progress(
            f"Embeddings generated: valid={valid_count}, skipped={skipped_count}, valid_pct={valid_pct:.2f}",
            80.0,
        )

        np.save(part_dir / "embeddings.npy", embeddings.astype(np.float32))
        np.save(part_dir / "image_ids.npy", valid_metadata_df["image_id"].astype(str).to_numpy())

        metadata_to_save = valid_metadata_df.copy()
        metadata_to_save["image_path"] = ""
        metadata_to_save["source_zip"] = zip_file.name
        metadata_to_save["train_folder"] = train_folder
        metadata_to_save["image_filename"] = metadata_to_save["image_id"].astype(str).apply(lambda value: Path(value).name)
        save_json_records(metadata_to_save.to_dict(orient="records"), part_dir / "metadata.json")

        _print_progress(f"Saved part artifacts with {valid_count} images", 100.0)

    # Temporary extraction directory is deleted automatically at this point.
    del embeddings
    del valid_metadata_df
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("Zip processing complete.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Process one PadChest zip file into raw embedding artifacts.")
    parser.add_argument("--zip_path", type=str, required=True, help="Path to a zip file, e.g., data/0.zip")
    parser.add_argument("--metadata_csv", type=str, required=True, help="Path to PadChest metadata CSV")
    parser.add_argument("--output_dir", type=str, default="artifacts", help="Artifacts output directory")
    parser.add_argument("--batch_size", type=int, default=32, help="Embedding inference batch size")
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader workers for embedding extraction")
    parser.add_argument("--device", type=str, default=None, help="Optional device override, e.g. cpu or cuda")
    parser.add_argument("--image_id_col", type=str, default="ImageID", help="Metadata CSV image id column")
    parser.add_argument("--labels_col", type=str, default="Labels", help="Metadata CSV labels column")
    parser.add_argument("--locations_col", type=str, default="Localizations", help="Metadata CSV locations column")
    parser.add_argument("--report_col", type=str, default="Report", help="Metadata CSV report column")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    process_single_zip(args)


if __name__ == "__main__":
    main()
