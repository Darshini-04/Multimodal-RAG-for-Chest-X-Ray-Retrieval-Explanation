from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Sequence


PART_PATTERN = re.compile(r"^part_(\d+)$")
ZIP_STEM_PATTERN = re.compile(r"^(\d+)$")
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def parse_zip_index(zip_path: str) -> int:
    stem = Path(zip_path).stem
    match = ZIP_STEM_PATTERN.fullmatch(stem)
    if not match:
        raise ValueError(
            "Zip filename stem must be numeric for part mapping (example: 0.zip, 1.zip). "
            f"Got: {Path(zip_path).name}"
        )
    return int(match.group(1))


def part_dir_for_zip(zip_path: str, output_dir: str) -> Path:
    zip_index = parse_zip_index(zip_path)
    return Path(output_dir) / f"part_{zip_index}"


def parse_part_index(part_name: str) -> int:
    match = PART_PATTERN.fullmatch(part_name)
    if not match:
        raise ValueError(f"Invalid part directory name: {part_name}")
    return int(match.group(1))


def list_part_dirs(output_dir: str, required_file: str | None = None) -> List[Path]:
    root = Path(output_dir)
    if not root.exists():
        return []

    dirs: List[Path] = []
    for child in root.iterdir():
        if not child.is_dir():
            continue
        if not PART_PATTERN.fullmatch(child.name):
            continue
        if required_file and not (child / required_file).exists():
            continue
        dirs.append(child)

    dirs.sort(key=lambda item: parse_part_index(item.name))
    return dirs


def find_extracted_images(extract_root: Path) -> Dict[str, Path]:
    image_map: Dict[str, Path] = {}
    for file_path in extract_root.rglob("*"):
        if not file_path.is_file():
            continue
        if file_path.suffix.lower() not in IMAGE_SUFFIXES:
            continue

        image_name = file_path.name
        if image_name not in image_map:
            image_map[image_name] = file_path
    return image_map


def save_json_records(records: Sequence[Dict[str, Any]], out_path: Path) -> None:
    ensure_dir(out_path.parent)
    out_path.write_text(json.dumps(list(records), ensure_ascii=True, indent=2), encoding="utf-8")


def load_json_records(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")

    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, list):
        raise ValueError(f"Expected list of records in JSON file: {path}")
    return loaded
