from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import DenseNet121_Weights, densenet121
from tqdm import tqdm


def get_device(explicit_device: Optional[str] = None) -> torch.device:
    if explicit_device:
        requested = str(explicit_device).strip().lower()
        if requested.startswith("cuda") and not torch.cuda.is_available():
            print("[device] CUDA requested but unavailable in this PyTorch build. Falling back to CPU.")
            return torch.device("cpu")
        return torch.device(explicit_device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_image_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def build_feature_extractor(device: Optional[Any] = None) -> nn.Module:
    model = densenet121(weights=DenseNet121_Weights.DEFAULT)
    feature_extractor = nn.Sequential(
        model.features,
        nn.ReLU(inplace=True),
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
    )
    feature_extractor.eval()

    runtime_device = device if device is not None else get_device()
    feature_extractor.to(runtime_device)
    return feature_extractor


class XRayDataset(Dataset):
    def __init__(self, metadata_df: pd.DataFrame, transform: transforms.Compose):
        self.metadata_df = metadata_df.reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.metadata_df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.metadata_df.iloc[idx]
        image_path = Path(str(row["image_path"]))

        if not image_path.exists():
            return {"ok": False, "idx": idx, "reason": "missing_file"}

        try:
            with Image.open(image_path) as img:
                image = img.convert("RGB")
                tensor = self.transform(image)
            return {"ok": True, "idx": idx, "tensor": tensor}
        except Exception:
            return {"ok": False, "idx": idx, "reason": "invalid_image"}


def collate_batch(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    valid = [item for item in batch if item["ok"]]
    invalid_indices = [item["idx"] for item in batch if not item["ok"]]

    if not valid:
        return {"tensors": None, "indices": [], "invalid_indices": invalid_indices}

    tensors = torch.stack([item["tensor"] for item in valid], dim=0)
    indices = [item["idx"] for item in valid]
    return {"tensors": tensors, "indices": indices, "invalid_indices": invalid_indices}


@torch.no_grad()
def extract_embeddings(
    metadata_df: pd.DataFrame,
    batch_size: int = 32,
    num_workers: int = 0,
    device: Optional[str] = None,
) -> Tuple[np.ndarray, pd.DataFrame, Dict[str, int]]:
    """
    Generate DenseNet121 embeddings for metadata rows.

    Returns:
    - embeddings: np.ndarray of shape [N_valid, 1024]
    - valid_metadata_df: DataFrame aligned with embeddings
    - stats: dict with total/valid/skipped counters
    """
    runtime_device = get_device(device)
    transform = get_image_transform()
    dataset = XRayDataset(metadata_df=metadata_df, transform=transform)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=(runtime_device.type == "cuda"),
        collate_fn=collate_batch,
    )

    model = build_feature_extractor(device=runtime_device)

    all_embeddings: List[np.ndarray] = []
    kept_indices: List[int] = []
    skipped_count = 0
    skipped_indices: List[int] = []

    progress = tqdm(loader, desc="Embedding images", unit="batch")
    for batch in progress:
        invalid_indices = batch["invalid_indices"]
        skipped_count += len(invalid_indices)
        skipped_indices.extend(invalid_indices)

        tensors = batch["tensors"]
        if tensors is None:
            continue

        tensors = tensors.to(runtime_device, non_blocking=True)
        features = model(tensors)
        all_embeddings.append(features.cpu().numpy().astype(np.float32))
        kept_indices.extend(batch["indices"])

    if not all_embeddings:
        missing_paths: List[str] = []
        if skipped_indices:
            seen_missing = set()
            for idx in skipped_indices:
                path_text = str(metadata_df.iloc[idx]["image_path"])
                if path_text in seen_missing:
                    continue
                if not Path(path_text).exists():
                    missing_paths.append(path_text)
                    seen_missing.add(path_text)
                if len(missing_paths) >= 5:
                    break

        detail = (
            "No valid image embeddings were generated. "
            f"Rows processed: {len(metadata_df)}, rows skipped: {skipped_count}. "
            "Check --images-root and CSV image path columns."
        )
        if missing_paths:
            detail += f" Example missing paths: {missing_paths}"
        raise RuntimeError(detail)

    embeddings = np.concatenate(all_embeddings, axis=0)
    valid_metadata_df = metadata_df.iloc[kept_indices].reset_index(drop=True)

    stats = {
        "total_rows": int(len(metadata_df)),
        "valid_rows": int(len(valid_metadata_df)),
        "skipped_rows": int(skipped_count),
    }
    return embeddings, valid_metadata_df, stats


@torch.no_grad()
def embed_single_image(
    image_path: str,
    model: nn.Module,
    transform: Optional[transforms.Compose] = None,
    device: Optional[str] = None,
) -> np.ndarray:
    img_file = Path(image_path)
    if not img_file.exists():
        raise FileNotFoundError(f"Query image not found: {image_path}")

    runtime_device = get_device(device)
    runtime_transform = transform if transform is not None else get_image_transform()

    with Image.open(img_file) as img:
        tensor = runtime_transform(img.convert("RGB")).unsqueeze(0)

    tensor = tensor.to(runtime_device)
    vector = model(tensor).cpu().numpy().astype(np.float32)
    return vector
