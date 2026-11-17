"""Extrai features com torchvision (ResNet50 IMAGENET1K_V2, avgpool, 2048-dim, D-03), em batch e no device pedido."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision.models import ResNet50_Weights, resnet50


def extract_features(frame_dir: Path, device: torch.device, batch_size: int = 16) -> np.ndarray:
    """Features `(n_frames, 2048)` dos jpgs de `frame_dir`, em ordem de nome."""
    weights = ResNet50_Weights.IMAGENET1K_V2
    model = resnet50(weights=weights)
    model.fc = torch.nn.Identity()
    model.eval().to(device)
    preprocess = weights.transforms()

    frame_paths = sorted(p for p in Path(frame_dir).iterdir() if p.suffix == ".jpg")
    if not frame_paths:
        raise ValueError(f"Nenhum frame .jpg em {frame_dir}")
    features: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(frame_paths), batch_size):
            batch_paths = frame_paths[start : start + batch_size]
            batch = torch.stack([preprocess(Image.open(p).convert("RGB")) for p in batch_paths]).to(device)
            features.append(model(batch).cpu().numpy())
    return np.concatenate(features)


def cached_features(frame_dir: Path, features_path: Path, device: torch.device) -> np.ndarray:
    """Lê `features_path` se existir; senão extrai de `frame_dir` e grava o .npy."""
    if features_path.exists():
        return np.load(features_path)
    features = extract_features(frame_dir, device)
    np.save(features_path, features)
    return features


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("frame_dir", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    feats = extract_features(args.frame_dir, torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Features extraídas: shape {feats.shape}")
    if args.output is not None:
        np.save(args.output, feats)
        print(f"Salvo em {args.output}")
