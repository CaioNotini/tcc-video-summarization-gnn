"""Extrai features com torchvision (ResNet50 IMAGENET1K_V2, avgpool, 2048-dim, D-03), em batch e no device pedido."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision.models import ResNet50_Weights, resnet50

from tcc_gnn.features.frame_extraction import frame_paths

BATCH_SIZE = 16


def extract_features(frame_dir: Path, device: torch.device, batch_size: int = BATCH_SIZE) -> np.ndarray:
    """Features `(n_frames, 2048)` dos jpgs `NNNNNN.jpg` de `frame_dir`, em ordem de nome."""
    weights = ResNet50_Weights.IMAGENET1K_V2
    model = resnet50(weights=weights)
    model.fc = torch.nn.Identity()
    model.eval().to(device)
    preprocess = weights.transforms()

    paths = frame_paths(frame_dir)
    features: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(paths), batch_size):
            batch = torch.stack([preprocess(Image.open(p).convert("RGB")) for p in paths[start : start + batch_size]]).to(device)
            features.append(model(batch).cpu().numpy())
    return np.concatenate(features)


def cached_features(frame_dir: Path, features_path: Path, device: torch.device) -> np.ndarray:
    """Lê `features_path` se existir e for coerente com os frames (mesma contagem, não mais antigo); senão extrai e grava."""
    paths = frame_paths(frame_dir)
    if not features_path.exists():
        features = extract_features(frame_dir, device)
        np.save(features_path, features)
        return features
    features = np.load(features_path)
    if len(features) != len(paths):
        raise ValueError(f"{features_path}: {len(features)} features para {len(paths)} frames")
    if features_path.stat().st_mtime < max(p.stat().st_mtime for p in paths):
        raise ValueError(f"{features_path} é mais antigo que os frames de {frame_dir}: apague o .npy para reextrair")
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
