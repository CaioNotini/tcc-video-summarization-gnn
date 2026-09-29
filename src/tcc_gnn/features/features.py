"""Extrai features com torchvision (ResNet50, avgpool, 2048-dim)."""

from __future__ import annotations

from pathlib import Path
import numpy as np
import torch
from PIL import Image
from torchvision.models import ResNet50_Weights, resnet50


def _build_backbone() -> tuple[torch.nn.Module, "torch.nn.Module"]:
    """Monta a ResNet50 pré-treinada com a camada fc removida (saída = avgpool)."""
    weights = ResNet50_Weights.IMAGENET1K_V2
    model = resnet50(weights=weights)
    model.fc = torch.nn.Identity()
    model.eval()
    return model, weights.transforms()


def extract_features_torch(frame_dir: Path) -> np.ndarray:
    """Extrai as features (avgpool, 2048-dim) de todos os frames de frame_dir, em ordem."""
    model, preprocess = _build_backbone()

    frame_paths = sorted(Path(frame_dir).iterdir())
    features = []
    with torch.no_grad():
        for frame_path in frame_paths:
            img = Image.open(frame_path).convert("RGB")
            batch = preprocess(img).unsqueeze(0)
            feature_vector = model(batch)
            features.append(feature_vector.squeeze(0).numpy())

    return np.stack(features)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("frame_dir", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    feats = extract_features_torch(args.frame_dir)
    print(f"Features extraídas: shape {feats.shape}")

    if args.output is not None:
        np.save(args.output, feats)
        print(f"Salvo em {args.output}")