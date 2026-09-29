"""Extração de features via CNN pré-treinada
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Model

_AVAILABLE_MODELS = {
    "vgg16": {"layer": "fc2", "img_size": (224, 224)},
    "resnet50": {"layer": "predictions", "img_size": (224, 224)},
}


def _build_backbone(model_name: str) -> tuple[Model, tuple[int, int]]:
    """Monta o modelo truncado até a camada de features."""
    if model_name not in _AVAILABLE_MODELS:
        raise ValueError(
            f"Modelo '{model_name}' não suportado. Use um de {list(_AVAILABLE_MODELS)}."
        )

    if model_name == "vgg16":
        base_model = VGG16(weights="imagenet", include_top=True)
    else:
        base_model = ResNet50(weights="imagenet", include_top=True)

    layer_name = _AVAILABLE_MODELS[model_name]["layer"]
    img_size = _AVAILABLE_MODELS[model_name]["img_size"]
    feature_model = Model(
        inputs=base_model.input, outputs=base_model.get_layer(layer_name).output
    )
    return feature_model, img_size


def extract_features_tf(frame_dir: Path, model_name: str = "resnet50") -> np.ndarray:
    feature_model, img_size = _build_backbone(model_name)

    frame_paths = sorted(Path(frame_dir).iterdir())
    features = []
    for frame_path in frame_paths:
        img = tf.keras.utils.load_img(frame_path, target_size=img_size)
        img_array = np.expand_dims(np.array(img), axis=0)
        processed = preprocess_input(img_array)
        feature_vector = feature_model.predict(processed, verbose="0")
        features.append(feature_vector.squeeze(axis=0))  # remove a dimensão de batch

    return np.stack(features)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("frame_dir", type=Path, help="Pasta com os frames extraídos")
    parser.add_argument(
        "--model",
        default="resnet50",
        choices=list(_AVAILABLE_MODELS),
        help="Modelo pré-treinado a usar (padrão: resnet50)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Caminho pra salvar as features em .npy (opcional)",
    )
    args = parser.parse_args()

    feats = extract_features_tf(args.frame_dir, model_name=args.model)
    print(f"Features extraídas: shape {feats.shape}")

    if args.output is not None:
        np.save(args.output, feats)
        print(f"Salvo em {args.output}")