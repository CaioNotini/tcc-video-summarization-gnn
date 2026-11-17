"""Leitura de configs/default.yaml, a fonte única dos parâmetros dos experimentos."""

from __future__ import annotations

from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "default.yaml"
SUMME_VIDEOS = REPO_ROOT / "data" / "raw" / "summe" / "SumMe" / "videos"
SUMME_PROCESSED = REPO_ROOT / "data" / "processed" / "summe"
VSUMM_PROCESSED = REPO_ROOT / "data" / "processed" / "vsumm"
SUMME_FEATURES_FILE = "features_resnet50_avgpool.npy"  # em SUMME_PROCESSED/<vídeo>/, ao lado de frames/
VSUMM_FEATURES_FILE = "features_teste.npy"  # em VSUMM_PROCESSED/<vídeo>/, das validações T01 e T06
_SQLITE_PREFIX = "sqlite:///"


def load_config(path: Path = DEFAULT_CONFIG) -> dict:
    """Conteúdo do yaml como dicionário."""
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def tracking_config(config: dict, repo_root: Path = REPO_ROOT) -> tuple[str, str]:
    """`(tracking_uri, artifact_location)` do mlflow, com os caminhos relativos resolvidos a partir de `repo_root`."""
    cfg = config["experiment_tracking"]
    if not cfg["tracking_uri"].startswith(_SQLITE_PREFIX):
        raise ValueError(f"tracking_uri deve ser sqlite (mlflow 3.18 recusa o backend de arquivo): {cfg['tracking_uri']}")
    db_path = (repo_root / cfg["tracking_uri"][len(_SQLITE_PREFIX):]).resolve()
    artifact_root = (repo_root / cfg["artifact_root"]).resolve()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return _SQLITE_PREFIX + db_path.as_posix(), artifact_root.as_uri()
