"""Valida a porta vetorizada best_cut.py contra Summary.bestCutNumber (HieTaSumm-lib) em pastas de frames reais.

Para cada pasta de frames e delta_t: roda a referência (via reference_best_cut_number, submódulo
real) e a porta, compara o valor bruto e o cutNumber derivado, e reporta os tempos. Qualquer
diferença termina com código de saída 1.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "experiments"))

from validate_mst_hierarchy import reference_best_cut_number  # noqa: E402
from tcc_gnn.hierarchy.best_cut import best_cut_number, cut_number_from_best  # noqa: E402

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames-dirs", type=Path, nargs="+", required=True)
    parser.add_argument("--delta-t", type=int, nargs="+", default=[32])
    parser.add_argument("--alpha", type=int, default=75)
    args = parser.parse_args()

    all_equal = True
    for frames_dir in args.frames_dirs:
        for delta_t in args.delta_t:
            t0 = time.time()
            ref = reference_best_cut_number(frames_dir, delta_t, alpha=args.alpha)
            t_ref = time.time() - t0

            t0 = time.time()
            port_raw = best_cut_number(frames_dir, delta_t)
            t_port = time.time() - t0
            port_cut = cut_number_from_best(port_raw, alpha=args.alpha)

            equal = (
                port_raw == ref["best_cut_number_raw"] and port_cut == ref["cut_number_ref"]
            )
            all_equal = all_equal and equal
            print(
                f"{frames_dir.parent.name} delta_t={delta_t}: "
                f"ref raw={ref['best_cut_number_raw']} cut={ref['cut_number_ref']} [{t_ref:.1f}s] | "
                f"port raw={port_raw} cut={port_cut} [{t_port:.1f}s] | "
                f"{'IDENTICO' if equal else 'DIFERENTE'}"
            )

    sys.exit(0 if all_equal else 1)
