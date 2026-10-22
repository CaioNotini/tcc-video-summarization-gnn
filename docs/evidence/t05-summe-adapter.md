```
Role: Evidence record for T05. Saves the real output of the test suite for the SumMe adapter.
Entry: docs/status.md
Related to: [status.md](../status.md), [phases.md](../phases.md), [decisions.md](../decisions.md)
```

# T05 evidence: SumMe adapter

Run on 2026-10-22, repository commit `da3158840992aa3b32c2bbc09ced56055eff8aeb`.

Environment: Python 3.12.6, h5py 3.14.0, numpy 1.26.2, Windows 11.

Scope: only the SumMe adapter (`src/tcc_gnn/data/base.py`, `src/tcc_gnn/data/summe.py`).
No MST hierarchy (T06), no HieTaSumm baseline (T07), no F-score. `references/` was not
modified.

Command: `python -m pytest -v`

```
============================= test session starts =============================
platform win32 -- Python 3.12.6, pytest-7.4.3, pluggy-1.6.0 -- C:\Users\caion\AppData\Local\Programs\Python\Python312\python.exe
cachedir: .pytest_cache
rootdir: C:\Estudos\Faculdade\Trabalhos\TCC\tcc-video-summarization-gnn
configfile: pyproject.toml
testpaths: tests
plugins: anyio-3.7.1, asyncio-0.21.1, cov-4.1.0
asyncio: mode=Mode.STRICT
collecting ... collected 8 items

tests/test_frame_graph.py::test_calc_window_end_dentro_dos_limites PASSED [ 12%]
tests/test_frame_graph.py::test_edge_weight_features_identicas_da_zero PASSED [ 25%]
tests/test_frame_graph.py::test_build_time_aware_graph_com_dados_falsos PASSED [ 37%]
tests/test_frame_graph.py::test_build_time_aware_graph_com_video_real_do_summe PASSED [ 50%]
tests/test_smoke.py::test_package_imports PASSED                         [ 62%]
tests/test_summe_adapter.py::test_list_videos_retorna_os_25_videos_em_ordem_numerica PASSED [ 75%]
tests/test_summe_adapter.py::test_load_ground_truth_video_1_bate_com_o_arquivo_do_submodulo PASSED [ 87%]
tests/test_summe_adapter.py::test_load_ground_truth_video_inexistente_levanta_erro PASSED [100%]

============================== 8 passed in 3.97s ==============================
```

## Result

The three new tests in `tests/test_summe_adapter.py` pass, together with the five
pre-existing tests (no regression). `test_load_ground_truth_video_1_bate_com_o_arquivo_do_submodulo`
reads `video_1` directly from
`references/HieTaSumm-lib/HieTaSumm/eccv16_dataset_summe_google_pool5.h5` with `h5py` and
asserts the adapter's `GroundTruth` matches those raw values field by field: `n_frames`
(4494), `keyframes` (the `user_summary` matrix, shape `(15, 4494)`, values in `{0, 1}`),
`scores.values` and `scores.frame_indices` (`gtscore` and `picks`, shape `(300,)` each,
no padding or interpolation), and `change_points` (shape `(30, 2)`).
`test_list_videos_retorna_os_25_videos_em_ordem_numerica` checks the adapter lists all 25
SumMe videos in numeric order (`video_1` .. `video_25`). A negative case
(`test_load_ground_truth_video_inexistente_levanta_erro`) checks that an unknown video id
raises `KeyError`.
