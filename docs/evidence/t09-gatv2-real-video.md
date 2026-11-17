```
Role: Evidence for T09. The GP block (Linear + GATv2 + residual + BN) runs untrained on the time-aware graph of the SumMe video Jumps, with exact locality and determinism checks on the CPU and a measurement forward on the GPU.
Entry: ../status.md
Related to: [status.md](../status.md), [phases.md](../phases.md), [t09-gatv2-spec.md](./t09-gatv2-spec.md), [decisions.md](../decisions.md), [architecture.md](../architecture.md)
```

# T09 evidence: GATv2 on the time-aware graph of a real video

Run on 2026-11-17, repository commit `ca5ab83` with the uncommitted T09 changes. The block and the
checks follow [t09-gatv2-spec.md](./t09-gatv2-spec.md). Hardware: NVIDIA GeForce GTX 1660 SUPER
6 GB, Intel Core i5-8400, Windows 10 Pro 10.0.19045, Python 3.12.10, torch 2.5.1+cu121
(CUDA 12.1), torch_geometric 2.8.1.

The block runs with untrained weights from seed 0, in `eval()` and without gradients.
Its output is not a result of the method (D-17 is open), and no mlflow run is made (D-12 note).

## Result

| Check                                                                     | Value                                                                |
| ------------------------------------------------------------------------- | -------------------------------------------------------------------- |
| Input                                                                     | 152 frames x 2048 (cached ResNet50 avgpool features), `delta_t = 32` |
| Graph                                                                     | 4216 undirected edges, `edge_index` of shape (2, 8432)               |
| Output shape (CPU and GPU)                                                | (152, 64)                                                            |
| Output finite (CPU and GPU)                                               | True and True                                                        |
| Parameters                                                                | 164,288                                                              |
| Attention: self pairs, shape                                              | 0, (8432, 4) (edges x heads)                                         |
| Attention: largest deviation of the per-node sum from 1                   | 4.8e-07                                                              |
| Locality (CPU, frame 76 + 1): nodes changed outside the window            | 0 (largest change 0.0)                                               |
| Locality (CPU): changed nodes equal to the window of 63 nodes             | True                                                                 |
| Determinism (CPU): block rebuilt with the same seed gives the same output | True                                                                 |
| GPU against CPU, same weights: largest output difference                  | 2.53e-07                                                             |
| GPU first call (after moving the block to the GPU)                        | 0.1622 s                                                             |
| GPU warm forward, mean of 10 after 2 warm-up calls                        | 1.409 ms (std 0.12 ms)                                               |
| GPU peak memory                                                           | 44.9 MB                                                              |

The script exits with `T09 CHECAGENS: OK`.

## GPU noise: why the exact checks run on the CPU

Same block weights, same input, frame 76 perturbed by +1, window of 63 nodes:

|     | Noise between two calls with the same input | Largest change outside the window | Nodes changed outside the window | Smallest change inside the window |
| --- | ------------------------------------------- | --------------------------------- | -------------------------------- | --------------------------------- |
| CPU | 0.0                                         | 0.0                               | 0                                | 7.758e-03                         |
| GPU | 5.96e-08                                    | 5.96e-08                          | 89                               | 7.758e-03                         |

On the GPU the change outside the window equals the noise between two identical calls, about
6e-8 in float32, caused by the atomic operations of the scatter sums in PyTorch Geometric. A first
run of the check with the criterion "change > 0" on the GPU reported all 152 nodes as changed for
this reason. On the CPU the noise is zero and the change outside the window is exactly zero, so
locality and determinism are checked there with an exact criterion (researcher's decision,
2026-11-17). Reproducibility of training on the GPU is a pending item for T13
([t09-gatv2-spec.md](./t09-gatv2-spec.md)).

The first-call time excludes the creation of the CUDA context, which happens when the block is
moved to the GPU. A first version of the script measured a single cold call of 1.59 s, which
included that start-up; it is not a forward time.

## Command and output (verbatim)

`python experiments/run_gatv2_real_video.py --json-out .cache/scratch/t09_jumps.json`

```
{
  "video": "Jumps",
  "n_frames": 152,
  "delta_t": 32,
  "n_undirected_edges": 4216,
  "edge_index_shape": [
    2,
    8432
  ],
  "input_shape": [
    152,
    2048
  ],
  "n_parameters": 164288,
  "seed": 0,
  "probe_frame": 76,
  "window_size": 63,
  "mode": "eval, torch.no_grad, untrained weights",
  "config_model": {
    "embedding_size": 64,
    "gat_heads": 4,
    "dropout": 0.2
  },
  "cpu": {
    "output_shape": [
      152,
      64
    ],
    "output_finite": true,
    "same_seed_rebuilt_identical": true,
    "repeat_noise": 0.0,
    "probe_outside_window_max_change": 0.0,
    "probe_outside_window_nodes_changed": 0,
    "probe_inside_window_min_change": 0.007758095860481262,
    "probe_changed_nodes_equal_window": true
  },
  "gpu": {
    "output_shape": [
      152,
      64
    ],
    "output_finite": true,
    "output_mean": 0.02521955966949463,
    "output_std": 0.196434885263443,
    "output_min": -0.5915120840072632,
    "output_max": 0.9177095890045166,
    "max_abs_diff_vs_cpu": 2.5331974029541016e-07,
    "attention_shape": [
      8432,
      4
    ],
    "attention_self_pairs": 0,
    "attention_min": 0.011833303608000278,
    "attention_max": 0.043696079403162,
    "attention_sum_max_abs_dev_from_1": 4.76837158203125e-07,
    "repeat_noise": 5.960464477539063e-08,
    "probe_outside_window_max_change": 5.960464477539063e-08,
    "probe_outside_window_nodes_changed": 89,
    "probe_inside_window_min_change": 0.0077580660581588745,
    "first_call_s": 0.1622,
    "warmup_forwards": 2,
    "timed_forwards": 10,
    "forward_mean_ms": 1.409,
    "forward_std_ms": 0.12,
    "peak_memory_mb": 44.9,
    "gpu": "NVIDIA GeForce GTX 1660 SUPER"
  },
  "torch": "2.5.1+cu121",
  "cuda": "12.1",
  "torch_geometric": "2.8.1",
  "python": "3.12.10",
  "processor": "Intel64 Family 6 Model 158 Stepping 10, GenuineIntel",
  "os": "Windows-10-10.0.19045-SP0",
  "commit": "ca5ab83-dirty"
}
T09 CHECAGENS: OK
```

## Unit tests

`pytest -v tests/test_gatv2.py`: 16 passed (`test_edge_index_calculado_a_mao`,
`test_mesmas_arestas_do_grafo_time_aware`, `test_um_frame_sem_arestas`,
`test_grafo_vazio_levanta_erro`, `test_nos_fora_de_0_a_n_levantam_erro`,
`test_self_loop_levanta_erro`, `test_configuracao_fiel_a_especificacao`,
`test_forma_e_valores_finitos`, `test_concat_false_mantem_a_dimensao_com_qualquer_numero_de_cabecas`,
`test_gradiente_chega_a_todos_os_parametros`, `test_deterministico_com_seed_fixa_em_eval`,
`test_atencao_so_sobre_vizinhos_e_soma_um`, `test_saida_de_um_no_so_depende_da_sua_janela`,
`test_frame_isolado_fica_so_com_o_residual`, `test_entradas_invalidas`, `test_bloco_lido_do_yaml`).
Full suite: 127 passed, 1 skipped (TensorFlow not installed).

## Result against the acceptance criterion

Tests check the shape and the finiteness of the outputs (and gradient flow, determinism and
locality), and the block runs on one real video with its output saved here. One video, one seed
and untrained weights are covered; nothing here measures the quality of the features.
