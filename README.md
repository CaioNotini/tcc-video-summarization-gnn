# TCC — Sumarização de Vídeo com GNN sobre Grafo Time-Aware

Trabalho de Conclusão de Curso (PUC Minas) + Iniciação Científica, orientação de
Zenilton Kleber Patrocínio Jr. Aplica uma GNN à sumarização de vídeo, partindo do
[HieTaSumm](https://github.com/IMScience-PPGINF-PucMinas/HieTaSumm-lib) (grafo
time-aware + hierarquia por MST) e adotando como método principal uma hierarquia
aprendida via Single-Link Clustering (SLC) pooling, inspirada no SLC-HELM da linha
de pesquisa do laboratório (ver [mHELMNet](https://github.com/IMScience-PPGINF-PucMinas/mHELMNet)).

Escopo completo, stack, estrutura e cronograma: ver o documento de planejamento
do projeto (link no board/projeto do Claude).

## Pipeline (resumo)

1. Grafo time-aware do vídeo (reaproveitado do HieTaSumm/Leonardo).
2. Extração de features por CNN pré-treinada.
3. Camada de atenção (GATv2) sobre o grafo base.
4. Expansão hierárquica: SLC pooling (método principal) e MST (comparação).
5. Processamento por multigrafo — módulos MGP, ramos temporal/hierárquico, estilo HELMNet/mHELMNet.
6. Leitura por escala (RGR) + atenção por quadro (VOGNet) para escolha de keyframes.
7. Montagem do resumo e avaliação no VSUMM (F-score, Kendall's τ, Spearman's ρ).

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

git submodule update --init --recursive  # baixa HieTaSumm-lib e mHELMNet em references/
```

`higra` e `torch-geometric` têm binários específicos por SO/CUDA — se a instalação
via pip falhar, checar as instruções oficiais de cada lib antes de reportar bug.

## Estrutura

```
configs/            # configs de experimento (YAML)
data/                # dados brutos/processados (gitignored)
docs/                # notas de decisão, ADRs
notebooks/           # exploração e prototipagem
references/          # submodules: HieTaSumm-lib, mHELMNet
src/tcc_gnn/
  graph/             # grafo time-aware
  features/          # extração de features
  hierarchy/         # MST baseline + SLC pooling
  models/            # GATv2, módulos MGP, leitura RGR, atenção VOGNet
  training/          # loops de treino não supervisionado
  evaluation/         # métricas vs. VSUMM
tests/               # pytest
experiments/         # scripts de treino/avaliação (tracked via mlflow)
reports/             # figuras e resultados
```

## Testes

```bash
pytest
```

## Referências principais

- Cardoso, Gomes, Guimarães, Patrocínio Jr. — *Hierarchical Time-Aware Approach for
  Video Summarization* (HieTaSumm), BRACIS 2023.
- Batisteli, Kenmochi, Bougleux, Lézoray, Guimarães, Patrocínio Jr. — *Single-Link
  Clustering Pooling in Hierarchical Multigraph* (SLC-HELM), IEEE GRSL 2026.
- Batisteli, Passat, Guimarães, Patrocínio Jr. — *Hierarchical layered multigraph
  network with scale importance estimation* (HELMNet), Applied Soft Computing 2025.

Lista completa de trabalhos relacionados no documento de planejamento do projeto.
