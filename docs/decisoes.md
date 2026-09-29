# Notas de decisão (ADR leve)

Registro curto de decisões técnicas tomadas ao longo do projeto. Uma entrada por decisão: contexto, decisão, alternativas
descartadas.

## 2026-09-22 — TensorFlow/Keras → torchvision na extração de features

**Contexto:** `HieTaSumm-lib` extrai features com
TensorFlow/Keras (VGG16/ResNet50). A GNN (PyTorch Geometric) já exige PyTorch.

**Decisão:** portar a extração de features para `torchvision` (ResNet50
pré-treinada ImageNet), evitando manter duas engines de deep learning como
dependência. Pendente de validação com o Zenilton.

**Alternativas descartadas:** manter TensorFlow só para essa etapa (mais
dependências, sem ganho técnico claro).

## 2026-09-22 — SLC pooling sem código de referência

**Contexto:** o SLC-HELM (Batisteli et al., 2026) é a base do método principal
(SLC pooling), mas o repositório não foi compartilhado — só o `mHELMNet`.

**Decisão:** implementar a camada de SLC pooling do zero a partir da descrição
do paper, usando os blocos de código do `mHELMNet` (`Linear`, `BatchNorm`,
`GraphSizeNorm`) como padrão de estilo.

**Acompanhar:** confirmar com o laboratório se o código do SLC-HELM fica
disponível mais adiante (reduziria retrabalho).

## 2026-09-29 — ResNet50: avgpool em vez de predictions na extração de features

**Contexto:** o `Models.py` do HieTaSumm-lib usa a camada `predictions` do ResNet50
(saída softmax de classificação, 1000-dim). Com a migração para torchvision,
era preciso escolher a camada de onde extrair as features, já que esse vetor
também vai virar a representação inicial de nó da GNN, alimentando o GATv2 e,
depois, o pooling SLC.

**Decisão:** usar a camada `avgpool` (global average pooling, 2048-dim,
pré-classificação) em vez de `predictions`. `predictions` é uma saída
softmax: satura, já que a rede comprime tudo em direção às classes
dominantes, então distâncias L1 entre frames com classes previstas diferentes
tendem a ficar quase todas no mesmo patamar, perdendo discriminação visual
fina. `avgpool` preserva um sinal mais rico (textura, cor, composição), é
prática padrão em sumarização de vídeo e retrieval, e é consistente com o
que HELMNet/mHELMNet consomem (embeddings de conv, não saída de
classificação).

**Alternativas descartadas:** manter `predictions` por fidelidade estrita ao
Leonardo (perderia sinal justamente na etapa que mais precisa dele, o
pooling hierárquico).
