# Notas de decisão (ADR leve)

Registro curto de decisões técnicas tomadas ao longo do projeto, pra não perder o
raciocínio depois. Uma entrada por decisão: contexto, decisão, alternativas
descartadas.

## 2026-09-22 — TensorFlow/Keras → torchvision na extração de features

**Contexto:** `HieTaSumm-lib` (código do Leonardo) extrai features com
TensorFlow/Keras (VGG16/ResNet50). A GNN (PyTorch Geometric) já exige PyTorch.

**Decisão:** portar a extração de features para `torchvision` (ResNet50
pré-treinada ImageNet), evitando manter duas engines de deep learning como
dependência. Pendente de validação com o Zenilton — não deveria afetar a
arquitetura decidida, é só engine.

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
