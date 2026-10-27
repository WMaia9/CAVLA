# Estado semântico, origem da representação e transferência

**Decisão desta rodada:** existe um sinal visual exploratório para distinguir o estado anotado de veículos quase parados. Esse sinal supera os controles geométrico, RGB/gradiente e um CLIP congelado específico. Ainda não há evidência suficiente de semântica específica do objeto, ganho de decisão ou método novo de nível ICML. Não aprovar um paper de classificação parked/stopped.

## Dados e hipótese

Reutilizamos as associações automáticas de caixas 2D/3D, as imagens originais de Counter-nuScenes e as anotações locais nuScenes. Na taxonomia nuScenes, `vehicle.stopped` descreve um veículo parado com intenção de mover-se; `vehicle.parked`, um veículo sem intenção imediata de mover-se. São proxies de anotação humana, não observações diretas da intenção mental do motorista.

O último atributo usado é de uma anotação timestampada antes ou no instante real da câmera. Isso evita usar uma anotação futura como entrada, mas não garante que o processo humano de anotação tenha sido realizado sem observar o futuro.

Seleção definida antes de contar os eventos: duas posições passadas recentes, velocidade estimada <=0,5 m/s, último atributo passado parked/stopped. Resultado:

| Conjunto | Agentes | Cenas | Logs | Parked | Stopped | Partidas >2 m em 6,4 s |
|---|---:|---:|---:|---:|---:|---:|
| Desenvolvimento | 240 | 106 | 42 | 179 | 61 | 9 |
| Avaliação anterior | 48 | 21 | 12 | 35 | 13 | 1 |

**Apenas um evento de partida na avaliação inviabiliza uma conclusão forte sobre antecipação de partida.** A tarefa executada foi decodificação do estado anotado. Ambos os conjuntos já haviam sido examinados em outras tarefas; nenhum é apresentado como confirmação independente.

Dados e procedência: `artifacts/semantic_attribute_data_v1_20261021/`. [Protocolo inicial](PROTOCOLO_ESTADO_SEMANTICO_V1.md), [controles posteriores e correção da hipótese sobre histórico](ADENDO_CONTROLES_ESTADO_SEMANTICO.md).

## Sondagem com controles

Todos os readouts recebem caixa 2D, classe, centro/velocidade/orientação/tamanho 3D passados. Comparamos logística e kernel RBF ridge. Features adicionais usam PCA16 e padronização ajustadas exclusivamente no treino de cada um dos três folds por log. Hiperparâmetros são escolhidos pela AUC da validação interna; threshold usa erro balanceado na mesma validação. O escore constante continua elegível. Nenhum peso do backbone foi treinado.

O CLIP é o `laion/CLIP-ViT-B-32-laion2B-s34B-b79K` já disponível localmente. Extraímos embeddings de crop do agente, imagem inteira e concatenação. O mapeamento dos pesos OpenCLIP para o encoder Transformers foi carregado estritamente e comparado com uma implementação direta das mesmas operações: diferença máxima de 4,30e-6 em duas entradas de teste. Nenhum ambiente compartilhado foi alterado. Esse é um controle visual pré-treinado específico, não uma comparação abrangente com todos os encoders ou com orçamento de pré-treino igual.

Resultados da **mesma família RBF**; média dos três modelos selecionados nos mesmos exemplos:

| Entrada adicional | AUC média de CV | AUC na avaliação anterior | Erro balanceado na avaliação |
|---|---:|---:|---:|
| Nenhuma, só dados numéricos | 0,694 | 0,637 | 40,04% |
| RGB/gradiente | 0,518 | 0,486 | 46,01% |
| Gaussian16 | 0,548 | 0,571 | 45,13% |
| CLIP crop | 0,624 | 0,685 | 36,96% |
| Média VLA de todos os ROIs da cena | 0,646 | 0,736 | 33,48% |
| ROI VLA original | 0,824 | 0,796 | 29,56% |
| ROI VLA com imagem preta | 0,650 | 0,588 | 47,58% |

A logística com ROI VLA tem AUC 0,810; todas as outras variantes, sem escolher retrospectivamente a melhor família, estão em `artifacts/semantic_state_encoder_audit_v1_20261021/summary.json`.

Bootstrap pareado de 10.000 reamostragens dos 12 logs, condicional aos modelos fixos, para diferenças de AUC do RBF VLA:

- vs numérico: +0,158, intervalo de 95% [0,059; 0,312];
- vs CLIP crop: +0,111, intervalo [0,022; 0,251];
- vs imagem preta no mesmo encoder: +0,208, intervalo [0,123; 0,349];
- vs média VLA da cena: +0,059, intervalo **[-0,012; 0,176]**.

Uma reamostragem sem ambas as classes é indefinida; 9.999 foram válidas. Esses intervalos não incluem incerteza de retreinamento nem correção por seleção de hipóteses ou múltiplos testes. Há apenas **uma cena com as duas classes na avaliação, com cinco agentes**. Portanto, a evidência não permite afirmar que a vantagem venha da semântica do objeto depois de controlar completamente o contexto da cena.

## A ablação resolveu uma hipótese de confundimento

Inicialmente suspeitamos que os ROIs pudessem codificar o histórico de movimento do ego, que também integra o prompt. Acrescentamos histórico completo e resumos físicos aos readouts numéricos e visuais; a vantagem ficou incerta. **Essa mudança de desempenho não identifica a origem do sinal**, pois também muda a dimensão e a estimação do readout.

A inspeção de `helper.create_message` revelou que os tokens de histórico ficam depois dos tokens de imagem. Com atenção causal, os ROIs não podem ler esse histórico posterior. Verificamos diretamente:

| Imagem | Histórico | Comparação das features |
|---|---|---|
| Original | Original | Referência |
| Original | Posições zero, rotações identidade | Diferença máxima da referência: **0,0**, nos 288 agentes |
| Preta | Original | Controle do mesmo encoder |
| Preta | Posições zero, rotações identidade | Diferença máxima do controle anterior: **0,0**, nos 288 agentes |

Na entrada instrumentada, o último token visual está na posição 2930; o primeiro token de histórico substituído está na 2933. A independência foi demonstrada para **esses tokens ROI de prefill**, não para tokens de raciocínio nem para o decoder da política. Os dois controles de imagem preta diferem somente por pooling nos ROIs, oferecendo uma referência para posição da caixa e constantes do modelo.

O sinal depende da imagem no protocolo testado. Isso não identifica efeito causal físico de uma intenção ou de um agente. Pré-treino, memorização, contexto de rua e estados visuais correlacionados continuam sendo explicações possíveis. A avaliação foi separada de experimentos anteriores por log, mas não há garantia de ausência de nuScenes no pré-treino do Alpamayo.

Artefatos: `semantic_blind_features_v1_20261021`, `semantic_zeroego_features_v1_20261021`, `semantic_blindzero_features_v1_20261021`; checagem exata em `semantic_ablation_inputs_v1_20261021/encoder_checks.json`. O caso imagem preta + histórico zerado reutiliza um único forward porque as entradas do modelo são idênticas; os 288 ROIs são calculados com suas próprias caixas. A igualdade exata com a extração separada por cena confirma essa reutilização.

## Teste de transferência: decodificar não bastou

Executamos uma regra simples, sem novo treino ou ajuste por erro futuro: se o readout prevê parked, usar posição estacionária; se prevê stopped, extrapolar velocidade constante. Os thresholds e modelos são os já selecionados por rótulos semânticos. Há controles que sempre usam velocidade constante, sempre congelam a posição, ou recebem o verdadeiro atributo passado.

ADE médio por log nos mesmos 48 agentes da avaliação:

| Previsor | ADE |
|---|---:|
| Velocidade constante | 0,380 m |
| Sempre estacionário | 0,306 m |
| Gate com atributo passado verdadeiro | 0,285 m |
| Gate com RBF VLA | 0,303 m |
| Gate com média VLA da cena | 0,302 m |
| Gate com CLIP crop | 0,304 m |

Diferença RBF VLA menos estacionário: **-0,0034 m**, intervalo condicional por log **[-0,197; +0,136] m**. Não há ganho convincente sobre o controle estacionário. No desenvolvimento, o estacionário também supera o gate VLA. Isso testa uma regra específica de previsão de outros agentes, não a política de direção; a escassez de partidas limita qualquer interpretação geral sobre utilidade de semântica.

Código e resultado: `src/cavla/semantic_motion_transfer.py`, `artifacts/semantic_motion_transfer_v1_20261021/`.

## Novidade e decisão

A literatura reforça por que o resultado de classificação não deve virar o paper:

- [LOKI, ICCV 2021](https://arxiv.org/abs/2108.08236) já combina estados/intenção e previsão de trajetórias de agentes.
- [Learning Road Scene-level Representations via Semantic Region Prediction, CoRL](https://proceedings.mlr.press/v205/xiao23b.html) já aprende representações semânticas para intenção e identificação de objetos de risco.
- [Act with Intent / INDI](https://arxiv.org/abs/2608.23478) já destila representação de intenção para o decoder de ações; acrescentar uma loss de intenção não estabelece novidade.
- [Encoded but Not in Control](https://arxiv.org/html/2610.06235v1) e [VLA-Trace](https://arxiv.org/abs/2605.30117) já distinguem informação decodificável de influência no comportamento.
- [CofactVLA](https://arxiv.org/html/2608.04396v1) já propõe mecanismos de deconfounding em features e ações. Uma projeção genérica de nuisance tampouco bastaria.

Foram consultadas as páginas/resumos primários de todos esses trabalhos e trechos de métodos/análises dos três últimos; não alegamos reprodução ou auditoria completa dos artigos.

**O que a rodada resolveu:** há sinal visual mensurável para uma distinção semântica nos dados locais; ele não é explicado pela posição do ROI ou pelo histórico posterior do ego. **O que continua aberto:** separar semântica do objeto e contexto da cena, obter número suficiente de mudanças relevantes de comportamento, e formular um método que faça essa informação melhorar decisões com supervisão contrafactual incompleta. Nenhum desses pontos está resolvido pelo readout.

Prioridade seguinte: testar informação e utilidade em cenas com agentes semanticamente distintos no mesmo contexto e com eventos de mudança suficientes, sem selecionar pelo erro do modelo. Os pares de remoção existentes podem ajudar a controlar conteúdo, mas não fornecem sozinhos a ação correta após a remoção. Manter os controles estacionário, velocidade constante, contexto compartilhado e encoder visual comum desde o início. Ainda não há método ICML validado.

Validação: **71 testes de código passaram**, incluindo testes novos de bootstrap por log com empates e comparação com reamostragem literal, recuperação de um sinal conhecido pela logística e tratamento de AUC indefinida. O forward do CLIP e as igualdades exatas das ablações foram checados adicionalmente nos artefatos. Log: `artifacts/tests_semantic_20261021.log`.
