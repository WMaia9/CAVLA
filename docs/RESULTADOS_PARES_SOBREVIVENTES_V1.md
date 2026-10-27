# Desenvolvimento do candidato: pares, controles visuais e composição

Atualização posterior: a [composição aprendida e os logs novos](RESULTADOS_COMPOSICAO_PARCIAL_E_HOLDOUT.md)
foram avaliados. Uma regra física sem rede venceu a versão atual. Os 584
objetos deste relatório são ROIs/detecções; a auditoria identificou 578
instâncias distintas por cena. Este documento preserva o resultado da etapa
anterior, não o veredito atual de candidatura.

21/10/2026. Foram executados mais 45 treinos pequenos: 36 para testar usos
simples dos pares editados e 9 para um controle visual sem pré-treino.
**O decoder geométrico com features da VLA continua promissor; a hipótese de
que uma loss simples nos pares acrescentaria ganho não recebeu suporte.**
O projeto continua exploratório, sem novidade ICML ou melhora de direção
demonstrada.

## Piloto de sobreviventes

O [protocolo](PROTOCOLO_PARES_SOBREVIVENTES_V1.md) foi congelado antes de extrair
features editadas e iniciar esses treinos; suas cópias e hashes foram preservados.
Selecionamos uma edição por cena, por ordem de hash independente do resultado,
com ao menos outro objeto cuja caixa não se sobrepõe à caixa alvo e cujo ROI
tenha no máximo 0,1% de pixels modificados. O filtro encontrou 368 objetos em
128 cenas; **todos os 368 ROIs selecionados têm, de fato, zero pixels alterados**.
Nenhuma das 128 edições é um no-op da imagem inteira. A fração mediana de pixels
alterados na imagem é 1,03%.

Esses objetos recebem o mesmo rótulo geométrico de consulta da imagem original.
O objeto alvo da edição não recebe rótulo de ausência. Isso evita assumir uma
remoção correta que ainda não foi validada. O experimento mede robustez do
predicado a alterações de contexto, mantendo o futuro registrado congelado;
não verifica preservação semântica nem reação física.

Foram extraídas features novas das 128 imagens editadas, nas mesmas quatro
camadas e ROIs do Alpamayo congelado. Uma repetição original reproduziu as
features com erro máximo zero. Todos os métodos usam as mesmas entradas,
geometria, capacidade, seeds 17/29/43, três divisões por gravação, minibatches
e 6.000 updates. A PCA de 16 dimensões usa apenas originais do treino, excluindo
validação e teste. O ruído tem a mesma norma por objeto que a diferença editada
no espaço PCA, mas direção isotrópica nova a cada update.

A baseline repete a original nas duas entradas. Augmentação usa BCE nas duas
imagens; consistência usa BCE original mais MSE entre probabilidades, com peso
1 fixado previamente. Para objetos não elegíveis, a entrada adicional é a
original. As repetições não são novos rótulos independentes. Checkpoint e limiar
são escolhidos somente nas consultas ID originais da validação interna por log.
Não se usam edições de validação/teste para essas escolhas.

### Resultados nas consultas novas

Erro balanceado menor é melhor. Médias de três divisões e três seeds.
As colunas de sobreviventes usam o mesmo subconjunto de objetos nas duas imagens.

| Treino | Originais: todos os 584 objetos | Originais: 368 sobreviventes | Edições: mesmos 368 sobreviventes |
|---|---:|---:|---:|
| Original repetida | 13,93% | 12,44% | 12,41% |
| Augmentação com edição | 14,16% | 13,12% | 13,22% |
| Ruído de mesma magnitude | 13,82% | 12,66% | 12,73% |
| Consistência pareada | 13,87% | 12,98% | 13,19% |

Não destacamos como vencedor o ruído por sua pequena diferença nas originais:
os intervalos descritivos incluem empate. No erro das originais, consistência
menos baseline é -0,055 ponto percentual, com intervalo condicional de 95%
[-0,934; +0,779]. Augmentação menos baseline é +0,236 ponto, intervalo
[-0,170; +0,764]. As variantes com pares não demonstram vantagem específica
frente ao controle de ruído.

Nas edições dos sobreviventes, augmentação tem erro 0,811 ponto maior que a
baseline, com intervalo [+0,030; +1,664]. Consistência tem diferença +0,786
ponto, intervalo [-0,366; +2,129]. Esses intervalos vêm de 10.000 reamostragens
pareadas de logs, estratificadas por divisão e condicionadas aos modelos
ajustados. Não corrigem busca de hipóteses/múltiplas comparações, não incluem
retreino e não representam confirmação independente.

A baseline já é estável neste filtro: 12,44% nas originais dos sobreviventes
e 12,41% nas edições; diferença -0,035 ponto, intervalo [-0,343; +0,332].
Isso enfraquece a motivação para impor mais invariância nesse subconjunto.

O resultado não depende apenas de calibrar limiares. A limiar fixo 0,5, os erros
nas originais são 14,42%, 14,45%, 14,03% e 14,49%, respectivamente; nas edições
dos sobreviventes, 12,77%, 13,36%, 13,02% e 13,39%.

### Auditoria

A nova baseline reproduziu exatamente todas as probabilidades dos nove modelos
correspondentes da rodada anterior. A auditoria dos 36 modelos verificou grupos
de logs disjuntos, média da PCA apenas no treino, índices de validação original,
escolha de checkpoint, optimalidade do limiar e métricas salvas. O erro máximo
de reconstrução das métricas foi zero. Foram aprovados 58 testes de código.

## Controle visual sem pré-treino

Construímos features de 2.048 dimensões usando RGB do crop do objeto em 16 × 16,
RGB da imagem inteira em 16 × 16 e gradientes de luminância do crop. Nenhuma
rede foi pré-treinada ou treinada para extrair essas features. A redução para
16 componentes, o decoder geométrico, as divisões, seeds, orçamento e regras de
seleção/calibração são os mesmos do estudo anterior.

| Entrada visual do decoder geométrico | Erro balanceado nas consultas novas |
|---|---:|
| Sem features visuais; metadados/priors | 17,36% |
| RGB/gradientes sem pré-treino | 16,34% |
| Features espaciais do Alpamayo | **13,93%** |

VLA menos pixels tem diferença -2,416 pontos percentuais, intervalo condicional
por log de 95% [-3,844; -0,909]. Replay dos nove modelos de pixels e métricas
reconstruídas têm erro máximo zero. A comparação dá suporte ao uso das features
congeladas neste teste, mas não mostra superioridade sobre DINO/CLIP, outro
encoder visual pré-treinado ou uma rede visual treinada com os mesmos dados.
O custo de inferência do backbone ainda deve ser medido.

## Composição e informação parcial

Implementamos um baseline de enumeração das atribuições de restrições por
objeto compatíveis com a resposta original e as remoções isoladas observadas.
Também implementamos limites inferior/superior para a resposta a uma remoção
composta nova, aceitando observações ausentes e detectando inconsistência com
as hipóteses exatas. Esses são limites de identificação sob o modelo OR, não
intervalos de confiança ou certificados de segurança.

Na auditoria de 12.600 consultas por cena, 3.854 são positivas e 1.061 (27,53%)
têm múltiplos agentes impondo a restrição. Dos 5.105 predicados positivos por
objeto, 2.312 (45,29%) não mudam a resposta composta quando somente seu agente
é retirado: outro ainda bloqueia a consulta. Os números consideram apenas os
584 agentes selecionados, a grade atual e futuros registrados congelados.
Não medem prevalência de redundância no trânsito nem efeitos observados da VLA.

O exemplo de três agentes `r=(1,1,0)` e `r=(0,1,1)` demonstra por enumeração que
original e todas as remoções isoladas podem ter respostas iguais, enquanto uma
remoção de dois agentes tem resposta diferente. A conexão é com
[group testing](https://arxiv.org/abs/1902.06002), já amplamente estudado;
não alegamos um teorema novo. A identificação por consulta ignora estrutura
geométrica entre consultas e, portanto, não é um limite de impossibilidade para
todo modelo de mundo. Veja a [formulação](FORMULACAO_RESTRICOES_COMPOSTAS.md).

Na revisão de novidade também encontramos
[CausalPlanner](https://ieeexplore.ieee.org/abstract/document/11425784), que já
combina identificação de elementos causais com supervisão fraca e restrições
contrastivas em planejamento. Foi lido o abstract primário disponibilizado na
busca; não alegamos equivalência de método com base em leitura completa.

## Decisão para o desenvolvimento

**Rejeitar augmentação/consistência simples de sobreviventes como argumento
central do paper.** Preservar o decoder geométrico e a pipeline espacial como
baseline executável. Priorizar treino com rótulos compostos parciais e teste em
combinações não observadas, comparando com OR/max, modelos de conjuntos e
enumeração/group testing. Mostrar separadamente o que as imagens pareadas
acrescentam; não pressupor que a mera presença delas estabelece causalidade.

Essa etapa ainda não foi treinada. A formulação e o baseline exato estão prontos;
os critérios e frentes pendentes estão no [plano do candidato](PLANO_CANDIDATO_RESTRICOES.md).
Nenhuma das edições múltiplas, adaptação de política, amostra independente ou
comparação com outro backbone foi demonstrada nesta rodada.

## Artefatos e reprodução

- Seleção e folha de revisão: `artifacts/survivor_pairs_v1_20261021/`.
- Features editadas: `artifacts/survivor_features_v1_20261021/`.
- 36 modelos: `artifacts/survivor_learning_v1_20261021/`.
- Auditoria/bootstrap atual: `artifacts/survivor_audit_v2_20261021/`.
  v1 foi preservada; v2 acrescenta a comparação consistência versus ruído.
- Controle visual e nove modelos: `artifacts/pixel_features_v1_20261021/`,
  `artifacts/pixel_geometry_v1_20261021/` e
  `artifacts/pixel_geometry_calibration_v1_20261021/`.
- Comparação pareada entre backbones: `artifacts/visual_backbone_comparison_v1_20261021/`.
  É um bundle de previsões já ajustadas; não representa outros treinos.
- Enumeração e auditoria geométrica: `artifacts/constraint_composition_bounds_v1_20261021/`.
- Testes: `artifacts/tests_survivors_final_20261021.log`.
- Comparações pareadas: [PNG](plots/survivor_pairs_v1_20261021.png) e
  [PDF](plots/survivor_pairs_v1_20261021.pdf).

```bash
export CAVLA_PYTHON=/mnt/datasets/Explainability/alpamayo/ar1_venv/bin/python
export PYTHONPATH=src OPENBLAS_NUM_THREADS=2 CUBLAS_WORKSPACE_CONFIG=:4096:8
"$CAVLA_PYTHON" -m cavla.survivor_learning \
  --reference-study artifacts/structured_geometry_log_validation_20261021 \
  --original-features artifacts/object_features_20261021/features.npz \
  --edited-features artifacts/survivor_features_v1_20261021/features.npz \
  --out artifacts/survivor_learning_reproduction
"$CAVLA_PYTHON" -m cavla.survivor_audit \
  --study artifacts/survivor_learning_reproduction \
  --out artifacts/survivor_audit_reproduction
```
