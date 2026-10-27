# Restrições por objeto: resultado exploratório e decisão científica

Rodada de 21/10/2026. Existe um pré-candidato técnico concreto, mas ainda não
uma contribuição original estabelecida para ICML. Aprender um predicado de
proximidade com estrutura geométrica e features espaciais mostrou vantagem
contra controles com as mesmas entradas. Não demonstramos melhora da política,
identificação causal, segurança nem benefício específico dos pares editados.

## O que foi executado

Foi associado o conjunto local Counter-nuScenes às anotações de nuScenes. Entre
2.975 pares em 210 cenas, 953 objetos têm uma correspondência 3D automática
candidata. Após exigir 6,4 s de futuro sem extrapolação nem lacunas de interpolação
acima de 0,75 s, restam 584 objetos em 150 cenas. As correspondências não foram
validadas por anotadores independentes. A seleção favorece objetos anotados e
com futuro disponível; não representa todas as situações de direção.

O alvo binário de uma consulta é: a linha central da curva passa a até 2 m da
caixa orientada de um agente específico durante o horizonte? As caixas seguem
o futuro registrado. Isso exclui reação à ação do ego, footprint do ego,
regras de trânsito e admissibilidade dinâmica. É um predicado geométrico, não
ground truth de colisão ou segurança.

Foram construídas 49.056 consultas: 35.040 na grade de treino e 14.016 com novas
velocidades, curvaturas e horizontes. Estas últimas estão dentro dos intervalos
da grade original: trata-se de generalização a novas combinações/valores, não
extrapolação para condições de direção extremas. A unidade observacional não é
cada consulta: existem somente 150 cenas.

Extraímos novamente ROIs de objetos nas imagens originais pelo Alpamayo local,
nas camadas 8, 16, 24 e 36 conjuntamente, sem geração de texto ou ações. O pooling
pondera sobreposição fracionária com a grade visual de 10 × 18 tokens. O backbone
fica congelado. A redução a 16 componentes e a normalização são ajustadas apenas
nos objetos de treino de cada divisão; no estudo com validação, excluem também
os objetos da validação interna.

Todos os quatro modelos principais recebem caixa 2D, classe, calibração da
câmera, raios e os mesmos priors fixos de dimensões por classe. Os modelos
espaciais recebem adicionalmente as mesmas features originais dos objetos.
O readout direto prevê o rótulo com um MLP. O modelo estruturado prevê cinco
caixas latentes no tempo e calcula a proximidade da curva por geometria
diferenciável. Os rótulos binários de consulta são a única supervisão otimizada:
posições 3D reais não são entradas nem alvos de uma loss auxiliar.

O MLP direto maior tem 23.801 parâmetros na variante espacial; o modelo
geométrico correspondente tem 22.929. Ambos usam os mesmos minibatches por seed,
AdamW, learning rate 0,001, até 6.000 updates, três seeds e três divisões.

**Os pares editados não entram no objetivo do protótipo geométrico atual.**
Usamos seus objetos/cenas para construir o subconjunto e estudamos a qualidade
das edições separadamente. Logo, não foi demonstrado que Counter-nuScenes traz
ganho sobre treinar apenas nas imagens originais e anotações de nuScenes.

## Resultados e controles que mudaram a interpretação

Erro balanceado = média entre taxa de falsos negativos e taxa de falsos
positivos. Menor é melhor. Valores abaixo são médias de três divisões e três
seeds, nas consultas com valores não usados no treino.

| Regime por cena | Direto: metadados/priors | Direto: + features espaciais | Geométrico: metadados/priors | Geométrico: + features espaciais |
|---|---:|---:|---:|---:|
| 1.500 updates, último checkpoint | 17,59% | 18,68% | 16,98% | **15,73%** |
| 6.000 updates, MLP direto maior, último checkpoint | 19,55% | 21,35% | **18,76%** | 19,63% |
| 6.000, seleção por validação interna, limiar 0,5 | 18,26% | 19,13% | 17,87% | **16,52%** |
| Mesmo estudo, limiar calibrado na validação interna | 18,43% | 18,79% | 17,41% | **15,03%** |

Na segunda linha, a vantagem inicial das features sobre o direto com metadados desaparece.
Não se deve selecionar apenas o regime curto favorável. Esse sobreajuste
motivou a seleção de checkpoint por validação interna para todos os modelos.
O negrito na tabela destaca comparações de interesse, não um teste estatístico.

O checkpoint é escolhido pelo menor erro balanceado da validação a limiar 0,5,
medido a cada 250 updates. Depois, somente nesse checkpoint, escolhe-se o limiar
que minimiza o mesmo erro na validação. Não se usa rótulo de teste para essas
escolhas. Recarregamos os 36 checkpoints desse estudo; as probabilidades de teste
reproduziram exatamente os arquivos salvos, com erro máximo zero.

Na comparação calibrada por cena, a redução do modelo geométrico espacial é
3,76 pontos percentuais frente ao readout direto espacial e 3,40 pontos frente
ao direto com metadados. O bootstrap pareado de cenas, estratificado por divisão
(10.000 replicações, modelos fixos), dá intervalos descritivos de 95% de
[-5,54; -2,10] e [-5,43; -1,62] pontos, respectivamente. Não incorpora a busca
prévia de hipóteses, toda a incerteza de retreino ou dependência entre conjuntos
de treino de cross-validation. Não equivale a confirmação independente.

O tradeoff é relevante: comparado ao direto com metadados, o geométrico espacial
reduz falsos negativos de 24,22% para 12,89%, mas aumenta falsos positivos de
12,63% para 17,16%. Sua AUROC sobe de 0,8994 para 0,9153. Não há dominância em
todas as métricas, e falsos negativos deste predicado não são acidentes reais.

O readout do token global de trajetória e a variante com diferença original/
editado não superaram a baseline simples no primeiro teste: erros 18,96% e
22,96%, contra 17,06%. Esses resultados motivaram extrair features locais,
mas não provam ausência de informação útil em outras representações da VLA.

## Controle mais forte: divisão por gravação

Repetimos o estudo em três divisões por `log_token`: as 150 cenas pertencem a
50 gravações. Nenhuma gravação é compartilhada entre treino, validação e teste
dentro de uma divisão. As contagens de logs (treino/validação/teste) são
26/8/16, 21/11/18 e 25/9/16. Hashes determinísticos definem os grupos, sem
selecioná-los por resultado. Não garantimos separação geográfica entre logs.

São mais 36 modelos, com os mesmos quatro métodos, seeds, otimizador, MLP
direto maior, orçamento de 6.000 updates e regra de seleção/calibração.
Não modificamos hiperparâmetros para melhorar o resultado por gravação.

| Método | Erro balanceado, limiar 0,5 | Erro balanceado, limiar calibrado | AUROC |
|---|---:|---:|---:|
| Direto: metadados e priors | 18,98% | 18,98% | 0,8901 |
| Direto: mesmas entradas + features espaciais | 19,33% | 18,79% | 0,8970 |
| Geométrico: metadados e priors | 17,31% | 17,36% | 0,8961 |
| Geométrico: mesmas entradas + features espaciais | **14,42%** | **13,93%** | **0,9261** |

O ganho calibrado frente ao direto espacial é **4,87 pontos percentuais**,
ou 25,90% de redução relativa, e aparece nas três divisões. Frente ao geométrico
sem features, o ganho é 3,44 pontos. Isso dá suporte ao uso de features espaciais
nesse decoder, sem estabelecer superioridade do backbone sobre alternativas.

O bootstrap pareado de **logs** (10.000 replicações, estratificado por divisão,
modelos fixos) produz intervalo descritivo de 95% de [-6,42; -3,50] pontos
para a diferença geométrico espacial menos direto espacial. Frente ao direto
com metadados, o intervalo é [-7,33; -2,84]; frente ao geométrico sem features,
[-4,68; -2,20]. Valem as mesmas limitações do bootstrap por cena: protocolo
exploratório, reuso da amostra e ausência de retreino em cada replicação.

Contra o direto espacial, falsos negativos caem de 23,09% para 13,25%; falsos
positivos ficam em 14,50% e 14,60%, respectivamente. Contra o direto com
metadados, ainda existe tradeoff: 26,32% → 13,25% em falsos negativos e
11,63% → 14,60% em falsos positivos. Não chamamos isso de redução de acidentes.

O replay dos 36 checkpoints reproduziu exatamente todas as probabilidades.
A auditoria verificou métricas, grupos disjuntos, médias de PCA ajustadas
somente no treino e seleção do checkpoint pelo menor erro de validação.
O estudo completo desta rodada treinou **198 modelos pequenos**, somando os
regimes por cena e gravação, além da inferência do backbone congelado.

Esta é uma análise exploratória mais exigente na mesma amostra, não uma nova
confirmação independente. O resultado apoia priorizar o pré-candidato técnico.

## Controles de edição no modelo real

O fatorial usa quatro imagens: original, alteração local, alteração exterior e
alteração conjunta. O suporte local é definido pelos pixels que diferem no
composite disponível, não por uma máscara semântica perfeita. As quatro
condições satisfazem a identidade fatorial em pixels. A interação é medida
na trajetória como diferença vetorial entre efeitos locais sob dois contextos.

| Coorte / contexto | Cenas | Resposta local (m) | Resposta exterior (m) | Interação (m) |
|---|---:|---:|---:|---:|
| Coorte anterior, seed de linguagem 123, frontal com padding | 6 | 0,303 | 0,261 | 0,167 |
| Mesmas cenas, seed de linguagem 456 | 6 | 0,220 | 0,230 | 0,155 |
| Novas cenas, seed 123 | 12 | 0,446 | 0,691 | 0,454 |
| Coorte anterior, quatro frames frontais reais no passado | 6 | 0,297 | 0,138 | 0,238 |

Na coorte nova, a alteração exterior produz resposta maior que a local em 9/12
cenas. A resposta exterior cai com contexto temporal frontal real nas seis
cenas antigas; o valor de interação continua diferente de zero. Portanto,
nem todos os números do protocolo com padding se transferem ao vídeo real.
Isso motiva controle de edição e contexto, não prova causalidade física.

São 99 novas gerações de linguagem e 396 decodificações de trajetória nos três
novos estudos, com quatro seeds do decoder após cada geração. A repetição
original inicial reproduz exatamente a trajetória. Seeds de decoder não são
novas cenas independentes. As 150 extrações de features são prefills separados.

Somente imagens da câmera frontal estão disponíveis nesta cópia local.
Tentativas multiview não puderam executar por ausência dos arquivos laterais;
seus logs foram preservados. O controle temporal bem-sucedido usa **uma câmera
com quatro frames passados reais**, sem padding, com desvio temporal máximo
de 50 ms. Não é o protocolo nativo de quatro câmeras, nem uma remoção física
consistente ao longo do vídeo.

## Literatura e limite de novidade

As fontes primárias consultadas mostram sobreposição substancial:

- [FCSRL, ICML 2024](https://proceedings.mlr.press/v235/cen24b.html):
  representações orientadas à viabilidade já são estudadas.
- [CabiNet](https://cabinet-object-rearrangement.github.io/): aprendizagem de
  consultas de colisão e integração com planejamento já existem.
- [CausalDriveBench](https://arxiv.org/abs/2609.32157): avaliações contrafactuais
  em nuScenes já tornam insuficiente apresentar somente um novo benchmark.
- [SafeAlign-VLA](https://arxiv.org/abs/2605.19524) e
  [FailBank](https://arxiv.org/abs/2609.39820): sinais negativos/corretivos para
  adaptação de políticas também não são novos em termos gerais.
- [Learning Robot Safety from Sparse Human Feedback](https://stanfordmsl.github.io/conformal-safety-learning/):
  feedback binário e calibração conformal de regiões latentes de risco já foram
  combinados; adicionar conformal prediction padrão não resolve a novidade.
- [UniPred](https://unipred.github.io/): features de foundation models,
  aprendizagem de predicados e planejamento já aparecem em conjunto.
- [Similarity Is Not Logic](https://arxiv.org/abs/2607.23052): separar extração
  de evidência e execução externa de restrições também já aparece em VLMs.
  A aplicação desse princípio a direção ainda exige uma contribuição própria.

FCSRL, CabiNet, CausalDriveBench, SafeAlign-VLA e o trabalho sobre certificados
com variáveis latentes têm PDFs/textos locais em
`artifacts/research_geometry_20261021/`. Os três últimos itens da lista foram inspecionados
por abstracts e páginas dos autores; não alegamos equivalência de métodos
com base em uma leitura completa desses três.

## Decisão e critério para continuar

Priorizar **aprendizagem de restrições por objeto sob supervisão parcial e
intervenções imperfeitas**, usando o protótipo atual como baseline executável.
O resultado positivo autoriza uma investigação focalizada; não autoriza chamar
o bottleneck geométrico, isoladamente, de contribuição ICML.

Para promover o pré-candidato, precisamos mostrar ganho que venha das
intervenções e generalize para combinações/remoções não vistas, com qualidade
das edições verificada e incerteza sobre o que os rótulos não identificam.
Ainda faltam comparação com supervisão 3D direta e baseline geométrica sem
treino, outro backbone, imagens sem edição e mudanças de aparência, remoções
compostas independentes, custo de inferência e efeito na seleção de ações.

A separação por log/gravação foi executada e manteve o ganho. Mesmo esse
controle reaproveita as cenas exploradas; confirmação exigirá uma
nova amostra congelada, sem escolher método e protocolo olhando seus resultados.

Se os pares editados não trouxerem benefício além das imagens originais e dos
rótulos geométricos, abandonar a alegação de aprendizagem por intervenção.
Se a vantagem depender só do decoder geométrico conhecido, a conclusão será
um bom baseline de engenharia, insuficiente como tese central para ICML.

## Artefatos

- Dados geométricos atuais: `artifacts/nuscenes_geometry_v3_20261021/`.
- Dataset de consultas: `artifacts/geometric_learning_20261021/`.
- ROIs novas: `artifacts/object_features_20261021/`.
- Estudo com validação: `artifacts/structured_geometry_validation_20261021/`.
- Calibração e replay: `artifacts/structured_geometry_calibration_20261021/`.
- Bootstrap por cena: `artifacts/structured_geometry_bootstrap_20261021/`.
- Divisão por gravação: `artifacts/geometric_log_split_20261021/`.
- Estudo por gravação: `artifacts/structured_geometry_log_validation_20261021/`.
- Calibração e bootstrap por gravação:
  `artifacts/structured_geometry_log_{calibration,bootstrap}_20261021/`.
- Fatoriais: `artifacts/editor_factorial_{new12,language456,front_video}_20261021/`.
- Testes de código: 50 testes passaram; log em
  `artifacts/tests_research_20261021_final.log`.
- Gráfico: [geometric_candidate_20261021.png](plots/geometric_candidate_20261021.png).
- Gráfico do controle por gravação:
  [PNG](plots/geometric_log_candidate_20261021.png) e
  [PDF](plots/geometric_log_candidate_20261021.pdf).

Versões geométricas v1/v2 foram preservadas, mas estão marcadas como substituídas.
v2 tem os mesmos 584 objetos usados nas consultas; v3 acrescenta a calibração
da câmera para os priors comuns aos modelos estruturados e diretos.

## Reproduzir o controle principal

Usar uma pasta de saída nova. O dataset de divisão por gravação preserva
consultas/labels/features do original e muda somente os grupos. As dependências
existentes no ambiente do Alpamayo foram usadas sem modificá-lo.

```bash
export CAVLA_PYTHON=/mnt/datasets/Explainability/alpamayo/ar1_venv/bin/python
export PYTHONPATH=src OPENBLAS_NUM_THREADS=2 CUBLAS_WORKSPACE_CONFIG=:4096:8
"$CAVLA_PYTHON" -m cavla.structured_geometry_learning \
  --data artifacts/geometric_log_split_20261021 \
  --geometry artifacts/nuscenes_geometry_v3_20261021 \
  --spatial-features artifacts/object_features_20261021/features.npz \
  --out artifacts/reproduction_log_training \
  --methods direct_metadata direct_spatial geometry_metadata geometry_spatial \
  --steps 6000 --direct-width 136 --validation
"$CAVLA_PYTHON" -m cavla.geometric_calibration \
  --study artifacts/reproduction_log_training \
  --out artifacts/reproduction_log_calibration
"$CAVLA_PYTHON" -m cavla.geometric_bootstrap \
  --calibration artifacts/reproduction_log_calibration \
  --group-key log_token --out artifacts/reproduction_log_bootstrap
"$CAVLA_PYTHON" -m cavla.verify_geometric_study \
  --study artifacts/reproduction_log_training \
  --out artifacts/reproduction_log_training/verification.json
```
