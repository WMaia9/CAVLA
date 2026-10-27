# Composição parcial e logs novos: a versão atual falhou no controle físico

Rodada de 21/10/2026. **Não promover esta versão como contribuição central
para ICML.** Os modelos geométricos venceram redes diretas, mas uma regra
geométrica sem treinar rede venceu os modelos aprendidos. A comparação anterior
estava incompleta sem essa baseline. O sistema fica como infraestrutura e
baseline para investigar um problema de aprendizagem mais exigente.

## O que foi executado

Foram treinados cinco modelos com apenas duas respostas por cena/curva:
a restrição composta original e a resposta após excluir um agente escolhido
previamente. Os rótulos individuais não entram na otimização. Comparamos
DeepSets, MLP+noisy-OR, decoder geométrico+noisy-OR e versões MLP/geométrica
com verossimilhança conjunta dos dois bits. Também treinamos o geométrico
conjunto sem features da VLA. São três divisões por gravação, três seeds,
6.000 updates e seleção/calibração exclusivamente nos pares ID da validação.

As labels simulam supervisão parcial a partir das anotações disponíveis; não
medem economia real de anotação. Excluir agentes significa retirar restrições
do modelo de movimento registrado. Não são imagens de remoções múltiplas nem
futuros reativos. Os modelos recebem a imagem original e a identidade dos
agentes a excluir. O teste mede generalização dessa operação algébrica.

O primeiro piloto encontrou seis detecções duplicadas da mesma instância 3D.
Corrigimos a composição para uma representante por instância e **retreinamos
todas as 54 variantes**. O piloto inicial de 54 treinos foi preservado como
substituído: **108 treinos nesta rodada, 54 usados na análise corrigida**.
Os 584 ROIs originais correspondem a 578 instâncias anotadas distintas por cena.
Ver [correção de duplicatas](ADENDO_DUPLICATAS_COMPOSICAO.md).

A PCA/normalização continuam usando as mesmas linhas originais de treino,
incluindo seis ROIs descartados da composição, em todos os métodos. Não usam
validação/teste. A supervisão e o pooling de agentes usam somente os 578 únicos.

## Protocolo e holdout

O [protocolo de composição](PROTOCOLO_COMPOSICAO_PARCIAL_V1.md) foi fixado antes
do piloto. O [adendo de controles](ADENDO_COMPOSICAO_PARCIAL_CONTROLES.md) e a
correção de duplicatas registram as mudanças posteriores ao início/resultado
inicial. Não chamamos a busca anterior de confirmação independente.

Reservamos logs fora das 210 cenas com pares/resultados explorados pelo projeto.
Os 16 logs reservados contêm 119 cenas; 112 têm os arquivos de entrada exigidos.
A associação 2D/3D e o futuro completo deixam 73 cenas, 164 agentes únicos e
16 logs. Para excluir dois/três agentes mantendo algum agente, restam **24
cenas em 13 logs**, com 5.928 casos de consulta OOD. Esses casos não são
observações independentes.

O [protocolo do holdout](PROTOCOLO_HOLDOUT_LOGS_COMPOSICAO.md) foi fixado antes
de gerar suas labels/features/previsões. Verificamos cena, sample, arquivo e
câmera CAM_FRONT em todas as 73 cenas. A diferença média RGB entre imagem
local e imagem fonte ficou abaixo de 0,97/255. Os nove preditores de cada
método foram avaliados sem retreino ou recalibração: cada um mantém PCA,
normalização e limiar da validação antiga.

É um teste em logs novos para o projeto, do mesmo nuScenes. Não sabemos se o
backbone viu esses dados no pretraining. A seleção nova inclui detecções sem
pares editados e, por isso, também muda o subconjunto de agentes. Essa diferença
de protocolo e o pequeno número de cenas compostas limitam a generalização.

## Resultados corrigidos

Erro balanceado em consultas com valores novos e exclusões duplas/triplas.
Menor é melhor. CV é média de três divisões/três seeds; holdout é média dos
nove preditores congelados sobre o mesmo conjunto novo.

| Método | CV por log | Logs novos |
|---|---:|---:|
| DeepSets | 25,79% | 28,08% |
| MLP + OR, BCE | 20,60% | 24,34% |
| MLP, verossimilhança conjunta | 20,42% | 22,90% |
| Geométrico + OR, BCE | 16,87% | 16,84% |
| Geométrico conjunto, features VLA | 16,77% | 17,13% |
| Geométrico conjunto, só metadados/priors | 19,59% | 16,52% |
| Referência com supervisão individual + OR | 18,30% | 18,48% |
| Referência com supervisão individual + máximo | 18,68% | 19,14% |
| Regra estacionária sem rede + OR, controle posterior | 13,96% | 9,16% |
| **Regra estacionária sem rede + máximo, controle posterior** | **13,88%** | **7,53%** |

As referências individuais reutilizam nove modelos antigos com rótulos e
seleção de checkpoint mais fortes; não são comparações de orçamento igual nem
limites superiores garantidos. A regra estacionária foi adicionada **depois**
de observar o resultado principal do holdout, como controle de falsificação.
Não deve ser apresentada como resultado confirmatório pré-registrado.

### O que os controles permitem concluir

Geometria vence o readout direto: no holdout, geométrico conjunto menos MLP+OR
tem diferença -7,21 pontos percentuais, intervalo descritivo de 95%
[-10,30; -3,90]. Contra DeepSets, a diferença é -10,95 pontos, intervalo
[-12,82; -5,43]. Isso sustenta uma vantagem da estrutura nesse alvo.

A loss conjunta **não venceu a BCE geométrica de forma clara**: diferença
na CV -0,10 ponto, intervalo [-1,59; +1,34]; no holdout +0,29 ponto,
intervalo [-1,88; +2,53]. Não há suporte para promovê-la como contribuição.

As features VLA ajudam na CV, mas **não confirmam ganho extra nos logs novos**:
geométrico conjunto com features menos sem features tem diferença +0,61 ponto,
intervalo [-3,67; +2,39]. Isso não prova ausência de informação na VLA;
rejeita a alegação de ganho demonstrado neste teste.

Os intervalos são bootstraps pareados de logs, 10.000 replicações, condicionados
aos preditores ajustados. Não incluem todo o retreino nem corrigem seleção de
hipóteses/múltiplas comparações. No holdout, todas as replicações usadas tiveram
ambas as classes. As seeds/folds dos preditores não são novos dados de teste.

### Peso das cenas

Cenas com mais agentes produzem mais combinações. A sensibilidade com peso
uniforme por cena mantém a vantagem geométrica sobre MLP+OR, mas muda os
valores: no holdout, 19,93% no geométrico conjunto, 24,48% no MLP+OR e 19,68%
no geométrico sem features. Não há vitória clara frente à BCE geométrica ou
às referências individuais nessa ponderação. O relato principal não deve
ocultar essa dependência do desenho de consultas.

## O controle decisivo: nenhuma rede aprendida

A regra usa somente caixa 2D, classe, câmera e os mesmos tamanhos fixos por
classe fornecidos a todos os modelos. Estima profundidade pela altura aparente,
fixa a caixa no tempo com yaw zero e calcula a distância às curvas consultadas.
Usa a mesma escala logística de 0,5 m do decoder inicial. OR e máximo são duas
regras fixas de composição; seus limiares são calibrados apenas nos pares da
validação antiga. Não há feature VLA, ajuste dos priors no holdout ou treino
de rede. **Há calibração de um limiar**, logo não alegamos ausência de qualquer
ajuste estatístico.

A fórmula foi comparada ao decoder geométrico inicial em 16 objetos × 84
consultas: diferença máxima de probabilidade 7,71 × 10^-7. Não é uma regra
que recebeu posições 3D verdadeiras como entrada.

Contra a regra com máximo, o modelo geométrico conjunto tem erro **9,60 pontos
maior**, intervalo condicional [+4,10; +12,48]. Contra OR, +7,97 pontos,
intervalo [+2,34; +10,14]. O controle é posterior e o holdout pequeno, mas o
resultado também é desfavorável na CV: 16,77% contra 13,88%/13,96%.

Portanto, a superioridade sobre redes diretas não demonstra que aprender esse
decoder era necessário. O pipeline anterior também precisava comparar com
checkpoint zero/inicialização, e não só com checkpoints após 250 updates.
Não faz sentido escalar a versão atual como contribuição ICML com base nesses
ganhos contra baselines aprendidas mais fracas.

## Diagnóstico do alvo

No holdout, 53,66% dos agentes se deslocam mais de 2 m no futuro anotado; não
é correto dizer que todos os agentes são estáticos. Entretanto, congelar a
**caixa 3D verdadeira em t+0,1 s** muda apenas 3,19% das labels compostas OOD,
com erro balanceado de 3,12% frente às labels de movimento registrado.

Esse último teste é um diagnóstico privilegiado, pois usa posição/tamanho
verdadeiros. Não é um competidor com as mesmas entradas. Ele mostra que a
grade e o predicado atuais frequentemente não exigem distinguir movimentos
futuros, mesmo quando os objetos se movem. Não prova que prever movimento seja
desnecessário no trânsito real.

## Verificação e decisão

Os 54 checkpoints corrigidos reproduziram exatamente probabilidades de
validação e teste. As auditorias verificaram instâncias únicas, álgebra das
labels, bounds, grupos disjuntos, PCA só no treino, checkpoint, limiares e
métricas. O holdout verificou hashes dos pesos e dos arquivos de limiar,
sem fitting novo. Foram aprovados 64 testes de código.

**Rejeitar esta versão como núcleo de paper.** Preservar a implementação como
baseline. A inferência parcial por grupos e modelos invariantes a permutação
têm precedentes: [group testing](https://arxiv.org/abs/1902.06002),
[regressão com estados individuais latentes](https://pmc.ncbi.nlm.nih.gov/articles/PMC3841014/)
e [Deep Sets](https://proceedings.neurips.cc/paper_files/paper/2017/hash/f22e4747da1aa27e363d86d40ff442fe-Abstract.html).
Nem a NLL conjunta nem compor caixas, isoladamente, estabelecem novidade.

O próximo problema precisa exigir informação além desses priors e justificar
um novo método de aprendizagem. Antes de outra loss, avaliar consultas que
tenham ligação com decisões, histórico temporal real e baselines estacionária/
cinemática com entradas comparáveis. Um subconjunto escolhido posteriormente
porque a baseline erra não basta como confirmação ou como contribuição.
Continua pendente demonstrar ganho dos pares de imagem, semântica/linguagem,
outro backbone, generalização externa e utilidade de ação. CARLA não foi usado.

## Artefatos atuais

- Dataset deduplicado: `artifacts/partial_composition_data_v2_20261021/`.
- Treinos: `artifacts/partial_composition_{learning,metadata}_v2_20261021/`.
- Referência mais supervisionada: `artifacts/composition_supervised_reference_v2_20261021/`.
- Auditoria/bootstrap CV: `artifacts/composition_audit_v2_20261021/`.
- Replay: `artifacts/composition_{replay,metadata_replay}_v2_20261021/`.
- Reserva/seleção: `artifacts/reserved_log_holdout_v2_20261021/` e
  `artifacts/composition_holdout_selection_v1_20261021/`.
- Geometria, objetos, dados e features novos:
  `artifacts/composition_holdout_{geometry,objects,data,features}_v1_20261021/`.
- Proveniência: `artifacts/composition_holdout_provenance_v1_20261021/`.
- Avaliação congelada/auditoria:
  `artifacts/composition_holdout_{eval,audit}_v1_20261021/`.
- Regra sem rede: `artifacts/composition_prior_reference_v2_20261021/`;
  verificação numérica em `composition_prior_reference_v1_20261021/zero_decoder_check.json`.
- Diagnóstico de movimento: `artifacts/composition_motion_{train,holdout}_v2_20261021/`.
- Testes: `artifacts/tests_composition_corrected_final_20261021.log`.

Os logs novos já foram avaliados; não podem ser reapresentados como holdout
intocado se o método for ajustado usando estes resultados.

Gráfico do controle: [PNG](plots/composition_hard_control_20261021.png) e
[PDF](plots/composition_hard_control_20261021.pdf).
