# Pareceres consolidados do rascunho v0 (26/10/2026)

Pareceres completos: `reviewer2.md` (4/10, confiança 4), `reviewer_balanced.md` (4/10, confiança 4; estima ~7/10 com as ações abaixo), `reviewer_presentation.md`.

## Consenso dos três

- **A Tabela 1 bate com os dados.** Os dois revisores técnicos recalcularam todos os τ_eff e células a partir dos JSON. O núcleo open-loop é sólido.
- **O rascunho afirma como resultado o que ainda não foi rodado:** closed-loop, FlashDrive oficial em escala e a parte do Coda (que nunca foi testada na mesma temperatura; no LIBERO o efeito do Coda nem apareceu).
- **O título e o abstract exageram.** Solvers de ordem maior e o Qwen-Drive não se comportam como temperatura; o escopo "primeira ordem" precisa estar explícito.

## Problemas verificados por nós nos dados

1. **Colisões: pseudo-replicação.** As 12 colisões discordantes (Euler4 vs Heun32 τ=1) vêm de 5 cenas distintas (`context_127` responde por 6). Por cena distinta: 5:0, p≈0,06. O "0:2 na mesma temperatura" vem de 1 cena. A trajetória humana colide em 1 de 99 cenas (ruído de rótulo da mesma ordem). O Euler1 colide mais que o Euler2/4: menos temperatura não é monotonicamente mais seguro. **Ação:** tirar do abstract; reportar por cena; o efeito de segurança depende do closed-loop.
2. **"Parar" se move em direções opostas.** A temperatura explícita aumenta a parada (A1.5 3,7→4,2%; R1 1,7→2,2%; A2 1,7→2,5%, τ 1→0,4), e o Euler diminui (Euler1: 2,7 / 0,3 / 0,0%). Desacelerar anda junto nos dois. **Ação:** reformular a tese. O Euler é colapso para a média, que coincide com resfriamento nos segundos momentos e na maioria das decisões, mas apaga decisões raras e extremas que a temperatura explícita preserva.
3. **Resíduos sistemáticos não reportados.** O minADE₆ do Euler é pior que a curva de temperatura nas 6 células (+0,00 a +0,08 m); o resíduo de entropia de decisão é negativo; no Alpamayo 2 as inversões são 9,7% vs 7,7%. **Ação:** virar um Finding ("onde o Euler não é temperatura").
4. **Números de amostras diferentes misturados.** O Finding 5 usa 200 clipes × 2 seeds com referência Heun16/160; a Tabela 1 usa 100/60/60 clipes × 6 seeds com Heun32. **Ação:** uma única amostra e uma única referência para o paper.
5. **τ_eff do cache do FlashDrive:** 0,555 [0,531; 0,578] por dispersão; 0,50 é a razão de encolhimento. As duas medidas não coincidem para o cache.
6. **"99,95% de 408.769 inversões"** veio de um script ad hoc, não de um artefato salvo. **Ação:** salvar o artefato reproduzível ou retirar o número.
7. **400/400 encolhimento** só foi medido no Alpamayo 1.5. **Ação:** medir no R1, no A2 e no SmolVLA ou restringir a frase.

## Lista priorizada

| # | Ação | Tipo | Quando |
|---|---|---|---|
| 1 | Reformular a tese: colisões por cena; seção "onde o Euler não é temperatura" (parar, minADE₆, entropia); escopo de primeira ordem no título | texto + análise | semana 0 |
| 2 | Estatística: bootstrap por cena, IC pareado por célula, teste de equivalência com margem declarada, "fração explicada" | análise (CPU) | semana 0 |
| 3 | Predição teórica do τ_eff: a fórmula Gaussiana (∏cos θᵢ) prevê ≈0,83 para Euler10? Ajustar por modelo | teoria + análise | semana 0 |
| 4 | FlashDrive oficial em escala (rodando): separar fine-tuning, cache e temperatura | experimento | semanas 0–1 |
| 5 | Comparar distribuições inteiras (energy distance) com Heun-k como controle que falha | análise | semana 1 |
| 6 | Variação das seeds de linguagem | experimento (GPU leve) | semana 0–1 |
| 7 | Closed-loop v2 com cálculo de poder e margem de equivalência; v1 como piloto | experimento | semanas 1–4 |
| 8 | Unificar amostras e referência; salvar artefatos ad hoc | análise | semana 0 |
| 9 | Apresentação: compilar com pdflatex (Times), figura-teaser, figura principal, nomes únicos, referências sem TODO | texto | semanas 0–2 |
| 10 | Coda: testar na mesma temperatura ou retirar a afirmação | experimento ou corte | semana 1 |

---

# Rodada 2 (sobre v1) e o que entrou na v2 — 26/10/2026

Notas: Reviewer 2 **4/10** (mais honesto, mas novas afirmações não sobrevivem); equilibrado **4/10** (projeta ~6 com as ações 1–3 e ~7 com 4–5); apresentação: produção do PDF resolvida.

## Verificado por nós e corrigido na v2
- **"Euler apaga paradas" retirado:** quase todas as paradas vêm de 2–4 cenas com o carro (quase) parado (`context_113` a 0 m/s, `context_141` a 0,77 m/s). Com reamostragem conjunta entre modelos, o IC inclui zero. Na v2 vira observação exploratória com contagens (6/360 planos na referência).
- **Tese recentrada:** equivalência *distribucional* do Euler com temperatura (80–120% do efeito em ADE₁ e inversões), com **controle negativo Heun-3/5** (resíduo de inversões +16,5 e +9,1 pt) que mostra que o teste discrimina.
- **Equivalência reportada honestamente:** subdimensionada com 60 clipes; o texto não afirma mais "reproduzido".
- **Por ruído individual não há equivalência:** Euler-4 e A_τeff discordam em ~8–9% das decisões sob o mesmo ruído. O escopo foi explicitado (distribuição, não amostra).
- **Gaussiana:** acerta a mediana, não o ranking de clipes; uma constante faz igual. Rebaixada a explicação do valor típico.
- **Contração unificada** (60 clipes × 6 seeds, Heun32, espaço de trajetória): ~90–93% das amostras Euler encolhem (o "400/400" era espaço de ações e referência Heun160).
- **99,95% da mistura** também vale para Heun/RK4 no toy: não distingue primeira ordem; não é usado como evidência.
- Números unificados: artefato `artifacts/paper_numbers_v1_20261026`, macros `paper/tables/numbers.tex`, tabelas geradas; referências com autores verificados no arXiv.

## Pendências priorizadas (v3 em diante)
1. **Confirmação pré-registrada** em 436 clipes inéditos, enriquecida com cenas ambíguas e de parada, seeds novas; desfechos primários: equivalência com margens declaradas e o teste da cauda (paradas). ~2–4 dias de GPU por modelo, após o closed-loop e o FlashDrive.
2. **FlashDrive oficial** na mesma temperatura (rodando).
3. **Outras famílias:** ReCogDrive e DiffusionDrive no NAVSIM (download em andamento).
4. Heun-32 em τ ∈ {0,3; 0,2; 0,1} para cobrir Euler-1/2 (hoje fora da faixa).
5. Variação das seeds de linguagem.
6. Figuras (teaser, curva de temperatura, mecanismo), conforme `reviewer_presentation_v1.md`.
7. Os números da seção de resultados ainda digitados à mão (Heun 17,8/9,7%, 0,95/0,98; Qwen-Drive) devem virar macros geradas.
8. Prova completa da Proposição e condição de Lipschitz perto de t=1.

---

# Rodada 3 (sobre v3) e o que entrou na v4 — 26/10/2026

Notas: Reviewer 2 **5/10**, equilibrado **5/10** (subiram de 4). Apresentação: entregou as três figuras (`paper/figures/`).

## Corrigido na v4
- Figuras: teaser (Fig. 1), mecanismo (Fig. 2, substitui a tabela de contração, que foi para o apêndice), curvas em 2 linhas (Fig. 3, figure*); versão com minADE₆ no apêndice.
- Faixas geradas por macro: fração explicada **79–121%** (não "90–100%"); resíduo máximo de inversões **2,0 pt** (não "±2").
- IC do Alpamayo-2 E10 [0,03; 4,4] **exclui zero**: dito no texto.
- minADE₆: o Euler é pior que a referência e um pouco pior que a temperatura equivalente (resíduo 0,00–0,11 m, positivo em 5/6 células). Os resíduos de entropia (−0,03 a 0,03) entram no Finding 2.
- Euler-1: fora da faixa; mesmo em τ=0,4 o resíduo é de +11 a +15 pt. Isso fica explícito.
- Orçamento: Heun-3 = 6 NFE, Heun-5 = 10 NFE (o mesmo do sampler publicado).
- Parte do resíduo do Heun é erro de integração dele mesmo, e o texto não afirma que "todo não-Euler falha".
- Reposicionamento: o teste identifica "contração para a média com o mesmo spread"; um encolhimento post-hoc também passa, e a temperatura é o botão explícito. O abstract virou proposta de protocolo de avaliação.
- "436 clipes inéditos" corrigido: a confirmação precisa de clipes e seeds novos.
- Protocolo e Setup fundidos; um único nome ("reference solver"); margens de equivalência declaradas; frações acima de 100% explicadas.
- Caixas de Finding agora quebram entre páginas; a nota † morta foi removida; a coluna Gaussiana está rotulada como outra amostra.

## Pendências (prioridade)
1. Braço de encolhimento (shrinkage) ao lado de A_τeff na Tabela 2 (CPU, script em `reviews/scripts/shrink_control.py`).
2. Heun-3/5 no R1 e no A2 e Heun casado em deslocamento; convergência da referência (Heun-32 vs Heun-64 fp32).
3. Referência em τ ∈ {0,3; 0,2; 0,1} para E1/E2.
4. Confirmação pré-registrada em clipes e seeds novos (desfecho primário: equivalência do E10 na média dos modelos; Heun-3 deve falhar).
5. FlashDrive oficial e closed-loop (rodando); ReCogDrive/DiffusionDrive no NAVSIM.
6. Qwen-Drive na amostra de 60 clipes; bib TODOs (Alpamayo, Qwen-Drive, AlpaSim, PhysicalAI); provas no apêndice.

## v4.1: controle de encolhimento (feito)
Artefato `artifacts/paper_numbers_v2_20261026` (bootstrap de 1000, agora registrado no JSON; as macros antigas não mudaram).
- Shrink_k = E1 + c(A1 − E1), com c casado ao spread do E_k. Os resíduos de inversões vão de **+0,6 a +2,6 pt**, e todos os ICs incluem zero: **passa no teste igual ao Euler**.
- Por amostra o Euler é diferente: o encolhimento contrai 100% das amostras com mediana 0,56–0,58 (E4) e 0,79–0,80 (E10); o Euler contrai 90–93%, com 0,66–0,68 e 0,84–0,86.
- Consequência: a afirmação fica "contração para a média, indistinguível por taxas de outras contrações do mesmo tamanho; temperatura = botão explícito". Linhas Shrink_k entraram na Tabela 2.
- Rodada 4 dos revisores lançada sobre `versions/main_v4.1.pdf`.

## Rodada 4 — revisor equilibrado: 5/10 (igual à v3); correções imediatas na v4.2
Parecer: `reviewer_balanced_v4.md`. O reposicionamento deixa o paper mais correto, porém menos significativo sozinho: um paper de protocolo precisa mudar uma conclusão publicada (FlashDrive) e transferir para outra família (NAVSIM).
- **Corrigido (verificado por nós):**
  - o encolhimento agora é casado exatamente em spread (bisseção; τ_eff idêntico ao do Euler);
  - artefato `paper_numbers_v3_20261026`;
  - E1 não é determinístico nas redes (9–14% do spread), e o texto foi ajustado;
  - a equivalência usa IC de 90% (TOST), como no código; o texto dizia 95%;
  - as margens foram escolhidas após ver os 60 clipes, e o texto agora diz isso;
  - "assim como o Euler" foi removido, porque o IC do A2 E10 exclui zero.
- **Achado novo (exploratório, mesma amostra), a assinatura por amostra:** mediana de contração com o mesmo spread:

  | Passos | Euler | Temperatura | Encolhimento |
  |---|---|---|---|
  | E4 | 0,66–0,68 | 0,69–0,73 | 0,53–0,55 |
  | E10 | 0,84–0,86 | 0,84–0,87 | 0,78 |

  Fração contraída: Euler 90–93%, temperatura 81–89%, encolhimento 100%. O Euler se parece com a temperatura, não com uma contração uniforme. Vira desfecho secundário pré-registrado.
- **Prioridade dos experimentos segundo ele:**
  1. FlashDrive na mesma temperatura (com a previsão declarada antes);
  2. confirmação pré-registrada (TOST do E10 agregado; Heun-3 deve falhar nos 3 modelos; assinatura como desfecho);
  3. NAVSIM/PDMS (DiffusionDrive não é rectified flow: declarar a previsão);
  4. closed-loop como sanity check.
- **Ainda pendente (barato):**
  - Heun-3/5 no R1/A2;
  - Proposição 3.1: provar ou rebaixar;
  - Qwen-Drive gerado por código;
  - zerar TODOs e apêndices.

## Rodada 4 — Reviewer 2: 5/10 (igual); apresentação: `reviewer_presentation_v4.md`
- **Reviewer 2: todos os números da v4.1 batem com o summary.json.**
- **Achado a favor:**
  - nos 40 clipes não usados do A1.5 (160–199), o E10 replica (resíduo −0,01 pt [−1,7; +1,7]);
  - nos 100 clipes, o IC de 90% [−0,78; +1,41] fica dentro da margem de ±2;
  - ressalva: isso é post-hoc e não substitui a confirmação.
- **Contradiz a concessão da v4.1** ("qualquer contração do mesmo tamanho passaria"), com controles dele com spread casado (E4, agregados):

  | Contração | Resíduo de inversões |
  |---|---|
  | encolher para a média das 6 amostras | −2,8 pt [−4,8; −0,9] |
  | colapsar amostras aleatórias no E1 | +6,5 pt [+2,9; +9,8] |
  | encolher para uma reta a velocidade constante | +16 pt |

  Ou seja, o teste rejeita algumas contrações; o Shrink passa porque encolher para E1 é quase o que a temperatura faz. A frase do abstract precisa mudar.
- **Distância de energia por clipe** (em relação à temperatura casada): Shrink4 0,28–0,36; E4 0,16–0,25; temperaturas vizinhas 0,07–0,08. A ordem é temperatura < Euler < encolhimento. Isso confirma a assinatura por amostra que adicionamos na v4.2.
- **Pendências dele:**
  - 13 TODOs e apêndices vazios: para ele, manuscrito completo é pré-requisito para passar de 5;
  - Heun só no A1.5; nos clipes 160–199 o τ_eff do Heun fica preso em 1;
  - a ordem de convergência parece errada, e a convergência da referência ainda não foi checada;
  - a introdução diz que o minADE registra ganho, mas os dados mostram piora;
  - "quase parado" inclui um clipe a 2,64 m/s;
  - não há artefato para "5 clipes distintos" de colisão.
- **Já corrigido na v4.2:** E1 com 9–14% de spread; IC de 90%; spread do Shrink casado exatamente.
- Scripts dele em `scratchpad/r2_round4/`. Copiar para `reviews/scripts/round4_reviewer2/`.

## Pergunta do Ross (Slack): onde "a média" falha (exploratório, 60 clipes × 6 seeds)
Script: `reviews/scripts/mean_fail*.py`. Contamos clipes ambíguos (a referência tem >1 decisão).

| Modelo | E1 (decisão ausente da referência) | Referência τ=0,4 (idem) | E4 (apaga decisão) | Referência τ=0,6 (idem) |
|---|---|---|---|---|
| A1.5 | 5 | 0 | 15 | 16 |
| R1 | 5 | 3 | 17 | 20 |
| A2 | 7 | 0 | 25 | 26 |

Exemplos:
- context_148 (A1.5 e A2): todas as amostras seguem reto; E1 desvia de faixa.
- context_105 (A2): straight → shift_right em 6/6.
- context_106 (R1): referência com 5 à esquerda e 1 à direita; o E4 produz um "straight".

# Rodada 5 (sobre v5, focada nas mudanças)
## Equilibrado: 5/10 (igual)
- **Enquadramento:** coerente. Manter o título; trocar só se a assinatura falhar na confirmação.
- **Assinatura com IC** (bootstrap conjunto, 3 modelos; scripts em `reviews/scripts/round5_balanced/`):

  | Diferença na mediana | E4 | E10 |
  |---|---|---|
  | Euler − encolhimento | +0,128 [+0,107; +0,146] | +0,067 [+0,057; +0,076] |
  | Euler − temperatura | −0,037 [−0,070; −0,011] (puxado pelo R1) | −0,002 [−0,019; +0,016] |

  A fração contraída do Euler fica 5–7 pt acima da temperatura. Medir até a média do E1 entre seeds não muda nada.
- **Correções de texto para a v5.1:**
  1. A assinatura do encolhimento é fixa por construção (razão = c, desvio-padrão 0). O conteúdo real: Euler e temperatura reduzem o spread entre pares mais que a distância à média, o que nenhum mapa linear faz. Isso vem da estrutura não-Gaussiana. Pôr ICs.
  2. O Euler fica entre temperatura e encolhimento, mais perto da temperatura; dizer isso em vez de "ambos deixam amostras sem contrair".
  3. O abstract se contradiz ("não por amostra" × "assinatura por amostra"). Distinguir pares com o mesmo ruído da distribuição da contração por amostra.
  4. Incluir as contrações do Reviewer 2 que falham no teste de taxas (média das 6, colapso aleatório no E1, reta) e notar que os 6 resíduos do encolhimento são positivos (0,7–3,4).
  5. Trocar "Heun falha nos dois testes" por "falha no teste de taxas".
- **Para 6:**
  1. FlashDrive com previsão declarada antes;
  2. confirmação com desfecho de assinatura de limiar fixo (|E−T|<0,05 e |E−S|>0,05; célula primária E10);
  3. correções 1–5;
  4. manuscrito completo.
- **Para 7:** NAVSIM + (FlashDrive muda OU assinatura confirmada OU closed-loop informativo).

## Reviewer 2: 5/10 (igual). Os números novos batem.
- **A assinatura está forte demais para o abstract.**
  - O Euler é diferente da temperatura no R1: E4 −0,072 [−0,120; −0,033] e E10 −0,032 [−0,051; −0,011].
  - A fração contraída difere em 4 de 6 células (+4,5 a +12 pt).
  - A mediana não é específica: um encolhimento uniforme para a média das 6 amostras (no E10) e uma contração radial não-uniforme (no E4) também caem na mediana do Euler.
- **Decisão:** tirar a assinatura do abstract. No texto, dizer "heterogênea, mais perto da temperatura que do encolhimento, mas não idêntica", com ICs por modelo e o perfil de quantis. Declarar agora a estatística, a margem e a direção do desfecho secundário.
- **"Heun falha nos dois testes":** remover. Na verdade o Heun contrai 69–70% das amostras, contra 21–42% da temperatura perto de τ=1, onde o τ_eff fica preso.
- **Menores:**
  - "sampled before any analysis";
  - reportar os 40 clipes de holdout do A1.5 (160–199);
  - abstract com resíduos em vez de frações;
  - coluna de equivalência (só o R1 E10 passa nas duas métricas);
  - o minADE₆ distingue Euler e encolhimento nas taxas (0,05 vs 0,20 m no A1.5 E10).
- Para passar de 5 ele exige o manuscrito completo.

## Apresentação (rodada 5): `reviewer_presentation_v5.md`
- Propôs: abstract sem a assinatura; Finding 4 com figura (`figures/fig_signature.pdf`); Shrink na Fig. 3 (`fig_curves_shrink_v3_2row.pdf`); tabela em uma coluna (`tab_matched_v5_proposal.tex`) colocada ao lado dos Findings; `\ifdraft`; Qwen-Drive como parágrafo "Boundary".

## v5.1 (19:00): o que entrou
- `paper_numbers.py` agora calcula a assinatura com bootstrap pareado, por modelo e agregado (artefato `paper_numbers_v4_20261026`; as macros antigas não mudaram). Os ICs coincidem com os dos dois revisores.
- **Abstract:** sem a assinatura; com resíduos (≤2,0 pt) em vez de frações; Heun +9,1 a +16,5 pt; "o encolhimento linear também passa".
- **Finding 4 reescrito:** fator único por construção; o Euler fica mais perto da temperatura que do encolhimento, mas não é idêntico (R1). A mediana sozinha não identifica temperatura.
- **Discussão:** a afirmação é no nível das taxas. A assinatura vira desfecho secundário com estatística e margem (0,05) declaradas.
- "Heun falha nos dois testes" → "falha no teste de taxas"; o IC agora aparece como [0.03, 4.4]; "sampled before any analysis" removido; as figuras leem um único artefato (v4).
- **Ainda não feito:**
  - tabela em uma coluna, figura da assinatura e Shrink na Fig. 3;
  - Qwen-Drive como parágrafo;
  - holdout de 40 clipes;
  - controles que falham no teste de taxas (portar os scripts do R2 para o código do paper);
  - apêndices e TODOs.

## Rodada 6 (v5.2) e v5.3 (26/10, 20:55)

Notas: Reviewer 2 5 (=), equilibrado 5 (=), apresentação 6 (↑ de 5).
- **Problemas apontados no parágrafo FlashDrive da v5.2 e corrigidos na v5.3:**
  - "further clips": são os mesmos 60 clipes.
  - O ganho de 0,28 m era medido contra o Heun32. Contra Euler-8 é 0,12 m.
  - "sits at the margin": o limite superior 0,051 passa da margem.
  - Faltava o intervalo de τ_eff.
  - O E4 que falha não era reportado.
  - Faltava dizer que o cache ≈ E4 (Δ −0,001 [−0,033; 0,032]).
  - Faltava dizer que o cache foi reimplementado.
- **Pré-registro do adendo 11b:** gravado no mesmo comando da análise, 3 s antes. A correção foi anotada no protocolo.
- **Pendente:**
  - promover o FlashDrive a Finding e citá-lo no abstract e na introdução (apresentação);
  - flips e minADE6 do FlashDrive;
  - decomposição z-lab;
  - 13 TODOs;
  - toggle de draft.

## v5.4 (26/10, 22:55)
FlashDrive promovido a Finding (find:flash), citado no abstract e na contribuição 2; TODO do FlashDrive na introdução removido (fica o do closed-loop). 8 páginas, refs na p. 6, 0 overfull. z-lab rodando (6/60).

## Rodada 7 (v5.4) e v5.5 (26/10, 23:10)

Notas: Reviewer 2 5 (=), equilibrado 5 (=), apresentação 6,5 (↑ de 6).
- **Correções na v5.5:**
  - título do Finding "is" → "behaves like";
  - o abstract não diz mais "within 0.051", diz "close to but not within the margin" e informa o escopo (checkpoint original, mesmos clipes);
  - E8 passa (reportado);
  - intervalo do ganho [0.09, 0.14];
  - interpolação linear declarada;
  - intervalos de 90% declarados;
  - aviso de que os τ_eff diferem da Tab. 1 (entrada de 4 frames);
  - "re-implemented" → "implemented ourselves";
  - a introdução não afirma mais o efeito para ProbeFlow, AdaVLA e Coda;
  - cabeçalho do adendo 11b corrigido.
- **Pendente (o equilibrado: "6 precisa de evidência, não de texto"):**
  - flips e minADE6 do FlashDrive a partir dos dados existentes;
  - fração explicada;
  - z-lab (7/60);
  - confirmação nos 436 clipes;
  - TODOs.

## v5.6 (26/10): revisão das citações e Qwen-Drive (reviews/cited_papers_review.md)

**Qwen-Drive.** A afirmação de fronteira estava errada. O modelo é x-prediction, treinado só até t ≤ 0,9, e o sampler oficial termina na previsão em t = 0,9; a nossa referência Euler-64 consulta tempos não treinados. Verificado em planning_expert.py:417 e no min_one_minus_t = 0,1 da config.
- O Finding find:qwen foi removido; o Qwen virou o parágrafo "Where the view does not apply" na Discussão.
- A contribuição 4 foi reescrita e o Qwen saiu de Policies.

**Bibliografia.**
- alpamayo2025: título e arXiv 2511.00088.
- qwendrive2026: tech report, arXiv 2609.00111.
- FlashDrive passou a citar só o arXiv.
- PhysicalAI e AlpaSim: ano 2025 e chaves renomeadas.
- Biroli: volume e artigo.
- Texto do GMFlow reescrito.
- Adicionados EDM, Glow, BigGAN e DiffusionDrive.

**Correções no texto.**
- Faixa s_d corrigida para 0,3–3,3 (verificado numericamente: f(10) = 0,617).
- Passos do FlashDrive descritos como do 4º ao 7º.

**Pendente.**
- Ning 2024, Analytic-DPM, Flow-GRPO e Yue 2025.
- Indicar qual arquivo de split do PhysicalAI usamos.
- Coda ("Fewer Steps, Better Actions") merece discussão explícita.

## v5.7 (27/10): cota do fator gaussiano

- Adicionada a cota ∏cos θ ≤ cos(π/2k)^k em qualquer grade: 0,883 para E10, 0,729 para E4 e 0,5 para E2. Conferida numericamente: o máximo em s_d e em grades aleatórias fica abaixo da cota.
- Não houve rodada de revisores. Fica para quando saírem os resultados nos clipes fora do treino (adendo 13), que podem mudar os números principais.

## v5.8 (27/10): Qwen-Drive v2 (RL)

- O parágrafo "Where the view does not apply" virou "A second family":
  - o mapeamento passos → temperatura vale (E10 com τ_eff 0,98; E4 com 0,42);
  - a temperatura não muda o erro do Qwen, então o E4 só custa precisão (+0,041 m [0,022; 0,062]).
- A contribuição 4 e as limitações foram ajustadas.
- Todos os números saem de macros geradas a partir do artefato.
- Falta o SFT.

## Rodada 8 (v5.8) e correções em v5.9 (27/10)

**Notas:**
- R2: 4 (era 5);
- equilibrado: 5 (igual);
- apresentação: 6,5 (igual).

**Consenso.** O ponto decisivo é a sobreposição com o treino do Alpamayo (57/60). O balanced diz que o adendo 13 virou o experimento decisivo.

**Corrigido em v5.9:**
- os dados agora são chamados "default test split", e a sobreposição de 57/60 está declarada (macro tirada de artifacts/split_overlap_v1_20261027);
- "real test data" foi removido;
- o abstract diz que só o E10 do R1 passa a equivalência nas duas métricas;
- FlipResidMaxAbs agora com 2 casas (2,04);
- Qwen: a falha da previsão está declarada, o resíduo de flips aparece com IC, e o escopo da cota (Euler em [0,1]) está explícito;
- a contribuição 4 abre pelo spread ser um décimo do erro;
- adendo 13b: TOST nos clipes fora do treino, registrado antes de abrir qualquer saída.

**Pendente:**
- resultados fora do treino (decisivos);
- SFT do Qwen;
- flips do FlashDrive e o checkpoint z-lab;
- prova da Prop. 3.1 no apêndice;
- apêndices vazios e 14 TODOs;
- tabela do Qwen;
- o abstract tem 252 palavras;
- Tabela 1 está na p.5;
- "linear" vs "uniform";
- "79–121%" na contribuição 2.

## v5.10 (27/10)
- "79–121%" virou razão: a temperatura sozinha muda ADE₁ e flips 0,79–1,21 vezes o que E4/E10 mudam, na contribuição 2 e no find:matched. Sem rodada de revisores; a próxima espera os resultados fora do treino.

## v5.11 (27/10): Qwen SFT
- O parágrafo 'A second family' agora cobre RL e SFT, com macros tiradas de artifacts/qwendrive_analysis_v2_20261027. Duas falhas de previsão declaradas: o ADE₁ do RL e os flips do SFT.

## v5.12 (27/10)
- O z-lab entrou no find:flash (τ_eff 0,65; ganho sobre o E8 de 0,11 m; resíduo +0,008 [−0,042; 0,062]). O TODO do checkpoint otimizado foi removido.
