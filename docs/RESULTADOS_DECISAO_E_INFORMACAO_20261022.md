# Rodada de 22/10 (noite): inversões de decisão, dados reais PhysicalAI-AV e schedules por informação

**Status: candidata em teste de confirmação. Nenhuma contribuição ICML estabelecida.** Protocolo e adendos: `PROTOCOLO_GATE_COMANDO_MULTIEPISODIO_V1.md`. Candidata: `CANDIDATO_INFORMACAO_DA_DECISAO.md`. Teoria: `TEORIA_INVERSOES_DE_DECISAO.md`.

## Dados

- **CARLA:** 14 episódios independentes / 28 contextos em Town01/02/04/10HD, com 20 NPCs estáticos e velocidade-alvo de 3–12 m/s. Town03 e Town05 (e 2 episódios de Town02) causaram segfault do servidor e foram descartados antes de qualquer inferência.
- **PhysicalAI-AV (NVIDIA), split `test`:** 200 clipes, um por chunk, no formato nativo do Alpamayo1.5 (4 câmeras × 4 frames, 16 poses), com o futuro real do motorista. Não verificamos de forma independente se esses clipes ficaram fora do treino.

## 1. Solvers mudam decisões, e minADE não mostra isso

PhysicalAI, 200 clipes × 2 seeds, decisões lidas do plano (longitudinal: parar/desacelerar/manter/acelerar; lateral: esquerda/direita/desvio/reto). Inversões em relação ao Heun160:

| Solver | NFE | Inversões (IC 95%) | ADE médio à ref. | Concorda com humano (long.) |
|---|---:|---:|---:|---:|
| Euler10 (publicado) | 10 | 6,0% (3,8–8,8) | 0,28 m | 84,3% |
| Euler4 | 4 | 11,0% (7,8–14,5) | 0,56 m | 86,0% |
| Cache estilo FlashDrive | 4 | **12,8% (9,8–16,0)** | 0,71 m | 87,3% |
| Heun5 / RK4×2 | 10 / 8 | ~10% (sobretudo lateral) | 0,38–0,42 m | 83% |
| Heun160 (ref.) | 320 | — | — | 83,0% |

Solvers grosseiros ficam **mais perto do humano** (o cache tem ADE ao humano 0,36 m menor). "Fidelidade ao fluxo" e "acerto" são perguntas diferentes, o que é consistente com [Coda](https://arxiv.org/abs/2609.21216). No CARLA, o PID com buffers vazios inverteu acelera↔freia em 2–9% dos casos, e o steering nunca mudou mais de 0,05.

## 2. A fragilidade numérica fica confinada à incerteza da própria política

Probe de comprometimento: 200 clipes × 6 seeds = 1.200 casos.

- Clipes em que as 6 seeds concordam na decisão (H=0, 47%): **0,0%** de inversões numéricas (congelamentos de janela e cache). Clipes com H>0: **19%**. Spearman(H, taxa de inversão) = 0,76 (p≈10⁻³⁸).
- 69% dos casos têm a classe já fixada no endpoint previsto em t=0; desses, só 0,36% invertem. Os demais invertem 31,6%.
- Suscetibilidade por janela de congelamento no Euler10: 2,3 / 2,2 / 1,1 / 0,7 / 1,9% de [0,.2) até [.8,1). É maior no início, onde a concordância da classe do endpoint previsto sobe de 0,73 para 0,86.

Isso é o análogo empírico da Proposição 1 (simetria ⇒ nenhuma inversão): sem incerteza de decisão, não há o que inverter.

**Análise descartada (vazamento):** "classe em t=0 ≠ classe final" parecia prever as inversões do cache com AUROC 0,88, mas usava a classe do Euler10 completo. Sem vazamento (contra a classe do próprio cache), a AUROC foi de 0,48–0,51. O cosseno inicial teve 0,72 e a margem de ramo 0,65.

## 3. Correção grátis com as avaliações existentes

A correção trapezoidal com as velocidades que o Euler10 já calculou reduz o ADE à referência de 0,283 para 0,222 m, sem nenhuma avaliação extra. As inversões caem de 6,0% para 4,75% (IC da diferença −3,0 a +0,25 pontos). É uma técnica clássica (multipasso/deferred correction), não uma contribuição.

## 4. Schedules com o mesmo NFE: modelo solúvel e Alpamayo

**Modelo solúvel** (`commitment_theory_v3.py`, 72 casos): inversões médias de 3,5% (informação da decisão + uniforme), 5,0% (uniforme), 7,8% (tempo entrópico reescalado, [Stančević et al., NeurIPS 2025](https://arxiv.org/abs/2504.13612)), ~13% (cache estilo FlashDrive). O tempo entrópico piora com a dimensão (d=64: 2–4× pior que uniforme), como previsto.

**Alpamayo, holdout 100–199** (calibração em 0–99), 2 seeds, deltas pareados em relação ao uniforme:

| NFE | Inversões: decisão+unif. | ADE à ref.: decisão+unif. | ADE à ref.: tempo entrópico |
|---|---|---|---|
| 4 | 11,5% vs 12,0% (decisão pura: 9,5%) | −0,067 m [−0,088; −0,046] | −0,041 m |
| 6 | **6,0% vs 8,5% [−5,5; −0,5]** | −0,055 m [−0,074; −0,038] | −0,033 m |
| 8 | 5,5% vs 7,0% [−3,5; 0,0] | −0,054 m [−0,083; −0,034] | −0,035 m |

O giro do campo (≈ProbeFlow) não melhorou e piorou o ADE. Nenhum schedule mudou a concordância com o humano. A direção é consistente, mas há poucos eventos. A confirmação com 436 clipes inéditos e 4 seeds, com a calibração congelada, está registrada no adendo 6.

## 5. Colapso numérico: solvers de 1ª ordem encolhem em direção à média condicional

**Fato analítico básico:** no rectified flow, v(z,0)=E[x₁|c]−z, logo um passo de Euler devolve exatamente a média condicional. Para alvos Gaussianos, o Euler em qualquer grade subestima o desvio em cada autodireção (fator Π cos θᵢ, Cauchy–Schwarz/convexidade). **Isso já está demonstrado** como desigualdade na prova da Proposição 2 de [Claustre, Negrel, Boyer, Nadjahi & Vanden-Eijnden (2026)](https://arxiv.org/abs/2609.25839). É base a citar, não contribuição nossa.

**Misturas, modelo solúvel** (`mode_collapse_theory.py`, 36 configurações com K∈{2,3,5}, d∈{2,16}, Euler/Heun/RK4/cache):
- 99,95% das 408.769 amostras invertidas terminam mais perto da média global que o endpoint exato.
- Com K=2, Euler: 100% das inversões vão para a classe majoritária, que contém a média. Com K=5, 66% caem na classe que contém a média e só 39% na mais provável; o mecanismo é atração para a média, não para a maioria.
- A entropia da decisão cai em 92% das configurações com inversões. O Heun às vezes inverte no sentido oposto (corretor de 2ª ordem).

**Alpamayo, PhysicalAI 200 clipes × 2 seeds:** razão (distância à média condicional do solver)/(do Heun160).

| Solver | Amostras encolhidas | Razão mediana |
|---|---:|---:|
| Cache estilo FlashDrive | 400/400 | 0,50 |
| Euler4 | 400/400 | 0,64 |
| Euler8 | 400/400 | 0,80 |
| **Euler10 (sampler publicado)** | **400/400** | **0,84** |
| Heun5 / RK4×2 | 50% / 22% | 1,00 / 1,05 |

**Consequência na distribuição de decisões** (6 seeds, IC por bootstrap de clipe), cache contra referência: parar −0,75 pt [−1,6; −0,2], desacelerar −2,5 pt [−3,8; −1,2], acelerar +1,25 pt [+0,1; +2,6], entropia da decisão −0,24 bit [−0,30; −0,19]. Euler10: entropia −0,14 bit [−0,19; −0,10], desacelerar −0,9 pt [−1,7; −0,3].

**Contra o humano não há prejuízo detectável.** O log-score da decisão humana sob a distribuição de 6 seeds do cache é −0,439 contra −0,457 da referência (diferença +0,018 [−0,027; +0,057]), e a acurácia balanceada é maior com menos passos. O encolhimento age como temperatura baixa: melhora o top-1 e o ADE ao humano e reduz a diversidade. Isso dá uma explicação mecanística para o efeito "Fewer Steps, Better Actions" ([Coda](https://arxiv.org/abs/2609.21216)), a verificar com o desenho deles.

**Métricas contra o humano seguem a previsão de "temperatura numérica"** (PhysicalAI, 200 clipes). O ADE₁ cai de forma monótona com o encolhimento: média condicional (1 passo) 1,84; cache 1,84; Euler4 1,92; Euler10 2,06; Heun160 2,20 m. A dispersão entre seeds também cai: 1,89 (Heun160) → 1,52 (Euler10) → 0,90 (cache) → 0,27 m (média). Com 6 seeds, a vantagem do Euler10 sobre o Heun16 em minADE_k some e se inverte à medida que k cresce: −0,180 [−0,219; −0,143] (k=1), −0,083 (k=2), −0,028 (k=3), +0,005 (k=4), +0,046 [−0,023; +0,153] (k=6). Isso é consistente com o FlashDrive ter relatado minADE₁ melhor e minADE₆ pior: a mesma aceleração recebe veredictos opostos conforme a métrica.

**Correção pós-hoc ("des-encolher"):** a' = média + (a−média)/c, com c calibrado em 0–99 e avaliado em 100–199. No Euler10, o ADE à referência caiu de 0,308 para 0,237 m e as inversões de 6,0% para 5,0%, com as frequências de decisão restauradas. No Euler4 e no cache, as inversões aumentam (o erro é amplificado). Fatores por coordenada ou por PCA pioram tudo. O encolhimento não é bem invertível depois do fato.

**Trabalhos relacionados nesta linha:** [SubFlow](https://arxiv.org/abs/2604.12273) (perda empírica de submodos raros com poucos passos, imagens) e [GMFlow](https://arxiv.org/abs/2504.05304) (hipótese Gaussiana única em poucos passos). Nenhum dos dois trata direção das inversões, decisões ou políticas.

## 6. Temperatura explícita vs implícita (adendo 7)

Holdout 100–199 com 6 seeds e 32 configurações (`temperature_report_v1_20261022/summary.json`). A temperatura explícita é ruído inicial ×τ.

**Hipótese de método falsificada.** No mesmo NFE, temperatura explícita + Heun não supera o Euler. Com 10 NFE, o Euler10 (ADE₁ 2,336; minADE₆ 1,220) domina todas as variantes Heun5·τ; o Heun3 (6 NFE) fica pior que tudo.

**Achado: o Euler com poucos passos ≈ fluxo preciso com temperatura menor.** Interpolando a curva do Heun32·τ na mesma dispersão entre seeds:

| Solver | τ efetivo | ADE₁ solver/temp. | minADE₆ solver/temp. | desacelerar | inversões vs Heun32 |
|---|---:|---|---|---|---|
| Euler4 | 0,65 | 2,193/2,189 | 1,355/1,313 | 0,192/0,185 | 0,108/0,091 |
| Euler6 | 0,75 | 2,263/2,253 | 1,262/1,226 | 0,197/0,194 | 0,077/0,064 |
| Euler10 | 0,84 | 2,336/2,316 | 1,220/1,157 | 0,203/0,200 | 0,050/0,047 |
| Heun5 | — | 2,563 vs 2,465 (τ=1) | 1,308 vs 1,074 | — | 0,113 vs 0 |

O τ efetivo pela dispersão coincide com a razão de encolhimento medida de forma independente (0,84 e 0,64). A temperatura explica ~94% das inversões do Euler10 publicado; o resíduo numérico aparece sobretudo em minADE₆ (+0,04–0,06 m). O erro dos solvers de ordem maior com poucos passos **não** é do tipo temperatura.

Com o solver preciso, baixar τ reproduz "menos passos, ações melhores": ADE₁ de 2,465 → 2,088, concordância modal com o humano de 0,82 → 0,84 e desacelerar de 21,0% → 17,3%. O efeito é de temperatura, não da numérica.

**Consequência metodológica:** comparações de schedules ou acelerações devem ser feitas na mesma temperatura efetiva (mesma dispersão), senão um ganho em "inversões" ou "ADE" pode ser só uma mudança de τ. Isso vale também para a confirmação da seção 4.

## O que isto ainda não é

- Não é um ganho de segurança nem de qualidade de direção: a referência é numérica.
- Não é closed-loop.
- A novidade depende de o funcional de decisão ser realmente distinto da linha de tempo entrópico e especiação; a revisão de literatura deve ser aprofundada antes de qualquer submissão.

## 7. Veredito dos schedules na mesma temperatura (adendos 6 e 8)

**Confirmação com 436 clipes inéditos × 4 seeds:** passou como pré-registrado. A informação da decisão + uniforme reduziu inversões contra a referência em −0,88 pt [−1,28; −0,52] em relação ao uniforme e em −0,46 pt [−0,76; −0,15] em relação ao tempo entrópico. Mas ficou mais longe do humano (ADE +0,017 m [+0,010; +0,023]), assinatura de menos encolhimento.

**Mesma temperatura efetiva** (v1 100–199, 6 seeds, curva Heun32·τ, `schedule_matched_report_v1_20261022`): o τ_eff da informação da decisão é maior que o do uniforme (0,71 vs 0,65 com NFE 4; 0,79 vs 0,75 com 6; 0,83 vs 0,81 com 8). O resíduo de inversões em relação ao uniforme foi +0,009 [−0,007; +0,023], −0,010 [−0,025; +0,006] e −0,012 [−0,024; 0,000]. Nenhum IC exclui zero. Pelo critério do adendo 8, **a alocação pela informação da decisão não é um método**: o ganho observado é mudança de temperatura. Todos os schedules testados (incluindo tempo entrópico e cache) ficam dentro da faixa de ruído da curva de temperatura. Exceção isolada: giro do campo com NFE 4, −0,029 [−0,052; −0,006], uma de várias comparações e não promovida.

**Leitura para o paper:** comparações de schedules, solvers e acelerações em políticas de flow matching devem ser feitas com temperatura efetiva pareada. Sem isso, até uma confirmação pré-registrada fora da amostra pode medir apenas uma mudança de τ.

## 8. Replicação em quatro VLAs de direção e em robótica (22–23/10)

**Três gerações do Alpamayo** (PhysicalAI v1 100–159 ou 100–199, 6 seeds, Euler-k comparado com Heun32·τ na mesma dispersão; relatórios `temperature_report_{v1,r1_v1,a2_v1}_20261022`):

| Modelo | τ_eff Euler10 | τ_eff Euler4 | Inversões Euler10: solver / curva de temperatura | ADE₁ Euler10: solver / curva |
|---|---:|---:|---|---|
| Alpamayo1.5 | 0,84 | 0,65 | 0,050 / 0,047 | 2,34 / 2,32 |
| Alpamayo-R1 | 0,84 | 0,61 | 0,061 / 0,061 | 2,28 / 2,30 |
| Alpamayo2-Super | 0,83 | 0,62 | 0,097 / 0,077 | 2,78 / 2,76 |

O sampler oficial das três gerações opera na mesma temperatura efetiva (≈0,84), e as métricas de decisão e ADE ficam a 0,0–0,02 da curva de temperatura explícita.

**Qwen-Drive-1.0-4B** (Alibaba; x-prediction; 60 clipes, 6 amostras; planejadores RL e SFT; referência Euler64 com piso min(0,1; 1/k)): o planejador é quase determinístico. A dispersão entre amostras no solver preciso é 0,037 m (contra ~2,2 m no Alpamayo), e τ quase não tem efeito (0,037 → 0,029 m; ADE₁ constante em 1,57 RL / 1,45 SFT). Isso vale também para o SFT, então **não** é colapso causado pelo RL. O sampler oficial Euler10 tem dispersão **maior** que o preciso (0,16 m): neste modelo o erro numérico **acrescenta** diversidade em vez de encolher. O minADE₆ oficial (1,42 RL / 1,33 SFT) é melhor que o do solver preciso (1,56 / 1,44) só por causa dessa diversidade numérica. Portanto, parte do ganho best-of-6 reportado por esse tipo de modelo pode vir do erro do sampler, não de multimodalidade aprendida (a confirmar no protocolo deles).

**Robótica, SmolVLA/LIBERO** (400 episódios pareados por configuração): o encolhimento aparece em 100% das amostras (τ_eff 0,92/0,78/0,53 para Euler 10/4/2). O sucesso em closed-loop é quase insensível a solver e τ na faixa 0,5–1 (todas as diferenças pareadas dentro de ±3 pt, ICs incluindo 0, exceto Euler10 − Heun16·τ0,9 = +3,0 [+0,2; +6,2]). Só temperatura muito baixa prejudica tarefas longas (LIBERO-10: Euler1 −18 [−31; −6], Heun16·τ0,25 −10 [−19; −1]). P3 ("menos passos melhora o sucesso") não apareceu; P2 fica inconclusiva porque a curva de sucesso é plana.

## 9. Consequência de segurança: colisões em replay não reativo (25/10)

`src/cavla/physicalai_safety_metrics.py`, `artifacts/safety_metrics_v1_20261025/`. Outros agentes seguem as trilhas autorrotuladas do PhysicalAI (`obstacle.offline`), convertidas do referencial do rig para o do ego em t0 e interpoladas a 10 Hz, como no PDMS do NAVSIM. Não há mapa (sem área dirigível). Horizonte de 4 s, caixa do ego 4,9 × 2,1 m. Foram 99 clipes avaliados; 1 sem rótulos foi excluído. A trajetória humana registrada colide em 1,0% (piso de ruído de rótulo/caixa). A métrica de conforto foi descartada: a trajetória humana passa em só 32% dos casos (ruído de diferenciação a 10 Hz).

Agregando Alpamayo1.5/R1/2 (1.302 pares clipe × seed; McNemar pareado):

| Comparação | Colisão | Discordantes | p |
|---|---|---|---|
| Heun32 τ=1 vs τ=0,6 | 1,61% vs 0,84% | 10:0 | 0,002 |
| Heun32 τ=1 vs τ=0,4 | 1,61% vs 0,69% | 12:0 | <0,001 |
| Euler4 vs Heun32 τ=1 | 0,69% vs 1,61% | 0:12 | <0,001 |
| Euler4 vs Heun32 τ=0,6 (≈ τ_eff) | 0,69% vs 0,84% | 0:2 | 0,50 |
| Euler10 vs Heun32 τ=1 | 1,31% vs 1,61% | 0:4 | 0,125 |
| Euler10 vs Heun32 τ=0,8 | 1,31% vs 1,08% | 3:0 | 0,25 |

Poucos passos de Euler reduzem colisões em replay, e esse ganho é reproduzido por temperatura explícita com solver preciso. Na mesma temperatura efetiva não há diferença detectável. O progresso não cai (41,7 → 42,2 m com τ menor). Os Qwen-Drive RL/SFT não colidem em nenhuma configuração (planos quase determinísticos). Limitações: eventos raros (21 vs 9), agentes não reativos, sem mapa, caixa fixa do ego.

## FlashDrive, checkpoint nvidia na mesma temperatura (26/10, adendo 11b)

Artefato: artifacts/flashdrive_analysis_nvidia_v1_20261026. 60 clipes (100–159) × 4 janelas × 6 seeds, código oficial do FlashDrive, unidade = clipe.
- **Cache do FlashDrive (Euler 8, passos 3–6 reutilizados):**
  - ADE₁ 2,405 contra 2,688 no Heun32 a τ=1, um ganho de 0,28 m;
  - τ_eff 0,64 [0,63; 0,65];
  - resíduo de ADE₁ +0,008 m [−0,034; +0,051].
  - Quase todo o ganho é temperatura, mas a equivalência formal falha por 0,001 m.
- **Euler 4:** τ_eff 0,63; resíduo +0,016 [−0,036; +0,070]. Não estabelece equivalência.
- **Euler 8:** τ_eff 0,80; resíduo +0,016 [−0,014; +0,048]. Passa.
- **Euler 10:** τ_eff 0,84 (replica o 0,83–0,84 do paper); resíduo +0,012 [−0,016; +0,041]. Passa.
- **Previsão:** a previsão de τ_eff 0,5–0,6 para o cache errou; o valor medido é 0,64.
- **Pendente:** o checkpoint z-lab (W4A8 + DFlash + streaming) foi relançado após a correção do reset de stream.

## Sobreposição com o treino do Alpamayo (27/10, 00:55)

O split oficial `splits/alpamayo_aligned.parquet` (PhysicalAI-AV v26.10, "segue a partição treino/teste dos modelos Alpamayo") foi baixado para artifacts/physicalai_splits_20261027. As nossas amostras vinham do split `default` de teste.
- **Amostra principal** (records v1, 100–159): 57 de 60 clipes estão na partição **train** do Alpamayo.
- **Conjunto de confirmação** (v2): 391 de 436 são train, 41 test e 4 alpasim.
- **Consequência:** os resultados open-loop do paper são, na maior parte, em clipes de treino. A comparação sampler × temperatura é dentro do mesmo modelo e das mesmas entradas, mas precisa ser refeita em clipes held-out.
- **Ação:** nova amostra de 400 clipes, um por chunk, só da partição test do alpamayo_aligned (artifacts/physicalai_heldout_contexts_v3_20261027, seed 20261027), a rodar quando houver GPU.

## Qwen-Drive v2, planner RL (adendo 12; 27/10)

**Dados.** artifacts/qwendrive_analysis_rl_v2_20261027/summary.json (código cavla.qwendrive_analysis): 60 clipes (context_100–159), seeds 42–47, bootstrap de 1000 por clipe re-resolvendo τ_eff.

**Referência e spread.**
- Referência: Heun-32 em [0; 0,9]. Difere do Heun-64 em no máximo 1,5 mm.
- Spread da referência: 0,136 m. ADE₁ da referência: 1,55 m.

| Sampler | τ_eff [IC 90%] | Resíduo ADE₁ (m) [IC 90%] | Flips | Previsão |
|---|---|---|---|---|
| E10 | 0,98 [0,98; 0,98] | +0,001 [0,000; 0,001]; TOST passa | 0 | τ_eff ≥ 0,9: **acertou** |
| E4 | 0,42 [0,41; 0,44] | +0,041 [+0,022; +0,062]; TOST **falha** | 4,4% contra 1,8% na mesma temperatura; resíduo +2,6 pt [−1,4; +7,2], falha | τ_eff 0,35–0,65: **acertou**; resíduo de ADE₁ dentro de ±0,05 m: **falhou** (a estimativa pontual fica dentro, mas o limite superior do IC passa) |
| E1, E2 | abaixo da grade (spread < τ=0,3) | — | — | — |

**Temperatura e erro.** Baixar a temperatura praticamente não muda o ADE₁ do Qwen: 0,001 m entre τ=1 e τ=0,42. Logo o E4 não tem ganho de temperatura e paga só o erro de discretização (+4 cm).

**Leitura.** O mapeamento passos → temperatura vale numa segunda família. O efeito sobre a acurácia não vale, porque o erro do Qwen é dominado pela média condicional e não pelo spread.

**Correção.** A afirmação anterior ("o sampler devolve a média condicional; não há temperatura para baixar") vinha do v1 inválido e foi retirada do paper (v5.8).

**Ressalvas.**
- Os clipes são do split test padrão. Não se sabe se o Qwen-Drive treinou neles.
- O planner SFT ainda está rodando.

### Qwen-Drive v2, planner SFT (27/10; artifacts/qwendrive_analysis_v2_20261027, RL + SFT)

**Referência.** Spread de 0,135 m; ADE₁ de 1,45 m; Heun-64 difere em no máximo 1,3 mm.

| Sampler | τ_eff [IC 90%] | Resíduo ADE₁ (m) [IC 90%] | Resíduo flips (pt) [IC 90%] | Ganho da temperatura |
|---|---|---|---|---|
| E10 | 0,98 | +0,000; TOST passa | 0,0; passa | — |
| E4 | 0,37 [0,36; 0,39] | +0,018 [−0,002; +0,039]; TOST **passa** | +3,5 [0,0; +7,5]; **falha** | 0,002 m |

**Previsões do adendo 12.**
- τ_eff do E10 e do E4: acertaram nos dois planners.
- Resíduos: falham o ADE₁ do RL e os flips do SFT.

**Leitura.** É a mesma do RL. Os passos mapeiam para temperatura, e a temperatura não altera o erro. O E4 custa precisão e muda mais decisões do que a temperatura explica.

**Paper.** Atualizado na v5.11.

## FlashDrive z-lab (stack otimizado oficial: W4A8, DFlash, streaming; 27/10)

Fonte: artifacts/flashdrive_analysis_zlab_v1_20261027. Mesmos 60 clipes, 4 janelas, bootstrap de 1000 por clipe.

| Sampler | τ_eff [IC 90%] | Resíduo ADE₁ (m) [IC 90%] |
|---|---|---|
| Cache | 0,65 [0,63; 0,67] | +0,008 [−0,042; +0,062] (TOST falha) |
| E4 | 0,64 | +0,011 [−0,046; +0,071] |
| E8 | 0,80 | +0,028 [−0,009; +0,069] |
| E10 | 0,84 | +0,021 [−0,010; +0,060] |

Comparações pareadas de ADE₁:
- cache − E8: −0,110 [−0,137; −0,082];
- cache − E4: +0,002 [−0,042; +0,047].

**Leitura.** O stack otimizado replica o checkpoint NVIDIA (τ_eff 0,64; ganho 0,118). Ressalva: 57/60 destes clipes estão no treino do Alpamayo.

## Closed-loop AlpaSim, interino (3 de 5 configurações; 27/10)

Fonte: artifacts/closedloop_analysis_interim_20261027. São 58 cenas comuns; 2 cenas falham a checagem de rota do AlpaSim em todas as configurações.

| Comparação (só não pareadas em temperatura) | Taxa de incidente | Discordantes | McNemar |
|---|---|---|---|
| E4 vs Heun-32 τ=1 | 0,31 vs 0,345 | 4:6 | p = 0,75 |
| E10 vs Heun-32 τ=1 | 0,36 vs 0,345 | 3:2 | p = 1 |

**Leitura.** Nenhuma diferença detectável.

**Pendente.** Os testes pareados em temperatura (C1–C3) precisam das rodadas τ = 0,84 e 0,65. A tentativa de 25/10 dessas duas falhou sem métricas, e a grade as estava refazendo na ordem certa: Heun-32 a τ=1 levou ~15 h. Às 11:46 interrompi a grade por engano; perdeu-se ~1 h da τ=0,84. Na retomada, a configuração E10 completa foi renomeada pela grade, porque nenhuma rodada agrega (2 cenas falham sempre), e voltou ao lugar sem perda de dados. A grade agora pula configurações com ≥ 90% de métricas por rollout e não sobrescreve logs. Relançada às 11:55 só para τ=0,84 e τ=0,65; o log original do E10 foi sobrescrito, mas as métricas estão intactas.

## Replicação fora do treino, Alpamayo 1.5 (adendos 13/13b; 27/10)

**Dados.** artifacts/paper_numbers_heldout_a15_v1_20261027: 120 clipes da partição *test* do alpamayo_aligned, seeds 9201–9206, bootstrap de 2000 por clipe. ICs de 95%; TOST a 90%.

| Sampler | τ_eff [IC 95%] | Efeito ADE₁ (m) | Resíduo ADE₁ (m) | Resíduo flips (pt) | TOST ADE₁ / flips |
|---|---|---|---|---|---|
| E10 | 0,85 [0,83; 0,86] | −0,175 [−0,210; −0,144] | +0,008 [−0,004; +0,019] | −0,4 [−1,6; +0,9] | passa / passa |
| E6 | 0,75 | −0,251 | +0,016 [−0,004; +0,038] | −0,5 | passa / passa |
| E4 | 0,66 [0,63; 0,68] | −0,313 | +0,029 [0,000; +0,060] | +0,9 [−0,9; +2,9] | falha / falha |
| Heun-3 | ≥0,97 (acima da grade) | +0,127 | +0,127 | +20,0 | falha |
| Heun-5 | ≥0,97 (acima da grade) | +0,040 | +0,040 | +12,8 | falha |

**Previsões do adendo 13.**
- Previsão 1 (τ_eff do E10 0,78–0,90; do E4 0,55–0,72; do E1/E2 ≤ 0,45): acertou.
- Previsão 2 (resíduo do E10 < 0,05 m e efeito bruto negativo): acertou.
- Previsão 3 (flips do E10 dentro de ±2 pt): acertou.
- Previsão 4 (Heun-3/5 com τ_eff ≥ 0,85): acertou.
- 13b: o E10 passa o TOST nas duas métricas no A1.5. No v1, o A1.5 E10 não passava no ADE₁.

**Leitura.** O resultado principal se mantém em clipes que o modelo não viu no treino, e fica mais forte que no v1.

**Pendente.** R1 e A2.
