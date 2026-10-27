# Gate de erro numérico → comando, multi-episódio (registrado antes da coleta)

Objetivo: testar se o achado das duas janelas (solver barato muda o comando do PID) se repete em episódios CARLA independentes, e se um preditor simples (norma/cosseno do campo inicial, margem de ramo) já prevê esse erro. Sem esse resultado, o estimador aprendido de `CANDIDATO_INFERENCIA_COM_ESTADO_DO_CONTROLADOR.md` não se justifica.

Coleta: servidor CARLA 0.9.16 próprio (portas 2200+), mapas Town01, Town03, Town04, Town05, Town10HD_Opt; 4 spawns por mapa; velocidade-alvo ciclando 3/6/9/12 m/s; 20 NPCs estáticos; 2 contextos por episódio (episódio = unidade independente; contextos do mesmo episódio são correlacionados). Seeds de coleta e spawn fixados antes de ver resultados. Alvo: 40 contextos / 20 episódios.

Inferência: Alpamayo1.5 congelado BF16, seed de linguagem 123, seeds de ação 9201 e 9202 (ambas reportadas; sem seleção posterior). Solvers: Euler4/10, Heun5, RK4×2, Euler denso inicial 10, referência Heun160. Controlador: OfficialPIDFollower com buffers vazios e velocidade real; steering anterior 0.

Métricas pré-declaradas: fração de contextos em que Euler10 e Heun160 diferem em modo (acelera/freia) ou em |Δsteer|>0.05; correlação de Spearman entre o erro de comando e (a) cosseno inicial, (b) norma do campo, (c) erro XY de trajetória. Reportar por episódio, com IC por bootstrap de episódios. Nenhum filtro retrospectivo de contextos.

Limites: domínio sintético, lane-following scriptado, referência numérica não é ação correta, não há closed-loop nem claim de segurança.

## Adendo de coleta (antes de qualquer inferência)

Town03 (4/4 episódios) e Town05 (2/2 tentados) causaram segfault do servidor CARLA no carregamento ou durante o episódio; um servidor novo por episódio não resolveu. Esses episódios foram descartados por falha de infraestrutura (pastas em `carla_multi_v1_20261022/failed/`) e substituídos por quatro episódios de Town02 (spawns 7/31/55/83, índices 20–23). Nenhum resultado de solver existia no momento da substituição. Total esperado: 16 episódios / 32 contextos em 4 mapas.

## Adendo 2: dados reais PhysicalAI-AV e lista final de solvers (antes de qualquer inferência)

Acrescentamos o split `test` do PhysicalAI-Autonomous-Vehicles (NVIDIA), formato nativo do Alpamayo1.5 (mesmas 4 câmeras, 4 frames a 10Hz, 16 poses), sem domain shift de simulador. Amostra: 200 clipes, um por chunk (seed 20261022), t0=5.1s. Não verificamos independentemente que esses clipes não foram vistos no treino do Alpamayo; o split vem de `clip_index.parquet`.

Cada clipe traz o futuro real do motorista. Ele vira uma terceira referência (`human`), além de Heun160 (fluxo convergido) e Euler10 (política publicada). O futuro humano não é a única ação correta; serve para checar se mudanças de solver aproximam ou afastam o comando do comportamento observado.

Solvers, iguais nos dois conjuntos: euler:4, euler:8, euler:10, heun:5, rk4:2, early_euler:10, cached_euler:8 (estilo FlashDrive: passos frescos 0,1,6,7 e reuso da última velocidade nos passos centrais, 4 avaliações; aproximação nossa, não o código deles) e heun:160 como referência.

Questão principal para ICML: métricas médias de trajetória (minADE/ADE, como as usadas para chamar o FlashDrive de "near-lossless") escondem inversões de comando? E sinais por input já usados em trabalhos anteriores (cosseno ProbeFlow/FlashDrive, curvatura AdaVLA, norma) preveem essas inversões? Concorrentes adicionados após nova busca: [AdaVLA](https://arxiv.org/abs/2608.29208), [ELASTIC](https://arxiv.org/abs/2606.31132), [Coda / Fewer Steps, Better Actions](https://arxiv.org/abs/2609.21216), [Adaptive time step FM para planejamento](https://arxiv.org/abs/2602.10285), [AdaFlow](https://proceedings.neurips.cc/paper_files/paper/2024/file/fa9cfdb49f7de3dee54007f84c0745b1-Paper-Conference.pdf).

## Adendo 3: decisões lidas da trajetória (registrado antes dos resultados do PhysicalAI)

O gate do PID usa buffers vazios; perto do equilíbrio, acelera↔freia pode inverter por diferenças mínimas. Acrescentamos classes de decisão lidas do plano, independentes do controlador: longitudinal = parar (extensão ≤1,5m em 6,4s) / desacelerar / manter / acelerar (velocidade do plano em 2–3s versus atual, banda max(1m/s, 15%)); lateral = esquerda/direita (heading final >20°), desvio esquerda/direita (offset final >1,75m), reto. Baseline de "margem de ramo": distância normalizada do plano Euler10 às fronteiras, disponível na inferência. Sinais comparados: cosseno inicial e mínimo (ProbeFlow/FlashDrive), curvatura acumulada do campo (aproximação da ideia do AdaVLA), margem de ramo. Os CARLA já analisados com o PID foram vistos antes deste adendo; o PhysicalAI não.

## Adendo 4: tempo de comprometimento e suscetibilidade por janela (antes de executar no PhysicalAI)

Hipótese inspirada em especiação de modelos de difusão (Biroli, Bonnaire, de Bortoli, Mézard 2024): a classe de decisão do plano se fixa num tempo t_c do fluxo, e erros numéricos importam mais perto de t_c. Medidas por contexto e seed (6 seeds, 9201–9206): classe do endpoint previsto x+(1−t)v ao longo de Heun32; t_c = primeiro t a partir do qual a classe prevista coincide sempre com a final. Intervenções causais sobre Euler20: congelar a velocidade numa janela de 4 passos ([0,.2),…,[.8,1)), como cache, e perturbar o estado no início da janela (ε=0,05·‖x‖, direção fixa por seed). Predições pré-declaradas: (P1) suscetibilidade não é uniforme em t; (P2) flips concentram-se em contextos com t_c tardio; (P3) entropia da decisão entre seeds prevê flips numéricos. Falsificação: se a suscetibilidade for uniforme ou não relacionada a t_c/entropia, a linha de "comprometimento" é abandonada.

## Adendo 5: redesenho da probe de comprometimento (por custo, antes de resultados interpretáveis)

A versão Euler20 com perturbações custava ~16h para 200 clipes; foi abortada após 12 casos (nenhuma inversão; pasta `commitment_physicalai_v1_aborted_20261022`). Nova versão: base = sampler publicado Euler10; congelamento de 2 passos por janela ([0,.2) só congela o passo 1, pois o passo 0 precisa de uma avaliação); referência Heun16 com rastreio de classe do endpoint previsto; sem perturbações (no modelo solúvel normalizado elas não localizaram o comprometimento); acrescentado o cache estilo FlashDrive (Euler8, passos 2–5 reaproveitados). Mesmas 6 seeds.

## Adendo 6: confirmação em escala dos schedules (antes de baixar os dados novos)

Resultado no holdout 100–199 (2 seeds): informação da decisão + uniforme teve menos inversões e menor ADE à referência que o uniforme em todos os NFEs, mas só um IC de inversões excluiu zero. Também registramos uma análise posterior com vazamento: o sinal "classe em t=0 ≠ classe final" usava a classe do Euler10 completo (AUROC 0,88); na versão sem vazamento (contra a classe do próprio cache) a AUROC caiu a 0,48–0,51 e foi descartada.

Confirmação: 436 clipes `test` novos, um por chunk ainda não usado (seed 20261023). Curvas de calibração congeladas em `artifacts/schedule_curves_physicalai_v1_20261022.json` (sha256 no manifest da avaliação), sem recalibrar. Seeds 9201–9204; referência Heun80 inline com o mesmo ruído. Hipótese primária: decision_info_plus_uniform tem menos inversões que o uniforme no mesmo NFE, agregando NFE 4/6/8 (diferença pareada média, IC bootstrap por clipe excluindo 0). Secundárias: idem contra rescaled_entropic, e ADE à referência.

## Adendo 7: temperatura explícita vs implícita (antes de executar)

Achado exploratório (seção 5 do relatório): solvers tipo Euler encolhem 100% das amostras em direção à média condicional, e isso troca ADE₁ por cobertura (minADE_k) e por entropia da decisão. Teste: nos clipes de holdout 100–199 do v1, com 6 seeds, comparar o encolhimento implícito (Euler 1/2/4/6/10 com τ=1) contra temperatura explícita (ruído inicial ×τ, τ∈{1,…,0,4}) com Heun3/Heun5 (6/10 NFE), Euler10 e Heun32 (referência). Hipótese: no mesmo NFE, temperatura explícita + Heun domina ou iguala a fronteira (ADE₁, minADE₆, entropia da decisão) do Euler implícito. Se não dominar, o "controle de temperatura" não é método; resta a análise.

## Adendo 8: schedules na mesma temperatura efetiva (antes de ver a confirmação do adendo 6)

O adendo 7 mostrou que o Euler com poucos passos se comporta como temperatura reduzida. Logo, um schedule pode "reduzir inversões" apenas porque encolhe menos. Complemento: rodar os mesmos schedules em v1 100–199 com as seeds 9201–9206 e referência Heun32 inline, salvando os planos. Para cada schedule, medir a dispersão entre seeds e comparar inversões, entropia da decisão e ADE₁ com a curva Heun32·τ (temperature_probe_v1) interpolada na mesma dispersão. Critério: a informação da decisão só conta como método se o resíduo (schedule − curva de temperatura) for menor que o do uniforme no mesmo NFE.

## Adendo 9: replicação multimodelo da temperatura numérica (antes de executar)

Mesma probe (Euler 1/2/4/6/10, Heun32·τ) no Alpamayo-R1 (mesmo cache de 4 câmeras, clipes v1 100–159, 6 seeds) e, quando a GPU permitir, no Alpamayo2-Super (perfil oficial de 7 câmeras, streaming dos mesmos clip_ids/t0). Previsões: (i) amostras Euler encolhidas em relação ao Heun32·τ=1; (ii) Euler-k sobre a curva de temperatura (resíduo de inversões pequeno frente ao Heun-k); (iii) ADE₁ monótono no encolhimento. Os integradores rodam dentro do `diffusion.sample` de cada modelo, com o `step_fn` (e CFG, se houver) do próprio modelo.

## Adendo 10: Qwen-Drive-1.0 (antes da execução em escala)

Qwen-Drive-1.0-4B (planejador flow matching com x-prediction, Euler de 10 passos, piso 1−t ≥ 0,1). Cenas PhysicalAI construídas segundo docs/data.md do release (3 câmeras a 2 Hz, histórico de 1,5 s e futuro de 5 s a 10 Hz, velocidade/aceleração do interpolador oficial rotacionadas do referencial do mundo para o do ego). O comando de navegação é derivado do heading futuro registrado (|Δψ|>0,35 rad), igual para todas as configurações. O piso do release impede integrações com k>10 de chegar ao ponto final (verificado no smoke: Euler64 com dispersão espúria de 6,5 m); usamos piso = min(0,1; 1/k), idêntico ao release para k≤10. Smoke (1 clipe, RL): dispersão entre amostras de ~0,09 m no solver preciso, insensível a τ. Execução: planner-rl e planner-sft, v1 100–159, 6 amostras, Euler 1/2/4/10 e Euler64·τ. Pergunta: o RL colapsou a estocasticidade? Previsão da tese: sem dispersão, temperatura e solver têm efeito pequeno.

## Adendo 11: FlashDrive oficial na mesma temperatura (26/10, antes da execução em escala)

Código e checkpoints oficiais (z-lab/flashdrive, MIT; `z-lab/Alpamayo-1.5-10B` com W4A8 + DFlash + streaming; `nvidia/Alpamayo-1.5-10B` original), em modo eager. Por janela, a partir do mesmo raciocínio do VLM: cache do FlashDrive (Euler 8 com passos 3–6 cacheados), Euler 8/10/4 e Heun32·τ (τ ∈ {1; 0,9; …; 0,5}), 6 seeds. Clipes v1 100–159 do PhysicalAI, 4 janelas consecutivas a 10 Hz terminando no t0 do clipe (no stack z-lab a primeira só faz prefill). Smoke de 1 clipe: o cache se comporta como τ≈0,5–0,55 nos dois checkpoints, com ADE₁ igual ao da curva de temperatura. Análise: decompor a melhora de minADE₁ reportada em (a) diferença entre checkpoints na mesma temperatura e (b) efeito do cache medido contra a curva Heun32·τ.

### Adendo 11b: previsão e análise (gravadas às 20:48:53, no mesmo comando da análise; ver correção abaixo)

O checkpoint nvidia terminou (60 clipes × 4 janelas). Nenhuma métrica foi calculada ainda. Unidade = clipe (média das janelas); referência = Heun32·τ.
- τ_eff de cada configuração = τ em que o spread médio entre seeds da curva Heun32 (interpolação linear em τ ∈ [0,5; 1]) iguala o da configuração. Se ficar fora de [0,5; 1], é marcado fora da faixa e não há casamento.
- **Previsão:** o cache do FlashDrive (flashcache8) e Euler 4/8/10 casam com a curva. O resíduo de ADE₁ (config − Heun32 em τ_eff) fica dentro de ±0,05 m, com intervalo de 90% (bootstrap de clipes, 1000 reamostragens, τ_eff recalculado em cada uma) contido na margem. Para flashcache, espera-se τ_eff ≈ 0,5–0,6.
- Correção no z-lab: a sonda não reiniciava o estado de streaming entre clipes (crash no clipe 101). O estado agora é reiniciado por clipe; o clipe 100 já tinha rodado com estado novo e é mantido.
- **Correção (20:53):** o horário "20:55" do adendo 11b está errado. O arquivo foi salvo às 20:48:53, 3 s antes do summary.json, no mesmo comando. Nenhuma métrica do FlashDrive tinha sido calculada antes disso, mas o pré-registro é do mesmo minuto da análise, não anterior a ela. Os clipes 100–159 são os mesmos 60 da amostra principal; Euler 10 a 0,84 é uma reexecução, não uma réplica.

## Adendo 12: Qwen-Drive v2, com a referência dentro do tempo treinado (27/10, gravado antes da rodada em escala)

**Motivo.** A referência v1 (Euler-64) consultava t > 0,9, faixa em que o planner nunca foi treinado (x-prediction, t ≤ 0,9, piso 1−t ≥ 0,1). O v1 também rodou sem raciocínio, em bf16 e com rótulos de seed errados. Ver paper/reviews/cited_papers_review.md.

**v2** (src/cavla/qwendrive_temperature_probe_v2.py):
- referência = Heun-32 em [0; 0,9] mais o salto final para o endpoint em t = 0,9;
- τ ∈ {1; 0,9; …; 0,3}, com Heun-64 a τ=1 como checagem de convergência;
- E1/2/4/10 = o sampler oficial;
- modo reasoning_planning (raciocínio greedy, compartilhado entre samplers);
- expert em fp32 com SDPA;
- seeds 42–47;
- clipes v1 100–159;
- planners RL e SFT.

**Já visto antes deste registro.** Smoke do clipe 100 (RL):
- spread E10 0,149 contra 0,152 da referência;
- E4 0,069;
- Heun-32 e Heun-64 diferem no máximo 0,6 mm.

**Previsão.** Mesmo protocolo de τ_eff casado por spread e resíduos com bootstrap de clipes:
- E10 com τ_eff ≥ 0,9;
- E4 com τ_eff entre 0,35 e 0,65;
- resíduos de ADE₁ dentro de ±0,05 m e de flips dentro de ±2 pontos, com a ressalva de que os flips são raros num modelo de spread tão baixo.

**Se falhar.** O Qwen-Drive é reportado como um modelo em que a regra não vale, com o motivo.

## Adendo 13: replicação em clipes fora do treino do Alpamayo (27/10 02:48, gravado antes de qualquer rodada nestes clipes)

Motivo: 57/60 clipes da amostra principal (context_100–159 de physicalai_test_contexts_v1) e 391/436 da confirmação estão na partição *train* de splits/alpamayo_aligned.parquet. A nova amostra artifacts/physicalai_heldout_contexts_v3_20261027 tem 400 clipes, um por chunk, todos na partição *test* (conferido 400/400), semente 20261027.

Desenho, idêntico ao v1 exceto pelos clipes: context_000–119 (120 clipes, primeiros em ordem de nome), mesmas configurações, mesmas sementes 9201–9206, mesmo código (cavla.temperature_probe para A1.5; cavla.multimodel_temperature_probe para R1 e A2), análise com cavla.paper_numbers (--range 0 120, bootstrap por clipe re-resolvendo τ_eff). Ordem: A1.5, R1, A2. Se houver tempo de GPU, context_120–399 entram como extensão, relatada separadamente; a análise primária é a de 120 clipes.

Previsões (a partir do v1):
1. τ_eff do Euler-10 entre 0.78 e 0.90 nos três modelos; Euler-4 entre 0.55 e 0.72; Euler-1/2 no piso da grade (≤0.45).
2. Resíduo de ADE1 do Euler-10 (sampler − referência na mesma temperatura) com estimativa pontual em |·| < 0.05 m nos três modelos; o efeito bruto do Euler-10 sobre ADE1 é negativo (menor erro que Heun-32 τ=1) nos três.
3. Resíduo de flips do Euler-10 dentro de ±2 pt (IC 90%) em A1.5 e R1.
4. Heun-3/5 (A1.5) com τ_eff ≥ 0.85.
Qualquer previsão que falhe é relatada como falha; se os números mudarem de forma material, o paper passa a usar os clipes fora do treino como resultado principal e o v1 vira análise secundária.

### Adendo 13b (27/10 04:53, antes de abrir qualquer saída dos clipes fora do treino; nenhum plans.npz destes clipes foi lido)

Critérios de equivalência acrescentados por sugestão da revisão, além das previsões pontuais do adendo 13:
- TOST a 90% com as mesmas margens do v1 (ADE₁ ±0,05 m; flips ±2 pt) para os resíduos de E4 e E10 nos três modelos.
- Previsão: E10 passa nos dois critérios em pelo menos 2 dos 3 modelos; E4 não precisa passar.

**Sementes.** As sementes 9201–9206 são as mesmas do v1, de propósito, para isolar o efeito da troca de clipes. Como os clipes são novos, as amostras de ruído não se repetem sobre as mesmas entradas.

**Margens.** As margens foram fixadas depois de ver os clipes do v1; nesta replicação elas são pré-especificadas.

### Execução (27/10 12:12)
R1 e A2 nos clipes fora do treino passam a rodar em 3 processos do mesmo código, em faixas disjuntas de clipes, mesclados depois (cavla.merge_probe_shards, que confere hash do código e seeds). Nenhuma mudança de código ou numérica: cada clipe roda como antes, só em paralelo.

### Execução do A2 no cluster (27/10)

**Onde roda.** O A2 nos 120 clipes fora do treino roda inteiro em H200 (cluster da UC Merced, partição test), em 4 shards de 30 clipes, com o mesmo código (caminhos configuráveis por variável de ambiente) e o mesmo snapshot do checkpoint.

**Checagem de hardware.** No clipe context_100 do v1:
- as trajetórias diferem das da estação local em ~0,09 m (mediana do máximo por trajetória), por arredondamento bf16;
- o spread entre seeds bate: E10 0,823 vs 0,822 m; Heun-32 1,288 vs 1,290 m;
- nenhum modelo mistura hardware dentro da replicação.
