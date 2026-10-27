# Cronograma até a submissão ao ICML 2027

Criado em 26/10/2026. **Prazo assumido: fim de janeiro de 2027**, pelo padrão dos anos anteriores; confirmar quando o call for papers do ICML 2027 sair e ajustar as datas.

Tese do paper: em VLAs de direção baseados em flow matching, o solver é um botão escondido de temperatura. Menos passos e cache equivalem a amostrar com temperatura menor, o que muda decisões e segurança e explica ganhos publicados (FlashDrive, Coda). Foco em direção; robótica é seção secundária.

## Estado em 26/10

| Peça | Estado |
|---|---|
| Encolhimento para a média (Alpamayo 1.5/R1/2, SmolVLA) | pronto |
| τ_eff consistente (0,84 nos 3 Alpamayos) | pronto |
| Decisões explicadas pela temperatura (open-loop) | pronto |
| Colisões em replay não reativo (eventos raros) | pronto, poder limitado |
| Contraexemplo Qwen-Drive | pronto; best-of-6 não verificado |
| Closed-loop AlpaSim v1 (60 cenas, 1 rollout, 5 configs) | rodando; base de 34% de incidentes nas 29 primeiras cenas |
| FlashDrive oficial | não iniciado |
| Variação do raciocínio em linguagem | não iniciado |
| Prova formal para misturas | não iniciado |

## Semana a semana

### Semana 0 — 26/10 a 01/11
- **27/10:** termina o closed-loop v1. Análise pré-registrada C1–C3.
- **27/10 — Gate 1 (closed-loop):**
  - com sinal (Euler4 ≠ Heun32 τ=1 e ≈ Heun32 τ=0,65 em incidentes): seguir para o closed-loop v2 ampliado;
  - sem sinal: discutir antes de gastar uma semana de máquina; o paper vira análise open-loop + replay, e reavaliamos o alvo.
- **27–30/10:** instalar o FlashDrive oficial e reproduzir os números do README (baseline e otimizado) nos mesmos 100 clipes do PhysicalAI.
- **30–31/10:** probe de temperatura com **várias seeds de linguagem** (raciocínio variando), Alpamayo 1.5 e R1.
- **Em paralelo, sem GPU:** começar a prova formal (direção das inversões em misturas sob Euler).

### Semana 1 — 02/11 a 08/11
- **FlashDrive na mesma temperatura:** checkpoint z-lab com e sem o cache de ações, e checkpoint NVIDIA, cada um comparado com o solver preciso no mesmo τ. Pergunta: o ganho de minADE₁ some com τ pareado?
- **Gate 2 (FlashDrive):** se o ganho for temperatura, entra como resultado central de reinterpretação; se não for, reportar honestamente e remover a afirmação.
- Desenhar e pré-registrar o **closed-loop v2**: ~150–200 cenas do split de teste, 3 seeds por cena, configs Euler10, Euler4, Heun32·τ{1; 0,84; 0,65} e o braço de **temperatura explícita** (Euler10·τ=0,6 com o parâmetro oficial do Alpamayo).
- Lançar o closed-loop v2 (~5–7 dias de máquina).

### Generalidade entre famílias de modelo (adicionado em 26/10)
Hoje a replicação positiva em direção está só na família Alpamayo (NVIDIA); o Qwen-Drive é caso-limite (quase determinístico) e o SmolVLA é robótica. Para responder à crítica de "uma família só":
- **26–27/10:** download do NAVSIM navtest (OpenScene test, câmeras e LiDAR, ~220 GB) e dos mapas nuPlan para `data/navsim` (em andamento).
- **28/10–05/11:** ReCogDrive (Xiaomi, ICLR 2026; VLM + planejador por difusão; checkpoints IL e RL abertos) e DiffusionDrive / DiffusionDriveV2 (HUST; planejador por difusão sem VLM) no navtest. Mesmo protocolo: encolhimento, τ_eff, resíduos e caudas, mais **PDMS** do NAVSIM como métrica de direção. Pré-registrar antes de rodar.
- **Secundário:** π0.5 no LIBERO, reaproveitando o pipeline do SmolVLA.

### Semanas 2–3 — 09/11 a 22/11
- Closed-loop v2 rodando (GPU ocupada).
- Qwen-Drive: verificar o best-of-6 no protocolo PhysicalAI deles, ou cortar a afirmação.
- Fechar a prova formal (ou registrar como conjectura com evidência numérica).
- Esqueleto do paper: título, abstract, contribuições, figuras principais.

### Semanas 4–6 — 23/11 a 13/12
- Análise do closed-loop v2 (McNemar e bootstrap por cena e seed).
- Se houver tempo de máquina: closed-loop com Alpamayo 2 (driver existe no AlpaSim) ou R1.
- **13/12: rascunho completo v1.**

### 14/12 a 03/01 — revisão e folga
- Revisão interna (orientador e colegas do laboratório).
- Experimentos para tapar lacunas apontadas na revisão (folga reservada para imprevistos).

### 04/01 a 17/01 — versão 2
- Texto, figuras finais, apêndice, reprodutibilidade (código, seeds, protocolos pré-registrados).

### 18/01 até o prazo
- Polimento, abstract, checklist do ICML, submissão.

## Riscos e respostas

| Risco | Resposta |
|---|---|
| Closed-loop sem efeito | reposicionar como análise open-loop + replay; avaliar se ainda é ICML |
| FlashDrive não reproduz ou o ganho não é temperatura | remover a reinterpretação, manter Coda e a análise geral |
| Máquina ou sessão cai | todos os jobs são serviços systemd retomáveis |
| Novidade questionada (Claustre et al. 2026, SubFlow, GMFlow) | deixar explícito que a contribuição é o nível da decisão, a direção autônoma e a reinterpretação |
