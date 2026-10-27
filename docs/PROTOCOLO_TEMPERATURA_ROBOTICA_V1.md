# Protocolo: temperatura numérica em políticas de manipulação (closed-loop)

Registrado em 22/10/2026, antes de qualquer rollout. Objetivo: testar fora da família Alpamayo, e com taxa de sucesso em closed-loop, a tese de `ESBOCO_PAPER_TEMPERATURA_NUMERICA.md`. Solvers tipo Euler com poucos passos equivalem a reduzir a temperatura do ruído inicial; ganhos de "menos passos" (Coda, arXiv 2609.21216: SmolVLA com 2 passos subindo o sucesso) seriam efeito de temperatura.

## Políticas e ambiente

- SmolVLA ajustado a LIBERO (`HuggingFaceVLA/smolvla_libero`) e π0.5 ajustado a LIBERO (checkpoint oficial do LeRobot, registrado no manifest), ambos congelados, via LeRobot 0.6.1 num venv isolado.
- LIBERO com as suítes registradas no manifest antes da execução; mesmas seeds de ambiente para todas as configurações (comparação pareada por episódio).

## Configurações (mesmo episódio, mesma seed de ambiente)

1. Sampler oficial (Euler com o número de passos default) com τ=1.
2. Euler com poucos passos: k ∈ {1, 2, 4}, τ=1.
3. Temperatura explícita com solver preciso: Heun com passos suficientes para convergir (definido por um teste de convergência offline antes dos rollouts), τ ∈ {1, 0,8, 0,6, 0,4, 0,2, 0}.
4. A temperatura do ruído é o parâmetro `temperature` já exposto pelo sampler, quando existir; caso contrário, o ruído inicial é escalado por τ.

## Medidas

- Sucesso por episódio (primária), com IC por bootstrap de tarefa/episódio.
- Offline, em observações registradas dos rollouts do sampler oficial: razão de encolhimento (distância à média condicional, Euler vs preciso) e τ efetivo pela dispersão entre seeds, como no Alpamayo.

## Predições (falsificáveis)

- **P1 (encolhimento):** amostras Euler-k ficam mais perto da média condicional que as do solver preciso, com τ_eff crescente em k.
- **P2 (temperatura explica o sucesso):** a curva sucesso(τ) do solver preciso, avaliada no τ_eff de cada Euler-k, prediz o sucesso de Euler-k. Diferença pareada |sucesso(Euler-k) − sucesso(preciso, τ_eff)| menor que |sucesso(Euler-k) − sucesso(preciso, τ=1)|.
- **P3:** se o sucesso aumenta com menos passos, então τ<1 com solver preciso reproduz esse aumento.

**Falsificação:** se Euler-k diferir do solver preciso no τ_eff correspondente tanto quanto difere do τ=1, a tese de temperatura não se sustenta em manipulação e deve ser restrita ao Alpamayo.

## Adendo 1: grade fixada antes dos rollouts em escala (22/10, após smoke)

Smoke (SmolVLA, LIBERO-Spatial tarefa 0): pipeline funcional, 9,3 s/episódio com 10 ambientes. Sonda offline do smoke (72 chamadas × 4 ruídos): 100% das amostras Euler encolhidas em direção à média condicional; razões 0,91 (Euler10), 0,77 (Euler4), 0,52 (Euler2), com τ_eff pela dispersão de ≈0,92, 0,78 e 0,53. P1 confirmada na manipulação (exploratório, uma tarefa).

Grade: política `HuggingFaceVLA/smolvla_libero`; suítes libero_spatial, libero_object, libero_goal e libero_10 (todas as 10 tarefas de cada); 10 episódios por tarefa (400 por configuração); seed de avaliação 1000 para todas. Configurações: Euler 10/4/2/1 com τ=1 e Heun16 com τ ∈ {1; 0,9; 0,75; 0,5; 0,25}. Ação executada conforme o checkpoint (replanejamento a cada passo, `n_action_steps=1`). Análise primária (P2): ajustar sucesso(τ) do Heun16 por suíte e comparar o sucesso de cada Euler-k com a curva no τ_eff (medido por sonda offline separada) contra o sucesso do Heun16 com τ=1. Driver retomável: `src/cavla/robot_temperature_grid.py`.
