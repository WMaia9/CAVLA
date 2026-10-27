# Protocolo: temperatura numérica em closed-loop (AlpaSim, Alpamayo 1.5)

Registrado em 25/10/2026, antes de qualquer rollout da grade. Smoke (1 cena, sampler oficial): pipeline completo funcional (renderizador NuRec, física, controlador, runtime, driver), ~0,8 s por decisão, ~200 decisões por cena.

## Pergunta

O efeito de segurança observado em replay não reativo (seção 9 de `RESULTADOS_DECISAO_E_INFORMACAO_20261022.md`: menos passos de Euler ⇒ menos colisões, explicado por temperatura) aparece em closed-loop, com agentes simulados e o ego dirigindo de fato?

## Desenho

- Simulador: AlpaSim (cópia em `.deps/alpasim`, commit affc2ea), renderer `nre-ga:26.04`, topologia 1 GPU, driver `alpamayo1_5` oficial (4 câmeras, sem CFG), 1 rollout por cena.
- Cenas: 60 da suíte `public_2604`, apenas clipes do split `test` do PhysicalAI-AV (459 elegíveis), sorteadas com seed 20261026 (`artifacts/alpasim/closedloop_scenes_v1.json`).
- Configurações (injeção por variáveis de ambiente em `alpasim_driver/models/cavla_sampler.py`; sem variáveis = sampler oficial):
  1. Euler10, τ=1 (oficial)
  2. Euler4, τ=1
  3. Heun32, τ=1 (fluxo preciso)
  4. Heun32, τ=0,84 (τ_eff do Euler10 medido open-loop)
  5. Heun32, τ=0,65 (τ_eff do Euler4 medido open-loop)
- Métricas do AlpaSim: `collision_at_fault`, `offroad`, `offroad_or_collision_at_fault`, `dist_to_gt_trajectory`, `dist_traveled_m`, `min_distance_to_obstacle_m`.

## Predições

- **C1 (temperatura reduz incidentes):** incidentes(Heun32 τ=0,65) ≤ incidentes(Heun32 τ=1).
- **C2 (Euler4 ≈ temperatura):** |Euler4 − Heun32 τ=0,65| < |Euler4 − Heun32 τ=1| em `offroad_or_collision_at_fault`.
- **C3:** o mesmo para Euler10 com Heun32 τ=0,84.

Análise pareada por cena (McNemar para taxas binárias; bootstrap por cena para contínuas). Com 60 cenas, o poder é limitado; resultados nulos devem ser reportados como tal.

**Falsificação:** se o Euler4 diferir do Heun32 no τ_eff tanto quanto do τ=1, a explicação por temperatura não vale em closed-loop.

## Adendo 1 (26/10, antes de qualquer resultado da grade)

A primeira execução da grade falhou em todas as configurações sem produzir rollouts: a decodificação JPEG na GPU do driver (`image_decode_device: cuda`, padrão do config do Alpamayo 1.5 no AlpaSim) gerou erro CUDA 710 com 4 rollouts concorrentes. Passamos a usar `driver.model.image_decode_device=cpu` (padrão do schema do driver), o que produz os mesmos pixels decodificados. Teste com 4 cenas concorrentes: 4/4 rollouts concluídos, sem erros. As tentativas falhas ficam em `closedloop_v1/*.failed.0`.

## Adendo 2 (26/10, antes de olhar métricas de qualquer configuração)

Duas cenas do sorteio têm rota inválida nos dados NuRec e são rejeitadas pelo runtime em qualquer configuração: `clipgt-1661c375-…` (waypoint inicial a 27,4 m da trajetória original, limite 10 m) e `clipgt-964ebe15-…` (rota dobra sobre si mesma, 180°). Com uma cena falha o AlpaSim não agrega; a análise usa as métricas por rollout (`rollouts/*/*/metrics.parquet`) das **58 cenas comuns** a todas as configurações. Defeito de dado, independente do sampler.
