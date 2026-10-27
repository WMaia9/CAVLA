# Candidata: integração orientada à informação da decisão em políticas generativas

**Status (22/10/2026): forte no modelo solúvel; validação no Alpamayo1.5 com dados reais em andamento. Não é contribuição ICML estabelecida.**

## Tese

Políticas de flow matching (Alpamayo, π0, etc.) são aceleradas reduzindo, reordenando ou reaproveitando avaliações do campo. As métricas usadas para declarar a aceleração "sem perda" são médias de trajetória (minADE). O que o veículo executa, porém, é uma decisão (parar/seguir, desacelerar, virar). Propomos que:

1. Erros numéricos só mudam a decisão quando ocorrem enquanto o fluxo está ganhando informação sobre ela (janela de comprometimento ou especiação).
2. Logo, o compute deve ser distribuído pela taxa de informação **da decisão**, dI(decisão; x_t)/dt, e não pela informação sobre a saída inteira (tempo entrópico, Stančević et al., NeurIPS 2025), nem pelo giro do campo (ProbeFlow, cosseno), nem pelo perfil médio (cache do FlashDrive).

## Resultados no modelo solúvel (`commitment_theory*.py`)

Mistura de duas Gaussianas com rectified flow e campo em forma fechada; decisão = modo final.

- **Simetria ⇒ zero inversões.** Com prior simétrico, Euler com 2–32 passos não inverte nenhuma decisão. O campo é equivariante à reflexão, o hiperplano de decisão é invariante e o passo de Euler o preserva (homeomorfismo para h·L<1). Inversões exigem quebra de simetria.
- **Lei de escala.** Com assimetria, a taxa de inversão do Euler-k cai ∝ 1/k (5,1→2,5→1,3→0,6%) e a do Heun cai mais rápido, consistente com ordem 2.
- **Suscetibilidade localizada.** Congelar a velocidade numa janela causa mais inversões perto do tempo de especiação teórico (0,7/0,5/0,2/0,1 contra 0,69/0,51/0,33/0,20). A correlação com dI/dt é 0,82–0,96.
- **Schedules com o mesmo NFE (72 casos: d∈{2,16,64}, separação∈{1,2,4}, w∈{.65,.8}, NFE∈{3,4,6,8}).** Taxa média de inversão:

| Schedule | Inversões médias |
|---|---:|
| Informação da decisão + uniforme | **3,5%** |
| Informação da decisão | 3,7% |
| Uniforme | 5,0% |
| Tempo entrópico reescalado (NeurIPS 2025) | 7,8% |
| Tempo entrópico plano | 26,7% |
| Cache do meio estilo FlashDrive (NFE 4) | ~13% |
| Giro do campo (cosseno, ProbeFlow) | até 35% (v2) |

O tempo entrópico reescalado só compete em d=2. Em d=16/64, a entropia da saída é dominada por detalhes dentro do modo, e os passos migram para t≈0,8. O schedule da decisão não depende da dimensão.

## Novidade: o que já existe

- [Entropic Time Schedulers](https://arxiv.org/abs/2504.13612) (NeurIPS 2025): informação sobre x₀ inteiro. É a baseline teórica principal.
- [Spontaneous Symmetry Breaking in Diffusion](https://arxiv.org/abs/2305.19693) (2023) e [Dynamical Regimes](https://arxiv.org/abs/2402.18491) (2024): especiação/janela crítica em geração de imagens.
- [Concurrence of SSB and nonlocality](https://arxiv.org/abs/2605.04830) (2026): localiza a janela crítica com classificadores (condicionamento, não integração).
- Aceleração de VLAs: [FlashDrive](https://arxiv.org/abs/2608.12932), [ProbeFlow](https://arxiv.org/abs/2603.17850), [AdaVLA](https://arxiv.org/abs/2608.29208), [ELASTIC](https://arxiv.org/abs/2606.31132), [Coda](https://arxiv.org/abs/2609.21216), [AdaFlow](https://proceedings.neurips.cc/paper_files/paper/2024/file/fa9cfdb49f7de3dee54007f84c0745b1-Paper-Conference.pdf).

O que seria nosso: (i) o funcional de decisão como alvo da alocação; (ii) a separação teórica com o tempo entrópico, que cresce com a dimensão; (iii) o resultado de simetria/ordem para inversões; (iv) evidência em VLA de direção real mostrando que acelerações "sem perda em minADE" invertem decisões.

## Gate no Alpamayo (próximo)

Calibrar em PhysicalAI `test` 0–99 e avaliar em 100–199, com o mesmo NFE (4/6/8): uniforme; cache estilo FlashDrive; cosseno (ProbeFlow); tempo entrópico reescalado estimado pela perda de flow matching contra o futuro humano (Algoritmo 2 deles); informação da decisão estimada pela concordância da classe do endpoint previsto entre seeds (probe de comprometimento). Referência Heun160; também concordância com a decisão humana. Abandonar se a informação da decisão não superar uniforme e tempo entrópico fora da amostra de calibração.
