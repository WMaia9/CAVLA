# Esboço de paper (ICML): o solver fixa a temperatura de políticas de flow matching

**Status: esboço de trabalho, 22/10/2026.** Os números vêm de `RESULTADOS_DECISAO_E_INFORMACAO_20261022.md`. Nada aqui é afirmação final.

## Título provisório

*Your ODE Solver Is a Temperature Knob: Decision-Level Effects of Accelerating Flow-Matching Driving Policies*

## Mensagem em uma frase

Acelerar uma política de flow matching com solvers de 1ª ordem, poucos passos ou cache não "perde precisão" de forma neutra: equivale quase exatamente a baixar a temperatura de amostragem. Isso muda a distribuição de decisões de forma sistemática (menos desacelerações e paradas, menos diversidade), e as métricas usuais dão veredictos opostos conforme contem uma ou várias amostras.

## Status da evidência (25/10)

- Direção (foco do laboratório): Alpamayo1.5, R1 e 2 com τ_eff(Euler10) = 0,84 / 0,84 / 0,83 e métricas na curva de temperatura. O Qwen-Drive é um contraexemplo informativo: fluxo quase determinístico, em que o erro numérico gera diversidade em vez de encolher.
- Robótica (secundário): encolhimento confirmado; sucesso em closed-loop pouco sensível a τ.

## Contribuições

1. **Decomposição temperatura + resíduo.** Na mesma dispersão entre seeds, o Euler com k passos é quase indistinguível do fluxo preciso com ruído inicial ×τ_eff (τ_eff=0,84 no sampler publicado do Alpamayo1.5; 0,65 no Euler4; ~0,5 no cache estilo FlashDrive). A temperatura explica ~94% das inversões de decisão do sampler publicado. Solvers de ordem maior com poucos passos não seguem esse padrão.
2. **Teoria para decisões.** A base Gaussiana (o Euler encolhe em qualquer grade; Claustre et al. 2026) é estendida a misturas. As inversões numéricas vão para a classe que contém a média (99,95% das amostras invertidas terminam mais perto da média); decisões protegidas por simetria nunca invertem; a taxa escala com k⁻ᵖ.
3. **Evidência numa VLA de direção real** (Alpamayo1.5, 10B, dados reais PhysicalAI-AV): 100% das amostras tipo Euler encolhem; parar −17% e desacelerar −13% com cache; entropia da decisão −0,14 bit (sampler publicado) e −0,24 bit (cache); a fragilidade fica confinada a contextos em que a política é incerta (0% vs 19%).
4. **Reconciliação da literatura.** "Fewer Steps, Better Actions" e ganhos de minADE₁ em acelerações são reproduzidos por temperatura explícita com solver preciso; minADE_k inverte o veredicto quando k cresce (consistente com minADE₁ melhor e minADE₆ pior relatados pelo FlashDrive).
5. **Temperatura pareada como protocolo.** Um schedule por informação da decisão passou numa confirmação pré-registrada (436 clipes), mas o ganho desapareceu na mesma temperatura efetiva. Todos os schedules testados ficam sobre a curva de temperatura. Isso é evidência a favor de comparar acelerações com τ pareado.

## Posição frente a trabalhos publicados (verificado nos PDFs)

- **FlashDrive** relata minADE₁ melhor (−0,13 m) e minADE₆ pior (+0,08 m), atribui isso a *"streaming fine-tuning acts as a regularizer that reduces prediction variance"* e diz que o cache dos passos do meio *"reduce[s] accumulated numerical error"*. Nossa medição sugere outra causa, a testar no setup deles: o cache estilo FlashDrive encolhe as amostras para ~0,50 da dispersão, o que sozinho produz esse padrão de métricas.
- **Coda** ("Fewer Steps, Better Actions") observa que mais passos não aumentam o sucesso, mas não discute temperatura, encolhimento ou diversidade. Nossa decomposição oferece o mecanismo.
- **Claustre et al. 2026:** o encolhimento do Euler para Gaussianas está na prova da Proposição 2 deles. É base a citar.
- **SubFlow / GMFlow:** perda empírica de modos raros e colapso para a média em poucos passos, em imagens. Não tratam decisões, direção das inversões nem políticas.

## Figuras planejadas

1. Diagrama: um passo de Euler = média condicional; k passos interpolam; ângulo cos θ.
2. Modelo solúvel: direção das inversões, entropia vs k e efeito da simetria.
3. Alpamayo: razão de encolhimento por solver (todas as amostras abaixo de 1).
4. Fronteira ADE₁ × minADE₆: curva Heun32·τ com os pontos Euler-k sobre ela e Heun-k fora dela.
5. Frequência de decisões por solver e temperatura (parar/desacelerar/acelerar).
6. minADE_k: vantagem do encolhimento em função de k.

## Lacunas antes de submeter

- **Closed-loop:** nenhuma avaliação fechada ainda. O AlpaSim local (`/mnt/datasets/alpamayo/alpasim`) e o CARLA estão disponíveis; o CARLA se mostrou instável em mapas grandes.
- **Segunda política:** replicar em outra política flow (π0/SmolVLA em LIBERO, ou o Alpamayo2) para mostrar generalidade.
- **Seletor de amostras:** o Alpamayo pode usar várias amostras com seletor; a temperatura muda o que o seletor vê.
- **Contaminação do split test:** não verificamos se esses clipes ficaram fora do treino.
- **Novidade:** confirmar que nem o paper do Coda nem o FlashDrive fazem a decomposição em temperatura; revisar a literatura de "DDIM variance / few-step diversity".
- **Prova da extensão para misturas:** hoje é empírica (modelo solúvel); falta proposição formal para a direção das inversões.
