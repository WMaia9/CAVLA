# Continuidade de ações com distribuição condicional preservada

Rodada exploratória de22/10/2026. Protocolo registrado antes do primeiro resultado deste teste. Não reutiliza o resultado de quantização como evidência desta hipótese.

Pergunta: é possível reduzir mudanças entre ações mantendo a lei da ação dada a informação especificada como estado? Autoregressive policies, noise relaying, RTC, transporte ótimo condicional e inovações gaussianas são antecedentes; nenhum destes princípios é anunciado como novo.

Primeiro gate: controle linear-gaussiano com posição e velocidade lateral, ação escalar de aceleração. Política congelada a=Kx+σz. Baseline sorteia z iid. Reutilização/correlação de ruído preserva N(0,1) marginal, mas pode correlacionar z ao estado que depende das ações anteriores. Preservar a distribuição marginal do ruído não demonstra preservar π(a|x).

Construção analítica: se (X,A_anterior) é conjuntamente gaussiano, o resíduo R=(A_anterior−E[A_anterior|X])/sqrt(Var(A_anterior|X)) é N(0,1) condicionalmente a X. Usar z=ρR+sqrt(1−ρ²)ξ mantém a distribuição condicional. Se a variância condicional colapsar, usar ξ novo. Este é transporte condicional/normalização por inovações clássicos; originalidade de um futuro método aprendido não está estabelecida.

Comparar iid, ruído AR(1)ρ=.9, normalização marginal da ação anterior, EMA da ação com peso anterior.9, e resíduo condicional comρ=.9 e1. Calcular covariâncias populacionais exatas por100 passos, dinâmica lateral discretizada dt=.1, políticaK=[−1,−1.5],σ=.3. Ruído de processo independente com desvio por coordenada0,.005,.02,.1. Não escolher hiperparâmetros pelo resultado.

Reportar variância de posição, velocidade, ação, diferença de ações, média condicional do ruído linear no estado e variância condicional. Verificar a igualdade da lei de estado/ação ao longo do tempo na construção condicional, dentro da precisão numérica. Caso isto falhe, corrigir ou rejeitar antes de escalar.

Simular50.000 trajetórias por configuração com cinco seeds para checar risco de sair da faixa(|posição|>.35) em algum dos100 passos. Igualdade de marginais em cada instante **não** garante igualdade da probabilidade de primeira saída, nem mesma lei condicionada ao histórico completo. O sistema é totalmente observado e o processo é artificial; não equivale a driving real.

Gate para continuar: redução de mudança de ação na construção condicional mantendo as covariâncias especificadas, evidência de alteração condicional nos controles de suavização, e revisão de novidade dirigida. Um resultado positivo constitui somente prova de conceito; falta aprendizagem fora do modelo gaussiano e utilidade em políticas reais.

## Adendo após v1, antes de v2: informação observada sobre a ação anterior

O v1 deu uma redução grande de mudança de ação ao condicionar apenas em posição/velocidade. Isso exige um controle mais forte: incluir a ação executada anterior na informação condicionante. Se ela é conhecida, sua variância condicional é zero. Preservar exatamente a distribuição da ação atual dada essa informação fixa também o custo esperado entre ações. Logo, não há melhoria gratuita desse custo.

O v2 acrescenta a normalização condicional em(posição,velocidade,ação anterior), mantendo os demais parâmetros. A hipótese é que essa variante volta exatamente à referência iid. Esse é um adendo adaptativo; os resultados anteriores permanecem preservados. Em uma VLA, é preciso auditar a interface real e o que o histórico permite inferir, em vez de omitir memória para obter ganho. Preservação de marginais em replay também não garante o comportamento em feedback.
