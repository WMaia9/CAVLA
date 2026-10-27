# Candidata de método: aproximação de políticas generativas com fidelidade ao controlador com memória

**Aposta de investigação, não contribuição original estabelecida ou paper pronto.** Esta proposta surge dos testes novos com o Alpamayo1.5 e inputs multiview temporais completos no CARLA. Não reabilita a compressão de cache, a composição geométrica ou a antecipação temporal rejeitadas anteriormente.

## Pergunta que orienta o método

Como aprender quando podemos economizar avaliações de uma política generativa mantendo a decisão que será executada e controlando o erro acumulado na memória do controlador?

A política produz um plano, mas a execução é um programa com seleção de alvos, saturação, buffers PID, estado de parada e, em alguns agentes, seleção de amostras baseada no plano anterior. Desvio pequeno em média no plano pode mudar uma decisão; dois planos diferentes também podem produzir o mesmo comando saturado. Mesmo nesse segundo caso, os buffers internos podem divergir e produzir ações diferentes depois.

Formalização de trabalho: a política de referência G(c,z) produz o planoτ para contexto c e ruído z. O executor com memória m calcula `(u,m_next)=C(τ,m)`. Uma aproximação produzτ_hat. O alvo de fidelidade deve incluir tanto u quanto os componentes relevantes de m_next. As tolerâncias e o executor fazem parte da tarefa, não podem ser trocados depois de ver resultados.

O alvo primário de preservação da política publicada é o sampler nominalEuler10, quando esse for o sistema declarado. Um fluxo mais convergido é outra referência e pode produzir outro comportamento. Nenhum dos dois é automaticamente uma política segura.

## Evidência inicial que justifica a pergunta

Na janela1/seed9202, o mesmo contexto do Alpamayo1.5 levou o PID a acelerar comEuler10 e frear comHeun160. Promover o expert aFP32 não eliminou a diferença numérica principal. A janela0 respondeu de outra forma aos métodos de mesmo custo. São somente duas janelas correlacionadas de um mapa, com seed selecionada para diagnóstico.

Outra observação é mais específica: Euler4 eEuler10 produziram o mesmo comando saturado na janela1, mas velocidades desejadas diferentes,5.603 e4.611m/s. Uma loss que olhe apenas para o comando atual pode aprovar uma aproximação que modifica a memória do PID. Isso não é um novo teorema: é um requisito concreto para a contribuição de aprendizagem.

Os controles clássicos importam. RK4 e uma grade fixa mais densa no início já reduzem bastante o erro. O cosseno inicial do campo distingue as duas janelas; não há ainda evidência de que um método aprendido superaria ProbeFlow. Um artigo precisa passar por esses controles.

## Construção a implementar e falsificar

1. **Interface de referência explícita.** Fixar modelo, sampler nominal, executor e memória que será preservada. Começar com uma amostra fixa por input para separar a seleção com memória; depois incluir o seletor e contabilizar seu estado. Não omitir dependências para fabricar equivalência.
2. **Aprender sensibilidades das quantidades do executor.** Em vez de reconstruir todo o plano com a mesma precisão, estimar o impacto de um passo aproximado sobre heading do alvo, velocidade desejada, comandos e atualização de memória. Usar estados intermediários do action expert e resíduos de integração como entradas; referências e adjuntos completos são rótulos de treino, não operações gratuitas na inferência.
3. **Tratar ramos não suaves.** O índice de lookahead, clipping e modos de parada mudam a função. Certificados locais por derivada só valem enquanto os ramos permanecem estáveis. Uma fronteira incerta deve acionar refinamento/fallback. Avaliar também uma baseline intervalar puramente geométrica.
4. **Manter um orçamento de erro de memória.** Limitar o desvio da próxima memória e acompanhar sua propagação. Igualdade de comandos atuais não basta. Exigir uma análise da combinação com a dinâmica do controlador; não anunciar um limite de segurança física apenas com fidelidade ao modelo.
5. **Cobrar o custo completo.** Uma estimativa com backward tão caro quanto vários forwards não é aceleração. Treinar aproximações baratas, medir sua generalização e compará-las com solver de maior ordem, schedules fixos, caching e verificação já existentes.

Esta é uma especificação de pesquisa; ainda não há um estimador aprendido implementado ou validado.

## Onde teria de estar a contribuição de ICML

A contribuição exigida seria um método de aprendizagem de certificados úteis para programas de decisão com memória e ramos, acoplado à inferência parcial de geradores, com uma análise de propagação do erro e vantagem que sobreviva a solvers fortes e custos reais. Dual-weighted residuals, adjuntos, abstração de programas, early exit e calibração são antecedentes; a combinação verbal deles não estabelece originalidade.

Dois caminhos de análise são distintos: um limite determinístico exige hipóteses verificáveis de resíduos, sensibilidade e margem de ramo; um limite estatístico calibrado vale sob a distribuição/hipóteses declaradas. Calibrar frames de replay e depois mudar a política closed-loop não fornece automaticamente uma garantia para a distribuição induzida. Evitar chamar um quantil empírico de certificado universal.

## Uso dos dados locais e avaliação

- **Counter-nuScenes:** avaliar fidelidade à política nas imagens originais e editadas, incluindo mudanças de contexto que invalidam um atalho. Não usar a trajetória logada como ação correta do mundo editado. O protocolo de imagens estáticas preenchidas com zeros do arquivo é um estrato separado dos inputs completos.
- **nuScenes:** cenas e logs separados para treino/calibração/teste, conforme disponibilidade de câmeras; registrar limitações se somente a frontal estiver disponível. As gravações já usadas em exploração não viram holdout intocado.
- **CARLA:** gerar episódios independentes com inputs completos, executores fixados e latência cobrada. Primeiro testar fidelidade de comandos/memória, depois resultados de direção. Os dois inputs coletados nesta rodada não bastam para treino, validação ou alegações de generalização.

Referências obrigatórias: [ProbeFlow](https://arxiv.org/abs/2603.17850), [FlashDrive](https://arxiv.org/abs/2608.12932), [FlowCast](https://arxiv.org/abs/2602.01329), [Realtime-VLA FLASH](https://arxiv.org/abs/2605.13778), [Reactive Real-Time Flow Policies](https://arxiv.org/abs/2609.36540), [goal-oriented model-error estimation](https://ir.cwi.nl/pub/12312/). A revisão precisa ser aprofundada antes de uma reivindicação de novidade.

## Gate de avanço

Prosseguir somente se o estimador aprendido prevê erro de comando/memória em cenas novas melhor que normas/cossenos simples e uma baseline baseada em margem de ramo, e se isso produz vantagem de custo versus fidelidade frente a Euler/Heun/RK4/schedules/caching. O ganho deve sobreviver ao controlador em feedback com o mesmo orçamento de latência. Se uma regra ou solver clássico resolve o problema, preservar como ferramenta e abandonar essa implementação como núcleo deICML.

Relatório completo: [resultados da rodada](RESULTADOS_ACOPLAMENTO_E_SOLVERS_20261022.md). **É a próxima hipótese de método a desenvolver, não uma candidata já aprovada pela régua solicitada.**
