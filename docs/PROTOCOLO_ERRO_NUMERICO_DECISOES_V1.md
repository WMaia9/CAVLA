# Gate de erro numérico em políticas generativas

Registrado antes da execução de `native_solver_probe_v1_20261022`. Candidata a investigar: aprender/estimar o erro nas quantidades usadas para decidir, em vez de escolher esforço numérico somente por norma/cosseno do campo de velocidade. Dual-weighted residuals e adaptatividade orientada a objetivos são técnicas clássicas; ProbeFlow, FlashDrive e FlowCast já aceleram samplers. Aplicá-las ou acrescentar uma loss não estabelece novidade ICML.

Primeiro teste: Alpamayo1.5 congelado, imagens de quatro câmeras com quatro frames reais síncronos a10Hz,16 poses passadas convertidas consistentemente para sistema destro. Duas janelas de uma gravação CARLA isolada; não duas cenas independentes. Dados simulados têm domain shift. Estado de linguagem e ruído inicial do action expert iguais entre solvers. Nenhuma câmera zerada.

Euler com4,8,10,20,40,80 passos; Heun com4,10,20,40 passos, cobrando duas avaliações por passo. Duas seeds de ação por contexto. Oito dezenas de passos são referência numérica, não ação correta; medir40vs80 para avaliar convergência. A configuração publicada de dez passos também fica explícita como referência comportamental distinta. Sem fine-tuning, safety ou speedup estabelecido nesta etapa.

Guardar ações, trajetórias, custo medido, estados e campos da integração Euler10. Quantidades iniciais: avanço terminal, deslocamento lateral terminal, distância percorrida e perfil de velocidade. Examinar se erro escalar e erro global da trajetória têm ordenações diferentes. Não selecionar retrospectivamente um indicador favorável e chamá-lo de ganho de decisão.

Só continuar para um método aprendido se existir erro relevante que controles de mesmo custo não resolvam e se o custo de estimar/corrigir o erro puder ser menor que simplesmente resolver melhor a ODE. Comparar com Euler/Heun, schedules fixos, ProbeFlow/cosseno e caching antes de alegar avanço. Uma garantia numérica preserva o comportamento de referência especificado; não certifica segurança física da política.

## Adendo após v2: referência ainda não convergida e precisão

O Euler40 ainda diverge do Euler80, especialmente na seed9202/contexto1. Portanto,80 passos não podem ser assumidos como solução convergida. Acrescentamos Euler40/80/160/320 e Heun20/40/80/160, primeiro emBF16 e depois com expert/projeções/cache promovidos aFP32, mantendo o VLM congelado emBF16. Usar seed9202 em ambos os contextos é uma seleção diagnóstica posterior, não replicação independente. A promoção de precisão muda o campo implementado e não deve ser confundida com trocar somente o solver. Comparar convergência dentro de cada precisão e depois a sensibilidade entre precisões.

## Adendo de controles antes de qualquer treinamento

A extensão de convergência mostrou que Heun ajuda, e promover o expert aFP32 não elimina o caso difícil. Acrescentamos controles clássicos: Euler com grade t=u², Euler com t=(1−cos(πu))/2, com4/8/10 avaliações; RK4 com1/2/5/10 passos(4/8/20/40 avaliações); Heun5 para comparar exatamente com Euler10. Essas grades são controles fixos simples, não reproduções de ProbeFlow ou FlashDrive. A seed9202 permanece diagnóstica. Medir os custos sem concorrer com outros jobs novos.
