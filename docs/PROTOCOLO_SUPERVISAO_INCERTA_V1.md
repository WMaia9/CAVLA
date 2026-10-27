# Supervisão interventional incerta — teste de falsificação v1

Registrado antes de treinar ou consultar resultados desta rodada. Não propomos
CLIC, mistura latente ou classificação de confiabilidade como métodos novos.
O objetivo é testar se uma candidata de aprendizagem conjunta acrescenta algo
além de filtragem e robustificação genéricas, evitando investir numa contribuição
básica. CLIC (2502.07645), IVA (2508.16292), Learning from weak labelers as
constraints (ICLR 2025) e Ren et al. (ICML 2018) são antecedentes obrigatórios.

## Alcance

Usamos o gerador visual do piloto de relações: CNN e dois IDs de comando,
224 cenas factuais, 96 pares editados em cenas factuais conhecidas, validação
factual de 96 cenas e quatro testes independentes de 256 cenas. Não é VLA real,
nem contém tarefas com premissa falsa: este é um stress test de **restrições
erradas**, aproximando apenas uma consequência de usar pares inválidos.
Não interpretar ruído de rótulos como validação do problema semântico completo.

Há três regimes: relações perfeitas; 24/96 relações erradas escolhidas ao acaso;
24/96 erradas concentradas nas 32 aproximações do obstáculo. Em cada par corrupto,
a relação de velocidade para a instrução da faixa bloqueada é +1, em vez de -1
(aproximação) ou 0 (intervenção nula). As demais componentes são mantidas.
No regime estruturado, o problema concentra o erro exatamente nos pares que
ensinam extrapolação de severidade. A corrupção não usa os erros da política.

Somente um auditor simulado acessa a verdade para retornar um bit de
confiabilidade por par consultado: 32 consultas aleatórias, mesmas em todos os
métodos. Ele não retorna a relação corrigida nem ações contrafactuais exatas.
As outras 64 confiabilidades ficam ocultas ao treino, exceto no upper bound
explicitamente rotulado `oracle_filter`. O teste usa alvos completos apenas
para avaliação. Guardar objetos fracos e auditorias separados da referência.

## Métodos e configuração congelada

- `factual`: continuação factual, sem usar relações.
- `naive`: usa todas as relações como verdade.
- `audited_only`: usa só relações auditadas e aprovadas.
- `hard_filter`: classificador de confiabilidade, limiar 0,5.
- `soft_filter`: mesmos scores congelados como pesos suaves.
- `anchor_soft`: mesmos pesos e restrição sobre ação editada menos âncora
  factual conhecida, controle simples forte do piloto anterior.
- `robust_fixed`: mistura entre relação informativa e ramo não informativo,
  com confiabilidade constante estimada nas auditorias (bits conhecidos fixos).
- `joint_mixture`: mesma mistura, mas permite treinar o classificador em conjunto
  com a política, preservando BCE sobre auditorias. Hipótese candidata, não
  reivindicação de novidade.
- `oracle_filter`: todas as confiabilidades verdadeiras conhecidas; upper bound
  com supervisão adicional, não comparação de orçamento equivalente.

Scores: features congeladas do pré-treino (média dos embeddings, resposta prevista,
relações observadas e magnitude RGB da edição), sem referência oculta. MLP
32 unidades; 500 passos AdamW lr 0,001, weight decay 0,01; só as 32 auditorias.
Os scores dos pares auditados são substituídos pelo bit observado em todos os
filtros. Sem tuning sobre erro final de política.

Pré-treino factual 2.000 passos; adaptação 1.200; peso de restrição 0,5.
Mistura `-tau log[p exp(-loss/tau)+(1-p)]`, tau=0,0001. Ramo não informativo
tem likelihood constante 1. BCE conjunta tem peso 0,001; essa escolha é fixa,
não calibrada por resultados. A mistura pode colapsar para declarar pares
difíceis como ruins: medir isso é parte do teste. Reportar percentil/média de
confiabilidade dos pares verdadeiros/bons que não foram auditados.

Seeds novas: 1636, 1747, 1858, 1969, 2080. Checkpoint somente pela validação
factual, a cada 100 passos. Relatar custo e todas as variantes. Smoke com outra
seed fica fora das médias.

Primária: macro MSE nos três grupos não ID, como no piloto anterior; reportar
grupos separadamente porque severidade domina numericamente essa média.
Gate candidato: em corrupção estruturada, `joint_mixture` precisa reduzir a
média em >=10% contra **cada** um de `soft_filter`, `anchor_soft`, `robust_fixed`,
vencer cada um em >=4/5 seeds e não piorar ID médio em >10% contra `soft_filter`.
Se falhar, não escalar essa mistura como contribuição; se passar, confirmar em
novas seeds antes de escolher configuração ou fazer fine-tuning real.

## Verificação real paralela

Alinhar os arquivos locais de Counter-nuScenes ao sample/frame nuScenes por
proveniência e verificar a imagem. Cruzar doScenes mantendo campos de validade
e referente **sem rótulo**, pois ainda não temos anotação independente.
Métricas de alteração fora de bounding box são diagnóstico aproximado de edição,
não ground truth causal ou critério automático de relevância/supervisão.
Não inferir identidade de máscara por índice quando metadata/máscaras divergirem.
