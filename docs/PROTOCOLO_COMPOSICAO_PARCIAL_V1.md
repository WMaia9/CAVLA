# Piloto de composição sob supervisão parcial

Fixado antes de preparar os novos exemplos/treinar. Exploratória: mesmas 150
cenas, 50 logs e features originais congeladas, não confirmação independente.

## Informação fornecida

Cada cena tem o subconjunto de agentes candidatos com futuro completo de 6,4 s.
Para cada uma das 60 curvas/horizontes ID, o treino vê somente dois bits:
OR dos predicados dos agentes originais e OR após excluir um agente. O alvo da
exclusão é o da seleção prévia `survivor_pairs_v1_20261021`, quando disponível;
nas outras cenas, primeiro SHA256(scene/stem). Não escolher pelo resultado.

As labels são calculadas das anotações já disponíveis e ocultadas parcialmente:
isso simula supervisão fraca, não mede economia de anotação real. A exclusão
opera sobre a lista de restrições. Todos os modelos recebem a mesma imagem
original, caixas/classes/calibração/priors/features e a identidade a excluir.
Não recebem imagens contrafactuais de remoções múltiplas nem futuro geométrico.
Não é uma avaliação de interpretação de imagem editada ou tráfego reativo.

## Modelos e treino

- DeepSets: encoder por objeto/consulta; soma, média e máximo dos agentes
  mantidos; readout de conjunto. BCE média das duas observações.
- MLP por objeto + noisy-OR; mesma BCE.
- Decoder geométrico por objeto + noisy-OR; mesma BCE.
- MLP por objeto com verossimilhança conjunta exata dos dois bits.
- Mesmo decoder geométrico com essa verossimilhança conjunta.

Noisy-OR representa Bernoullis independentes condicionais como hipótese de
modelagem. A versão conjunta soma a probabilidade de todas as atribuições
compatíveis com os dois bits. Esse é um baseline conhecido de observação
parcial/group testing; não alegamos novidade da fórmula.

Sem pesos de classe em todos os objetivos. Dividir a NLL conjunta por dois,
o número de bits observados por grupo. AdamW lr=0,001, wd=0,01, clipping=10,
batch de 64 grupos cena/curva, 6.000 updates, seeds 17/29/43. Os mesmos grupos
de minibatch são usados em todos os métodos; registrar capacidade e custo.
PCA de rank 16 e normalização ajustadas só nas originais do treino. Reutilizar
as três divisões externas e validação interna por gravação já fixadas.

Checkpoint: menor BER a limiar 0,5 nas duas observações ID da validação, a cada
250 updates. Depois calibrar um limiar nessas mesmas observações de validação.
Não usar consultas de exclusão múltipla para seleção ou calibração.

## Avaliação fixada

Separar original, todas as exclusões simples e exclusões de dois/três agentes.
As múltiplas mantêm pelo menos um agente: excluir todos produziria um alvo
negativo trivial. Enumerar as combinações; separar curvas ID e 24 curvas OOD
(novos valores dentro dos intervalos). Nenhuma combinação múltipla é usada no
treino ou validação. Cenas/logs de teste são disjuntos dos de treino/validação.

Métrica principal: BER nas curvas OOD de exclusões múltiplas, com duplas/triplas
também separadas. Reportar AUROC, FNR/FPR, limiar 0,5, capacidade e todos os
resultados. Incerteza pareada por log com modelos fixos, sem tratar combinações
como cenas independentes. Não interpretar um melhor ponto médio como ganho
confirmado; considerar sobreposição de intervalos e escolhas exploratórias.

Os limites exatos usam, como diagnóstico privilegiado, os dois bits verdadeiros
da própria cena/curva de teste; as redes não recebem esses bits no teste.
Portanto, cobertura/ambiguidade desses limites não é comparação de risco sob
as mesmas entradas. Podemos estratificar erros pela ambiguidade sem passá-la
como feature. Não chamar isso de certificado de segurança.

## Critério científico

Se DeepSets ou MLP+OR explicarem o resultado, não atribuir ganho à geometria.
Se a versão conjunta não superar a BCE das observações, não promovê-la como
contribuição. Se houver ganho, testar em nova amostra e com baselines mais
fortes; somente depois acrescentar os pares de imagem e mostrar seu ganho
incremental. Esta rodada não estabelecerá novidade ICML ou melhora de política.
