# Controles adicionais após a triagem de relações v1

O piloto principal terminou e passou o gate; a confirmação nas cinco seeds
predefinidas já está rodando. Este adendo é exploratório, escrito depois de
observar a triagem e antes de executar estes controles. Não substitui nem altera
o gate original.

1. `factual_3x`: continuação factual por 3.600 passos, conservadoramente mais
   tempo que os 1.200 passos de relações. Mesmo checkpoint e validação factual.
   Igualar passos não iguala FLOPs; tempo real será reportado.
2. `anchor_relation`: mesmas 96 relações, mesma margem/peso/passos. Aplicar a
   restrição diretamente sobre `ação_editada - ação_factual_conhecida`, em vez
   de sobre duas previsões. É uma baseline simples de supervisão parcial com
   âncora conhecida, sem alinhamento de representação nem alvo exato editado.

Usar todas as dez seeds do piloto e confirmação, sem alterar hiperparâmetros,
e reportar o erro por grupo. Se a baseline ancorada explicar os ganhos, atribuí-los
à informação parcial adicionada, sem promover a loss de diferença entre
previsões como contribuição nova. Mesmo que não explique, novidade não está
estabelecida: monotonicidade, corrective feedback e consistency são anteriores.
