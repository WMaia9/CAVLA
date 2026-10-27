# Pares editados e restrições de objetos que permanecem

Protocolo exploratório congelado antes da extração das novas features e do
treino desta rodada. Usa cenas já exploradas; não é confirmação independente.

## Pergunta e alcance

As edições disponíveis melhoram a previsão de restrições dos outros objetos,
além de uma simples perturbação de features de mesma magnitude?

Selecionar uma edição por cena, pela primeira candidata na ordem SHA256 de
scene/stem que tenha ao menos um outro objeto com caixa 2D sem interseção com
a caixa alvo e no máximo 0,1% de pixels alterados no seu ROI. A escolha não usa
saída do modelo, rótulo de consulta ou desempenho. Não se usam rótulos de
ausência para o objeto editado. As identidades de remoção não estão verificadas.

O alvo dos sobreviventes é o mesmo predicado geométrico da imagem original,
calculado com movimento registrado, sem reação. Isso define uma tarefa
operacional de robustez a alterações de contexto; não prova invariância física.

## Comparações fixadas

Backbone congelado, pooling nas mesmas quatro camadas, PCA de 16 dimensões
ajustada somente em originais do treino, decoder geométrico e metadados/priors
idênticos. Usar as mesmas três divisões por gravação e seeds 17, 29, 43; até
6.000 updates; mesmo otimizador, minibatches e orçamento por método.

1. Original repetida: BCE nas duas cópias idênticas.
2. Augmentação por edição: média de BCE na original e no sobrevivente editado.
3. Perturbação de magnitude igual: média de BCE na original e em feature com
   ruído isotrópico com norma igual à diferença original/editada no espaço PCA.
4. Consistência pareada: BCE original mais MSE entre as probabilidades original
   e editada, com coeficiente 1 fixado, sem escolher pelo teste.

Para objetos sem edição elegível, as duas entradas são a original. Para o
controle de ruído, as magnitudes vêm dos pares do próprio treino, sem usar
estatísticas de validação ou teste. Todos os métodos são avaliados nas mesmas
originais e nos mesmos sobreviventes editados. Estes não contam como novas
cenas independentes nem novos rótulos independentes.

Checkpoint: menor erro balanceado a limiar 0,5 em consultas ID originais da
validação interna por log, a cada 250 updates. Calibrar o limiar apenas nessa
validação original, sem consultar edições de validação ou teste para seleção.
Reportar também limiar 0,5. Empates de limiar seguem a regra já implementada.

## Leitura dos resultados

Métricas principais: erro balanceado nas consultas novas das imagens originais
(todos os objetos e subconjunto de sobreviventes) e nas edições dos
sobreviventes; AUROC, falsos negativos/positivos e mudança de probabilidade.
Comparar métodos de forma pareada por gravação, sem tratar consultas como
observações independentes. Se a augmentação empatar com ruído de mesma magnitude,
não atribuir um ganho a informação causal dos pares. Se só ajudar em imagens do
mesmo editor, interpretar como robustez específica a esse editor.

Um ganho positivo ainda exigirá outro editor/perturbação, baselines com o mesmo
orçamento e amostra nova. Um ganho ausente rejeita este uso simples dos pares;
não prova que nenhuma aprendizagem por intervenção possa funcionar.
