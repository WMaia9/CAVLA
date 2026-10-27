# Holdout por gravação para o candidato de composição

Fixado antes de gerar suas novas labels/features ou consultar previsões.
Usar os 16 logs/119 cenas reservados em `reserved_log_holdout_v2_20261021`:
nenhum log pertence às 210 cenas com pares editados/resultados explorados pelo
projeto. Imagens brutas podem já existir; não se alega ausência do pretraining.

## Dados e exclusões

Usar o frame frontal e histórico já existentes no cache por cena, com tokens
em `scenes_ranked.csv`, para preservar o pipeline de observação anterior.
Não escolher um novo frame por desempenho. Exigir imagem frontal, histórico,
detecções 2D e proveniência. Registrar cenas sem esses arquivos.

Considerar detecções das mesmas seis classes. Aplicar a mesma associação
única classe/confiança, IoU projetado >=0,5, margem >=0,15, futuro de 6,4 s sem
extrapolar ou interpolar lacunas >0,75 s. Deduplicar por instância física com
maior IoU (desempate por stem). Selecionar até dez agentes por cena por ordem
SHA256(scene/instance_token), independentemente de labels ou previsões.
Isso muda a seleção em relação ao subconjunto com edições: reportar essa
diferença de protocolo e os números de cenas/agentes/logs após attrition.

Mesmas grades de 60 consultas ID e 24 com valores novos. Mesmas exclusões
simples/duplas/triplas; múltiplas retêm >=1 agente. Novo alvo de exclusão simples
por hash para diagnóstico; nenhum bit do holdout entra em treino/calibração.
São exclusões de restrições registradas, não imagens contrafactuais físicas.

## Modelos congelados e avaliação

Avaliar todos os checkpoints da composição v2 deduplicada: cinco métodos e
controle geométrico conjunto sem features, três seeds × três folds. Cada
checkpoint usa sua própria PCA, normalização e limiar do treino/validação
anteriores. Nenhum retreino, ajuste de threshold, seleção de seed ou arquitetura
com base no novo resultado. Avaliar também as referências de supervisão
individual, identificadas como mais fortes e com seleção diferente.

Reportar média dos nove modelos por método, sem chamar os nove de amostras de
dados independentes. Métrica principal: BER de exclusões múltiplas/consultas
OOD; separar duplas/triplas, FNR/FPR, AUROC e limiar 0,5. Reportar ponderação
uniforme de cenas como sensibilidade fixada. Reamostrar logs de modo pareado
condicionando nos modelos, sem alegar incorporar incerteza de todo o retreino.

Este teste confirma, se houver sinal, generalização para outros logs do mesmo
dataset sob um pipeline semelhante. Não resolve novidade ICML, benefício dos
pares de imagem, outro backbone, reação de tráfego ou melhora da política.
