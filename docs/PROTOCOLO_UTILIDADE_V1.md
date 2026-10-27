# Diagnóstico de utilidade de intervenções — v1

Registrado antes de calcular os ganhos das micro-adaptações. Não é um método
novo de aquisição: DataMIL (2505.09603), QoQ (2603.09056) e Learning Loss for
Active Learning já abordam seleção por utilidade/erro. Pergunta específica:
erro na resposta à intervenção prevê o benefício de corrigir um par melhor
que o erro da própria ação? Relações apenas categóricas bastariam para ranquear?

Usar os cinco checkpoints factuais de relações v1 (seeds 616 a 1020).
Por seed: 48 candidatos novos (12 de cada mecanismo: aproximação, fundo,
billboard, troca de faixa), independentes do treino e de 96 quartetos de
calibração, também balanceados entre esses mecanismos. Não consultar os testes
finais de relações v1. Os rótulos de todos os candidatos são acessados neste
diagnóstico; não reivindicar eficiência de anotação ou aquisição sem rótulos.

Antes de adaptar, medir quatro proxies por candidato: MSE da ação editada,
MSE do efeito, violação de relação categórica (mesma margem 0,01), e produto
interno do gradiente de treino do candidato com o de calibração (baseline
genérico de influência de primeira ordem, sem inversão de Hessiana).

O gradiente candidato usa 50% BC do quarteto e 50% BC factual completa.
A referência de utilidade é a redução da MSE nos quartetos de calibração após
40 passos AdamW (lr 0,001; weight decay 0,0001), cada passo com o quarteto
inteiro candidato mais 32 exemplos factuais amostrados. Cada micro-adaptação
reinicia no checkpoint original e usa a mesma sequência de replay por seed.
Sem seleção de checkpoint e sem ajuste de hiperparâmetros pelos resultados.

Reportar Spearman por seed entre cada proxy e ganho real, fração de ganhos
positivos e utilidade média dos oito maiores scores contra média de todos.
Sem intervalos baseados em tratar candidatos da mesma seed como independentes.

Gate exploratório da hipótese específica: erro de efeito ou relação categórica
supera correlação do erro de ação por >=0,10 em média e em >=4/5 seeds,
e o top-8 tem utilidade positiva em >=4/5 seeds. Mesmo se passar, investigar
transferência entre mecanismos e custo de anotação antes de propor algoritmo.
Este diagnóstico usa calibração para medir a utilidade; não é evidência de
generalização de um seletor otimizado nessa mesma calibração.
