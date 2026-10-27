# Protocolo exploratório: incerteza física e informação sobre dispersão

Escrito antes da execução desta rodada. Dados e avaliações já examinados; não é preregistro independente, método novo ou demonstração de segurança.

## A. Sensibilidade física dos predicados existentes

Pergunta: a supervisão binária por distância é estável sob pequenas mudanças contínuas dos centros futuros? Comparar nominal CV (somente estado passado privilegiado) e nominal futuro registrado (oráculo diagnóstico). Manter orientação/tamanho nominais; deslocar apenas centros. Predicado existente: distância assinada centro da curva–caixa <=2m em cada janela, seguido de OR dos agentes da cena. As 32 curvas antigas não satisfazem necessariamente dinâmica inicial, rota ou regras.

Envelope de translação: r(t)=r0+0.5*a*t², r0 em {0,0.25,0.5}m e a em {0,0.25,0.5,1,2}m/s². É uma varredura de hipóteses, sem calibrar valores com avaliação. Distância assinada a uma caixa é 1-Lipschitz em translação: min_t(d+r)<=2 é condição suficiente para proximidade sob todas as translações; min_t(d-r)>2 é suficiente para ausência de proximidade. Restante: certificado inconclusivo, não prova de ambiguidade atingível. OR conserva certificados. Não cobre mudanças de orientação/tamanho, agentes ausentes, reações ou footprint do ego. Não chamar conjunto de velocidades/acelerações fisicamente admissíveis na via.

Controle de conservadorismo: gerar oito acelerações constantes de módulo a e centro inicial inalterado. Se dois futuros gerados discordam, há testemunha de não unicidade do rótulo nesse subconjunto cinemático. Nominal está incluído. Tais perturbações não são amostras de comportamento real nem contrafactuais identificados.

Verificar reprodução exata dos rótulos armazenados quando r=0. Reportar proporções por cena/query e média por log, sem tratar queries como amostras independentes. Não ajustar decisões ou escolher o melhor envelope depois.

## B. Informação sobre dispersão, além de correção da média

Pergunta: a sondagem anterior falhou em reduzir ADE, mas as mesmas features conseguem antecipar a escala do erro cinemático? Alvos: resíduo 2D futuro–CV nos horizontes 1,2,4 e6.4s. Modelos Gaussianos isotrópicos, todos com média CV congelada. O objetivo é NLL por horizonte, média primeiro por agente e por log. A distribuição Gaussiana é hipótese de avaliação, não verdade nem probabilidade de colisão.

Baseline: variância global de treino por horizonte E[||erro||²]/2. Readouts linear ridge e RBF aprendem log(0.01+||erro||²/2), com as mesmas entradas19 numéricas, numéricas+PCA16 VLA, numéricas+Gaussian16. PCA/normalização só no treino. Correção multiplicativa de retransfomação calculada somente no treino: média exp(logalvo-predição). Variância mínima0.01m². Mesmos três splits antigos por log. Cada família pode escolher variância global pela validação interna; grades iguais à sondagem anterior. Escolha por NLL médio por log; nenhum ajuste no conjunto de avaliação. Predições de avaliação: mistura uniforme dos três modelos, NLL calculada como mistura (logsumexp), e não Gaussiana com variância média.

Este piloto testa uma limitação específica do ADE e não estabelece novidade. Só avançar para representação de incerteza se VLA melhorar NLL fora do log frente a numérico e ruído, sem depender de um único log; inspecionar variação por horizonte. Mesmo resultado positivo não demonstra calibração, valor de decisão, identificação contrafactual ou contribuição ICML. Resultado negativo rejeita este readout/protocolo, não toda informação visual.

## Barreiras para investimento maior

Não propor simplesmente information bottleneck, Mori–Zwanzig + RNN, conformal + reachability ou loss de decisão: já existem métodos diretamente relacionados. Qualquer candidato deve resolver diferença formal frente aos vizinhos, ser recuperável com dados disponíveis e vencer controles simples. Falta de novidade ou supervisão verificável impede promovê-lo, mesmo se uma métrica exploratória subir.

## Adendo após os resultados B: controle de distribuição comparável

O readout Gaussiano visual teve NLL4.212 contra4.504 do numérico na avaliação, mas piorou no CV. Uma mistura sem entrada selecionada em validação teve NLL3.074, rejeitando promover o resultado visual Gaussiano como candidato. A comparação não testa ausência de informação visual, porque muda a família de distribuição.

Para testar a informação com família comparável, congelar os componentes/variâncias da mistura incondicional escolhida por fold (k em1,2,4,8, já escolhidos só na validação); aprender somente pesos softmax dependentes de entradas numéricas, numéricas+PCA16 ou numéricas+Gaussian16. Otimizar NLL ponderada por log diretamente com L-BFGS, início em mistura incondicional; regularização L2 em{.001,.01,.1,1}, incluindo escolha de manter a mistura incondicional. Mesmas partições e PCA de treino. Seleção apenas por NLL na validação. Avaliar mistura uniforme dos três folds e CV fora do log; sem busca extra por horizonte. É experimento adaptativo pós-hoc, sem independência estatística das hipóteses anteriores. Não basta vencer em avaliação se CV contradizer o resultado. Um ganho consistente permitiria pesquisar método; gating de mistura não é novidade.
