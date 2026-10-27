# Porta de entrada: existe sinal visual além de cinemática?

Experimento exploratório de 21/10/2026, definido após a rejeição do candidato de composição. Os dois conjuntos já foram examinados. Nenhum deles é um novo holdout intocado. Não é uma contribuição metodológica proposta.

Pergunta: representações espaciais da VLA em imagem original melhoram a previsão do desvio da trajetória futura de um agente em relação à extrapolação de velocidade constante baseada estritamente no passado?

Dados: 578 instâncias distintas no desenvolvimento e 164 na avaliação anterior, sujeitos a disponibilidade de duas anotações passadas recentes. Usar os mesmos três splits por log e a validação interna anterior. Nunca estimar velocidade com anotações posteriores ao timestamp real da câmera. Mesma entrada numérica para todos: caixa 2D/classe, centro 3D, velocidade passada, orientação e tamanho. As entradas 3D são privilegiadas; esta é uma sondagem de informação incremental, não um sistema de percepção implantável.

Alvo supervisionado: 64 centros futuros menos a trajetória de velocidade constante. Isso usa supervisão 3D mais forte que os experimentos anteriores e NÃO é comparação de orçamento com eles. Imagem original apenas; pares editados não entram.

Controles: velocidade constante sem correção; regressão ridge linear e kernel RBF sobre entrada numérica; mesmas duas famílias acrescidas de PCA16 de representações VLA; mesmas famílias acrescidas de 16 variáveis gaussianas fixas. PCA e padronização ajustadas exclusivamente no treino de cada fold. Hiperparâmetros fixados no código e selecionados somente pela média de ADE por log de validação. Correção zero é opção para toda família, impedindo obrigação de piorar a base.

Primário: ADE médio por log, depois média dos três modelos na avaliação anterior. Secundários: FDE, ADE por agente e curva de erro no tempo. Bootstrap pareado por log, condicional aos modelos fixos; os três modelos não são três conjuntos independentes. Sem alegação de novidade, segurança, causalidade identificada ou ganho em política. Se o ganho só aparecer contra ridge fraco e desaparecer no kernel numérico ou no controle aleatório, não declarar valor incremental da VLA. Resultado negativo encerra esta sondagem; resultado positivo apenas justifica investigar uma hipótese de método.
