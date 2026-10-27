# Sondagem semântica de veículos quase parados

Definido em 21/10/2026, após inventário de 240 agentes de desenvolvimento e 48 da avaliação anterior. Não há amostra suficiente para estudar partida futura: são nove e um eventos, respectivamente. Portanto, **não treinar nem alegar ganho em antecipação/decisão com esse recorte**.

Pergunta limitada: as features espaciais VLA originais codificam a distinção anotada entre `vehicle.parked` e `vehicle.stopped`, além de caixa/classe/centro/velocidade/orientação/tamanho? São estados anotados por humanos; não são medida direta de intenção mental. A própria anotação retrospectiva pode empregar contexto futuro.

Seleção fixa: duas anotações passadas recentes e velocidade estimada <=0,5 m/s, com último atributo passado parked/stopped. Mesmos três splits por log e validação interna anteriores. Todas as features são de uma imagem frontal original. Não usar futuros como entrada nem como critério de seleção.

Baselines: escore simples de velocidade passada; logística numérica; logística numérica + PCA16 VLA; logística numérica + Gaussian16; kernel RBF ridge com as mesmas três entradas. Incluir, se disponíveis e alinhadas, features RGB/gradiente PCA16 como controle adicional de imagem. Padronização/PCA somente no treino. Grade fixa, seleção pela AUC de validação, desempate pela ordem da grade; threshold escolhido na validação pelo menor erro balanceado. A seleção sempre pode escolher o escore constante.

Primário exploratório: AUC na avaliação anterior, média das três configurações selecionadas, com bootstrap pareado por log. Secundário: erro balanceado e CV agrupada. Os conjuntos já foram examinados; não são confirmação independente. Ganho de decodificação não implica que a política use essa semântica ou que melhore decisões. Nem readout, nem distinção parked/stopped são propostos como método novo; LOKI e trabalhos de representação semântica já cobrem intenções e estados.
