# Teste restrito de custo em bits sob remoções

Registrado antes da execução de `maxplus_rate_probe_v1_20261022` e depois de observar os resultados da grade uniforme. É uma extensão adaptativa exploratória, não confirmação intocada.

Domínio: DAG completo ordenado, n eventos, pesos independentes em [1,1.4]. Encoder conhece todos os pesos. Família de respostas: todos os fechamentos max-plus após até k exclusões de vértices. Nenhuma VLA ou imagem participa deste teste.

Tolerância uniforme ε=0.02. Quantização por potencial usa grade factual δ0=ε/2; uma aresta que salta r vértices recebe grade δr=εr/k, para r>=1. Os erros de potencial telescopam e a soma de vértices saltados é no máximo k; logo erro<=ε/2+ε/2. ε<0.2 garante que subdividir uma aresta ainda aumenta o peso após quantizar. Limites e grades são informação comum ao decoder; contar os códigos inteiros, sem alegar compressão de pesos de rede.

Controles: grade uniforme com δ=2ε/(k+2); feedback clássico apenas na cadeia, com δ0=ε/[2(k+1)] e mesmas grades de salto. Este último também tem garantia<=ε: cada um dos no máximo k+1 segmentos da cadeia contribui<=δ0 e saltos somam<=ε/2. Assim, não alegar vantagem assintótica exclusiva dos potenciais.

Configurações: n∈{8,32,128,256}, k∈{0,1,2,4,8,16,32,64} válido; cinco grafos iid por configuração. Enumerar todas as exclusões em n=8; nos demais, amostrar 64 conjuntos por tamanho de exclusão, incluindo factual. Testar por fechamento completo no caso pequeno, e pela fórmula de caminho verificada nos testes de unidade nos maiores. Reportar máximo apenas sobre consultas testadas, com garantia algébrica separada.

Limite inferior de cobertura: a família de respostas revela cada uma das m_k coordenadas de peso. Responder com erro<=ε impõe erro<=ε nessas coordenadas. Um código de B bits cobre o cubo de lado η=0.4 com no máximo 2^B caixas de lado 2ε; portanto B>=m_k log2(η/(2ε)). Não requer encoder contínuo. Este é um argumento volumétrico clássico.

Hipótese de taxa: grades de salto proporcionais a r reduzem a soma de custos log(k/r). Como Σ_{r=1}^k log(k/r)<=k e m_k>=n(k+1)/2, o custo adicional fica O(m_k), e não m_k log(k). Para η fixo e ε pequeno, a taxa é m_k log2(1/ε)+O(m_k), compatível com o limite inferior até um termo linear. A prova depende desta classe específica; originalidade e interesse para aprendizagem ainda precisam ser estabelecidos. Quantização adaptativa, potenciais, discrepancy e feedback são anteriores.
