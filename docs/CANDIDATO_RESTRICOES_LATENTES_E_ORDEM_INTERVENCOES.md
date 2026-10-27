# Candidato teórico em avaliação: conservar restrições que ainda não estão ativas

**Status: hipótese de pesquisa, com proposição restrita verificada algebricamente e por enumeração. Não é novidade estabelecida, método ICML aprovado ou resultado de direção.** Esta linha passa a merecer uma investigação própria porque fornece um problema de representação mais preciso que trocar backbone ou acrescentar uma loss de efeito. A versão anterior de composição por OR continua rejeitada; seus resultados não são reabilitados aqui.

## Pergunta

Uma representação pode prever perfeitamente o caso factual e todas as remoções de um objeto, mas descartar restrições que passam a importar quando dois objetos são removidos? Quanto precisa conservar para responder a uma família maior de intervenções, e qual experimento revela o que falta?

A conexão com engenharia é com sistemas de eventos discretos e caminhos críticos: uma restrição que não determina o horário atual pode tornar-se o gargalo depois de outras serem retiradas. Em trânsito, tempos de liberação, precedência e espera podem ter essa forma em modelos específicos. Não assumir que uma VLA arbitrária ou os futuros reais obedecem a esse modelo.

## Modelo restrito que conseguimos verificar

Há n eventos ordenados. Uma aresta i→j com atraso w[j,i] impõe t[j]>=t[i]+w[j,i]. Com liberações externas b, o menor vetor que satisfaz as restrições é t=A*⊗b em álgebra max-plus. A matriz A* contém os maiores pesos dos caminhos. A operação experimental elimina um nó e suas arestas incidentes. **Eliminar uma restrição/nó é uma semântica especificada deste modelo; não é a mesma coisa que fixar o valor temporal de uma variável por um do-intervention convencional.**

No primeiro exemplo, há dois caminhos paralelos de peso2 entre origem e destino e uma possível aresta direta de peso1. Ter ou não a aresta direta produz exatamente o mesmo fechamento factual e após qualquer remoção individual. Remover os dois intermediários revela a diferença. Isso vale para qualquer vetor b, não apenas para uma execução escolhida.

Construímos a generalização com r caminhos paralelos e verificamos todas as remoções de ordem<r para r=2,...,6. A primeira ordem que distingue as redes é r. É um sanity check de identificação, não uma descoberta de que interações existem.

## Proposição de trabalho: capacidade cresce com a família de remoções

Considere o DAG completo ordenado, com todos os pesos independentes em um intervalo aberto dentro de[1,1.4]. Para suportar todas as remoções de até k nós, com0<=k<=n−2, basta conservar as arestas de distância ordinal até k+1. Seu número é

`m_k = (k+1)n − (k+1)(k+2)/2`.

Justificativa do limite superior: trocar uma aresta por duas aumenta o peso, pois2>1.4. Assim, entre dois nós remanescentes, o caminho crítico visita todos os nós remanescentes intermediários. Após até k remoções, cada salto desse caminho passa por no máximo k nós excluídos, usando uma aresta de distância ordinal<=k+1. Pesos de saltos maiores não influenciam essas respostas.

Justificativa do limite inferior exato: para qualquer aresta i→j com j−i<=k+1, remover todos os nós entre i e j usa no máximo k remoções e deixa o fechamento[j,i] igual ao peso dessa aresta. Portanto, a família de respostas recupera individualmente os m_k pesos independentes. Uma representação **contínua** da família inteira, com decoder exato, precisa distinguir um conjunto aberto de dimensão m_k; uma codificação contínua injetiva em dimensão menor é impossível. O argumento usa um resultado topológico clássico, não uma nova teoria de dimensão.

Para k=0, bastam n−1 pesos adjacentes. Isso não estabelece um limite de dimensão para prever **uma ação escolhida**: a família aqui contém todos os fechamentos após as remoções permitidas. Tampouco é um limite sobre número de parâmetros treináveis ou sobre rank de um cache VLA sem mapear rigorosamente sua interface para este problema.

Verificação:20 redes ponderadas por configuração, n∈{5,8,12}, k∈{0,1,2,3}, todos os subconjuntos de remoção de ordem permitida; fechamento completo versus banda preservada coincide. Cada configuração tem contraexemplo construído na ordem seguinte. As coordenadas de resposta que revelam pesos têm Jacobiana identidade de rank18 no teste n=8,k=2. Esses testes verificam a implementação e a construção; a prova vem do argumento acima.

## O que já foi testado na aquisição

Enumeramos todos1.024 DAGs ordenados unitários de5 nós, com oracle de fechamento completo. Existem885 classes observacionalmente equivalentes após factual+singles;248 grafos pertencem a classes ainda ambíguas. O exemplo completo deixa8 hipóteses e5 das20 consultas compostas sem identificação.

Uma testemunha construída por minimum vertex cut separa duas hipóteses compatíveis. No exemplo completo, ela precisa de3 consultas. O primeiro random irrestrito precisou de mediana12, mas esse controle desperdiçava consultas já identificadas. **Random restrito ao desacordo também precisa de3** nesse exemplo, e entropia exata precisa de3. Não vender a comparação contra random irrestrito como ganho do método.

Na auditoria uniforme de todos os grafos, entre casos inicialmente ambíguos, as médias foram1.129 consultas para testemunha por corte,1.105 para entropia e1.248 para random informativo. O corte não supera a entropia clássica em número de consultas. A enumeração continua exponencial; este código ainda não é um algoritmo escalável de aprendizagem.

## Codificação finita: construção implementada e limites restritos

Preservamos a família de intervenções sob orçamento finito de bits, sem acumular erro com o comprimento total do caminho. A construção implementada usa potenciais temporais cumulativos e error feedback, técnicas clássicas de codificação:

1. Calcular os tempos cumulativos s[i] do caminho adjacente e arredondá-los em grade δ; definir o erro de potencial e[i]=s_hat[i]−s[i].
2. Codificar arestas adjacentes como diferenças de potenciais arredondados.
3. Nas arestas de salto que precisam ser conservadas, arredondar w[j,i]+e[j]−e[i].

Ao longo de um caminho, os erros de potencial telescopam. Se os caminhos críticos continuarem visitando todos os nós remanescentes, há no máximo k saltos e o erro de resposta fica limitado por δ+kδ/2, em vez de crescer diretamente com n. Uma condição suficiente de margem é δ<(1−η)/4.5, com η=0.4. Não omitir essa condição. Potenciais de grafos, quantização e error feedback são conhecidos.

O teste de grade fixaδ=0.01 compara arredondamento independente, feedback somente na cadeia factual e feedback por potenciais, com o mesmo código inteiro por aresta. Em n=256,k=4, nos20 grafos iid e consultas amostradas por grafo, os maiores erros testados foram0.13289,0.04561 e0.02727, respectivamente; a construção por potenciais respeita o limite0.03. No exemplo deliberadamente alinhado para acumular erro, foram1.2495,0.0303 e0.0295. O grande ganho sobre arredondamento independente não pode ser apresentado como ganho exclusivo do novo candidato, porque feedback clássico já elimina grande parte do problema.

### Custo mínimo em bits, com hipóteses explícitas

Considere um encoder de B bits e decoder que aproxima todas as respostas permitidas com erro absoluto uniforme<=ε. A família contém consultas que revelam individualmente cada peso da banda. Portanto, cada código só pode cobrir um conjunto de pesos cujo diâmetro em cada coordenada é<=2ε. Um cubo independente de ladoη requer

`B >= m_k log2(η/(2ε))`, paraε<η/2.

Esse argumento de cobertura não exige encoder contínuo. É uma complexidade minimax de código de comprimento fixo, **não uma função rate-distortion de Shannon para uma distribuição de grafos**. O argumento volumétrico é clássico.

Para o limite superior, usamos precisão variável: potencial factual na gradeδ0=ε/2; uma aresta que salta r vértices tem gradeδr=εr/k. O erro de potencial nas extremidades é<=ε/2. Como a soma de vértices saltados ao longo de um caminho remanescente é<=k, os arredondamentos dos saltos somam<=ε/2. Assim todas as respostas têm erro<=ε. Os pesos codificados ficam em[1−ε,1.4+ε]; ε<0.2 garante que subdividir uma aresta ainda aumenta o peso. A regra é apenas para k>=1; em k=0, basta feedback cumulativo da cadeia.

Há n−r−1 arestas que saltam r vértices. O custo adicional de precisão é limitado pela soma desses números vezes log2(k/r). Como Σlog2(k/r)<=k log2(e) e m_k>=n(k+1)/2, a soma éO(m_k). Logo, paraη fixo eε pequeno, o código custa

`B <= m_k log2(1/ε) + O(m_k)`,

com constantes independentes de n e k e interface n,k,ε,domínio compartilhada. O limite inferior tem o mesmo termo principal. Isto é uma proposição derivada para esta classe muito específica, ainda sem revisão matemática externa ou originalidade estabelecida.

**Controle forte:** feedback somente na cadeia também alcança a mesma ordem de custo. Basta usarδ0=ε/[2(k+1)] e as mesmas grades dos saltos. Há no máximo k+1 segmentos factuais; seus erros somam<=ε/2, e os saltos somam<=ε/2. Os bits adicionais nos n−1 pesos adjacentes sãoO(n log(k+1))=O(m_k). Portanto, não há vantagem assintótica exclusiva da construção por potenciais.

Executamos26 configurações, cinco grafos iid por configuração, n∈{8,32,128,256}, k válido até64,ε=0.02. Em n=8, todas as exclusões permitidas foram enumeradas e o cálculo rápido de erro foi comparado ao fechamento completo. Nos maiores, consultas foram amostradas; máximo observado não é máximo exaustivo. Todas as condições testadas respeitaramε.

Em n=256,k=64: limite inferior48.151 bits, código uniforme144.950, precisão variável por potenciais93.287 e controle clássico94.817 bits. O controle clássico inclusive teve erro empírico menor nessa amostra; não concluir superioridade prática dos potenciais. Em k=0, a configuração conservadora de precisão variável custa mais que a uniforme. Os números contam payload com parâmetros da interface compartilhados, sem headers por grafo. Não medimos compressão de VLA, velocidade ou desempenho de direção.

Protocolo: [codificação sob remoções](PROTOCOLO_CODIFICACAO_REMOCOES_V1.md). Artefatos: `maxplus_quantization_probe_v1_20261022/`, `maxplus_rate_probe_v1_20261022/`. A extensão de precisão variável foi formulada após os resultados de grade fixa e é exploratória.

## O que precisa ser novo para valer um paper

A contribuição teria de ir além do exemplo e das técnicas clássicas: uma caracterização útil com pesos gerais, ruído/observação parcial e restrições desconhecidas; construção aprendível de uma representação adequada às intervenções; ou aquisição/codificação com vantagem demonstrável frente a entropia, identificação de sistemas e preservadores de caminhos. Esses pontos permanecem abertos.

Vizinhos conceituais consultados:

- [Max-linear models on directed acyclic graphs](https://arxiv.org/abs/1512.07522): a álgebra e problemas de identificação têm literatura anterior; conferir detalhes de quais grafos/coeficientes são recuperados.
- [Markov models for accumulating mutations](https://arxiv.org/abs/0709.2646): redes conjuntivas e inferência censurada de eventos não são novas.
- [Tail dependence of recursive max-linear models](https://arxiv.org/abs/1701.07351): identificação do DAG mínimo não equivale automaticamente a conservar todas as restrições redundantes sob a semântica de exclusão que escolhemos.
- [Efficient Discovery of Approximate Causal Abstractions via Neural Mechanism Sparsification](https://arxiv.org/abs/2602.24266): abstrações neurais fiéis a intervenções e sparsification já são objeto de método próprio.
- [Consistent Rounding of Edge Weights in Graphs,2016](https://ojs.aaai.org/index.php/SOCS/article/view/18389): otimizar arredondamento preservando custos/estrutura de caminhos, com ILP e heurística em redes viárias, já existe. A nossa classe e erro absoluto são distintos; isso não estabelece novidade.
- [Preserving Distances in Very Faulty Graphs,2017](https://drops.dagstuhl.de/entities/document/10.4230/LIPIcs.ICALP.2017.73): preservadores sob falhas e limites que alcançam estruturas compactas já são estudados. Negar pesos em um DAG transforma max-plus em menor caminho com pesos negativos; separar semântica e hipóteses de cada resultado antes de alegar diferença.
- [Fault-Tolerant Distance Oracles Below the n·f Barrier,2026](https://arxiv.org/abs/2603.24530): distinguir preservador-subgrafo e código arbitrário é essencial; um limite de arestas não limita automaticamente todo encoder.
- [The Discrepancy of Shortest Paths,2024](https://arxiv.org/abs/2401.15781): correlação de erros em somas de caminhos tem uma literatura matemática forte; generalizar a análise para grafos gerais exige confronto com ela.
- [The Causal Information Bottleneck, UAI2025](https://proceedings.mlr.press/v286/simoes25a.html): comprimir variáveis preservando controle causal já é uma proposta existente. Não vender o princípio geral de compressão causal como contribuição nossa.
- [From Mechanistic to Compositional Interpretability,2026](https://arxiv.org/abs/2605.08934): interpretação por complexidade sob fidelidade e distinção de fidelidade observacional/intervencional também têm formulação anterior.
- Identificação de sistemas max-plus, network tomography, grafos tolerantes a falhas e técnicas clássicas de codificação precisam de revisão específica. Ausência de resultado em busca por uma expressão não comprova uma lacuna.

## Uso legítimo do material local e gates

Counter-nuScenes tem edições e respostas da política, sobretudo remoções individuais. Elas não são medidas do fechamento de uma rede de restrições, nem ações corretas de mundos físicos editados. A geometria conhecida também fornece informação lateral que pode resolver ambiguidades que dados de resposta isolados não resolvem; a baseline geométrica vencedora anterior ilustra por que isso precisa ser controlado.

Executamos um primeiro gate de compatibilidade no arquivo: selecionamos169 originais com avanço>=5m, desvio lateral<=1m e progresso quase monótono. Entre1.899 edições de veículos/pessoas,122 atrasam em mais de0.5s algum marco de progresso previamente atingido pelo original;46 cenas apresentam ao menos um caso. Marcos2,5,10,20m, com alcance original até5.8s, deixam janela para observar o atraso no horizonte6.4s. A seleção pelo futuro original é diagnóstica, não coorte prospectiva ou protocolo de treinamento.

Isso contradiz a hipótese conjunta simplificadora de que as amostras arquivadas seguem a mesma rota, escolhem sempre os menores tempos viáveis e a edição apenas exclui restrições. Não refuta monotonicidade de conjuntos físicos viáveis: rota, utilidade e atitude a risco podem mudar; há ruído de amostragem e edições imperfeitas. As trajetórias escolhidas não identificam todos os tempos viáveis. Artefato: `maxplus_archive_gate_v1_20261022/`.

Assim, **não usar diretamente os tempos escolhidos da política como supervisão de uma rede de precedência**. Podemos aproveitar imagens, objetos e estados nuScenes para investigar inferência de papéis/restrições num problema devidamente identificado, ou avaliar a teoria em sistemas de eventos controlados. Nenhuma transferência para VLA foi demonstrada. A extensão necessária continua sendo um problema aberto; não iniciar treino grande apenas porque a prova restrita funciona.

O gate científico permanece: **um resultado teórico correto em uma classe restrita, mais uma implementação, ainda não estabelece novidade ICML.** Esta é a candidata de investigação mais concreta desta rodada, com mecanismo e testes definidos, não um paper já aprovado pela régua solicitada.

Artefatos iniciais: `artifacts/maxplus_intervention_probe_v1_20261021/`, `maxplus_intervention_probe_v2_20261021/` e `maxplus_capacity_probe_v1_20261021/`. **Veredito atual: candidata teórica para revisão, sem candidato AV/VLA de nível ICML validado.** O próximo avanço precisa resolver originalidade e aprendizagem sob observação parcial/decisão, em vez de apenas melhorar mais um quantizador da classe conhecida.
