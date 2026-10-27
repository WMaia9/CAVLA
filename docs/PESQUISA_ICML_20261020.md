# CAVLA: pesquisa e decisão de execução — 20/10/2026

## Decisão atual

Começar por experimentos que possam eliminar rapidamente uma ideia fraca. Não
congelar arquitetura, título ou alegação de novidade antes das comparações.
O primeiro piloto foi executado, incluindo uma rodada de convergência. A
formulação simples de loss de efeitos **não passou o critério para escalar**:
um contraste supervisionado genérico obteve menor erro na extrapolação de
severidade. Isso não elimina toda a hipótese de alinhamento, mas elimina a
justificativa para treinar imediatamente uma VLA grande com essa variante.

A direção prioritária de investigação passa a ser **aprender efeitos de
intervenções sobre políticas estocásticas com orçamento de amostragem limitado**,
separando resposta decisional, variabilidade do gerador e dependência da tarefa.
Esta é uma pergunta de pesquisa candidata, não uma lacuna inédita já demonstrada.
MMD, U-estatísticas, common random numbers e best-arm identification são
ferramentas existentes: empacotá-las em uma VLA não constitui novidade suficiente.

## O que a pesquisa acrescentou ao PDF

As linhas abaixo resumem fontes primárias consultadas. As implicações e decisões
são nossa avaliação, não conclusões dos autores citados. A triagem de novidade
inclui leitura de resumos e seções metodológicas relevantes; não é uma revisão
sistemática exaustiva nem substitui leitura integral antes de congelar o método.

| Trabalho | Contribuição existente relevante | Consequência para nós |
|---|---|---|
| [CVAA / What Do They See?](https://arxiv.org/html/2607.16938v1) | Auditoria por remoção de objetos; compara médias de trajetórias e discute estabilidade entre seeds | Repetir auditoria, trocar ADE por deslocamento ou simplesmente adicionar múltiplas amostras não basta. O arquivo white-box local auditado tem uma amostra por entrada; isso não caracteriza todos os experimentos do artigo. |
| [CofactVLA](https://arxiv.org/abs/2608.04396) | Deconfounding factual/contrafactual em representação e ação | Loss contrastiva ou remoção genérica de confundimento já tem concorrência direta. |
| [RAF-VLA](https://arxiv.org/abs/2609.17728) | Alinhamento com representação de quadros futuros para planejamento | Precisamos comparar princípios de supervisão com orçamento de dados e treino equivalente. |
| [Embodied Interpretability](https://arxiv.org/abs/2605.00321) | Atribuição interventiva, nuisance attribution e ligação com generalização | Diagnosticar causal alignment ou usar um estimador estocástico não é novidade por si só. |
| [Deep Bisimulation for Control](https://arxiv.org/abs/2006.10742) | Geometria latente para comportamento e invariância a detalhes irrelevantes | O argumento geral de representação relevante para controle é antigo. |
| [Causal Bisimulation Modeling](https://arxiv.org/abs/2401.12497) | Abstrações mínimas por tarefa usando estrutura causal | Precisamos distinguir alinhamento de efeitos de abstração causal para controle. |
| [Goal-conditioned bisimulation](https://proceedings.mlr.press/v162/hansen-estruch22a.html) | Equivariance funcional e abstrações condicionadas ao objetivo | Condicionar relevância pela tarefa também não é novidade isolada. |
| [Value Equivalence](https://arxiv.org/abs/2011.03506) | Modelos definidos pelo que preservam para planejamento | Um quociente de equivalência decisional exige relação explícita com esta literatura. |
| [CAST](https://arxiv.org/abs/2508.13446) | Relabeling contrafactual para instruction following | Melhorar seguindo instruções pode vir apenas de mais supervisão. |
| [When Vision Overrides Language](https://arxiv.org/abs/2602.17659) | LIBERO-CF e Counterfactual Action Guidance | A falha de grounding precisa de comparação direta, não apenas motivação. |
| [CRAFT / Same Scene, Different Task](https://arxiv.org/html/2610.00524) | Alinhamento de skills contrafactuais através do action expert | Aproximar representação contrafactual e comportamento já foi abordado. |
| [ECT / When Instructions Retrieve Trajectories](https://arxiv.org/abs/2609.39971) | Demonstrações e loss pareadas para generalização instruction–scene | É um dos concorrentes mais próximos para a proposta de quartetos. |
| [IMPACT-VLA](https://arxiv.org/abs/2609.15005) | Atribuição em trajetórias interativas e interações não aditivas | Medir interações multimodais com diferenças de diferenças já tem precedente. |
| [CausalDriveBench](https://arxiv.org/abs/2609.32157) | QA causal e trajetórias alternativas em driving | Um novo benchmark causal de nuScenes precisa diferenciar referências e perguntas. |
| [Distributed Alignment Search](https://proceedings.mlr.press/v236/geiger24a.html) | Subespaços e interchange interventions para abstrações causais | Patching ou IIT em uma VLA não deve ser apresentado como novo princípio. |
| [Mechanistic study / Action Atlas](https://arxiv.org/abs/2603.19233) | Probes, injeções e ablações em diversas VLAs | Um estudo mecanístico precisa ir além de decodabilidade e exemplos visuais. |
| [Does a Learned Corrector Beat a Simple Retreat?](https://arxiv.org/abs/2610.06921) | Placebo pareado e comparação com solução simples sob variabilidade estocástica | Uma melhora aparente tem de sobreviver a repetições e controles baratos. |
| [Kernel Two-Sample Test](https://jmlr.org/papers/v13/gretton12a.html) | MMD e testes de igualdade de distribuições | Usar MMD é baseline estatística, não contribuição nova. |

## Ideias comparadas

| Ideia | Pergunta científica | Risco de novidade / execução | Decisão |
|---|---|---|---|
| Efeitos calibrados em um head auxiliar | Prever efeitos a partir de diferenças latentes ajuda a política? | Pode funcionar como tarefa auxiliar sem alterar decisão; supervisão extra e bisimulation confundem atribuição | Implementada como baseline; não escalar isoladamente |
| Interações tarefa × mundo em quartetos | A política aprende que a mesma mudança visual tem efeitos distintos conforme a tarefa? | Muito próxima de ECT/CRAFT e interações de IMPACT; ganho pode ser apenas otimização | Piloto executado; variante simples não passou extrapolação após treino longo |
| Efeitos distribucionais sob orçamento de amostragem | Quais mudanças de política persistem ao marginalizar a aleatoriedade e como supervisionar isso com poucas consultas? | CVAA/Embodied já discutem estimativa estocástica; estimadores padrão não bastam como novidade | Prioridade para falsificação e desenho de contribuição; diagnóstico real iniciado |
| Necessidade/suficiência de subespaço para efeito decisional | O efeito decodificado é efetivamente usado na geração de ações? | IIT/DAS e estudos mecanísticos já cobrem muitos componentes; patching fora da distribuição pode confundir | Controle necessário, candidato secundário |
| Tail-risk / CVaR ou world model causal | Efeitos raros importam mais; composição de intervenções preserva dinâmica? | Exige referências de risco e rollouts confiáveis; custo alto antes do sinal básico | Adiar |

## Experimento mínimo de quartetos

Entradas: `(o,l0)`, `(oI,l0)`, `(o,l1)`, `(oI,l1)` para o mesmo estado exógeno.
Definimos `D_l = a(oI,l)-a(o,l)` e interação `J = D_l1-D_l0`.
No controle sintético há um bloqueador em uma de duas faixas; a instrução escolhe
a faixa. Aproximar o bloqueador muda o progresso permitido apenas na faixa
escolhida. Alterar um billboard muda pixels sem alterar a referência.

O piloto usa imagens 24×24, uma CNN e dois IDs de comandos. Não usa VLM
pré-treinada, linguagem natural aberta nem dinâmica física de direção.
Serve para falsificação de uma loss, não para validar uma VLA.

Seis métodos recebem os mesmos quatro rótulos e imagens: BC; head auxiliar de
efeito; efeito da política; interação mista da política; contraste supervisionado
genérico; interação com rótulos embaralhados. Modelo, inicialização por seed,
sequência de minibatches e passos são controlados. Checkpoints são escolhidos
por MSE de ação em validação ID independente. Os testes não selecionam checkpoints.

Foram executados 400 e depois 4.000 passos por método, três seeds em cada
rodada: 36 treinos ao todo. A rodada longa foi motivada pelas curvas de validação
ainda descendentes no piloto curto. Todo resultado continua exploratório;
aumentar o treino não converte os testes reutilizados em confirmação independente.

Com os quatro rótulos completos, efeitos são combinações dos próprios rótulos:
a loss extra testa viés de otimização, não uma nova fonte de informação causal.
Se o benefício aparece apenas em treino curto, o baseline correto inclui mais
treino de BC e comparação por tempo/consultas, além de passos.

## A direção estocástica que vale testar a seguir

Uma mudança em uma trajetória gerada pode refletir: mudança no significado da
cena, diversidade válida das ações, mudança do texto gerado, alteração na
sequência de sorteios ou efeitos numéricos. O protocolo precisa fixar o estimando.

1. **Resposta do algoritmo com ruído acoplado:** compara duas execuções com uma
   definição explícita de variáveis aleatórias compartilhadas. Seed global é uma
   implementação possível, mas pode não alinhar etapas após gerações de tamanhos distintos.
2. **Mudança da média de ações:** marginaliza o gerador; pode deixar de detectar
   mudanças importantes em variância ou multimodalidade.
3. **Mudança da distribuição de ações:** é invariante a reparametrizações do ruído,
   porém precisa de múltiplas amostras e não fornece a resposta correta do mundo.
4. **Efeito em sucesso/segurança:** exige referência externa e ambiente de
   execução adequado; nenhum dos diagnósticos anteriores o estabelece sozinho.

Não devemos treinar uma representação para reproduzir a sensibilidade de uma
política congelada e depois chamar isso de correção causal da política. Isso é
distilação do comportamento existente. O alvo normativo precisa vir de uma
referência independente ou de hipóteses de intervenção explicitadas.

Potencial de contribuição, ainda a demonstrar: um procedimento de aquisição e
aprendizagem que encontre efeitos decisionalmente relevantes com menos consultas,
com incerteza calibrada e validação em políticas generativas distintas. Para ICML,
precisaria oferecer algoritmo/resultado formal além das estatísticas padrão e
mostrar utilidade para generalização ou seleção de ações, não apenas uma nova métrica.

## Critérios para investir em uma submissão

- **Novidade:** ler integralmente CVAA, Embodied, ECT, CRAFT, IMPACT e os trabalhos
  de abstração mais próximos; escrever a distinção por estimando, supervisão,
  algoritmo e consequência, sem depender do domínio driving.
- **Alvos válidos:** conferir identidade do objeto, mudanças fora da máscara,
  instruções aplicáveis, câmera/tempo e procedência dos rótulos. Inpainting sozinho
  não fornece trajetória contrafactual correta.
- **Variabilidade:** múltiplas seeds por entrada, no-op repetido, ruído do texto e
  do action expert explicitados. Estimativas individuais e agregadas separadas.
- **Orçamento justo:** mesmos dados, rótulos de referência, consultas à política,
  capacidade, tuning e compute comparável. Incluir baseline genérica forte.
- **Generalização:** separar cenas, famílias de intervenção, severidades e
  mecanismos de nuisance; reservar confirmação nova depois de escolher o método.
- **Utilidade:** demonstrar melhora em ação/comportamento, não somente no head.
  Uma alegação de segurança interativa exige avaliação apropriada.
- **Escala:** após sinal robusto, ao menos uma VLA real e uma segunda arquitetura
  ou domínio que teste generalidade. Estes são critérios de nossa estratégia,
  não requisitos universais impostos pelo ICML.

Um único teste sintético positivo, uma correlação em logs ou um experimento de
uma cena não satisfazem essa régua. O próximo lote real deve cobrir várias cenas,
objetos e severidades, e comparar a estabilidade do estimando com o custo de
amostragem antes de fine-tuning caro.

## Infraestrutura e limitações atuais

Há uma RTX PRO 6000 Blackwell de aproximadamente 96 GB, nuScenes, doScenes,
imagens editadas e um checkpoint local Alpamayo 1 que já executamos. A existência
de outros diretórios de modelos não comprova que suas integrações funcionem.
O release original do [Alpamayo 1](https://github.com/NVlabs/alpamayo) não oferece
condicionamento de navegação explícito; o [Alpamayo 1.5](https://github.com/NVlabs/alpamayo1.5)
é candidato a verificar para o eixo de instruções. Não devemos apenas inserir
comandos no prompt do modelo antigo e presumir que esse uso foi treinado.

CARLA não é dependência do código iniciado. Caso a contribuição passe a ser sobre
consequências de trânsito interativo, será preciso escolher CARLA, outro simulador
ou experimento físico apropriado. A versão original do
[NAVSIM](https://arxiv.org/abs/2406.15349) é não reativa e sustenta outro tipo de evidência.

Não confirmamos deadline oficial de 2027. O [CFP de 2026](https://icml.cc/Conferences/2026/CallForPapers)
é referência histórica, não calendário de 2027. A pesquisa deve produzir primeiro
uma contribuição defensável; a decisão de submissão depende do sinal e do prazo real.
