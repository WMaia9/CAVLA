# Pré-candidato: representações por objeto aprendidas com consultas de restrição

**Estado atual: versão rejeitada como contribuição central.** O
[teste de composição e logs novos](RESULTADOS_COMPOSICAO_PARCIAL_E_HOLDOUT.md)
encontrou uma regra física sem rede melhor que o modelo aprendido. Os resultados
abaixo registram a investigação inicial e não devem ser lidos como aprovação
da versão atual para ICML.

21/10/2026. Protótipo de aprendizagem executado; novidade e contribuição de nível
ICML ainda precisam ser estabelecidas. O alvo atual é um predicado geométrico
restrito, não segurança nem viabilidade completa de direção.

Resultados consolidados, incluindo o controle de sobreajuste, seleção de
checkpoint e calibração, estão no [relatório da rodada](RESULTADOS_GEOMETRIA_E_EDICAO_20261021.md).
Os pares editados ainda não entram no objetivo deste protótipo. Sua contribuição
para aprendizagem por intervenção precisa ser demonstrada separadamente.

O [desenvolvimento seguinte](RESULTADOS_PARES_SOBREVIVENTES_V1.md) executou 45
treinos adicionais. O decoder com features da VLA venceu um controle de pixels
sem pré-treino, mas augmentação e consistência nos pares de sobreviventes não
trouxeram ganho claro. A supervisão composta parcial foi posteriormente treinada
e testada em logs novos; o controle físico derrubou a versão atual. A pipeline
geométrica permanece como baseline.

## Ideia concreta

Usar os pares de Counter-nuScenes como ponto de entrada para aprender quais
restrições um objeto impõe às trajetórias possíveis. Remover um agente altera
uma restrição; não determina uma nova trajetória correta nem exige que a ação
mude. Assim, o objeto da aprendizagem é um conjunto de consultas admissíveis,
em vez de uma direção arbitrária de mudança da trajetória.

O protótipo implementa uma parte dessa ideia:

1. Extração de features do VLM nas regiões dos objetos, com pooling ponderado
   pela sobreposição dos pixels nos tokens visuais.
2. Uma rede prevê um conjunto compacto de caixas orientadas ao longo do tempo.
3. Um módulo geométrico diferenciável avalia curvas consultadas nessas caixas.
4. A rede recebe apenas rótulos binários dessas consultas durante o treino,
   sem loss para imitar posições ou dimensões 3D anotadas.

Todas as variantes recebem caixas 2D, classe, raios de câmera e os mesmos
priors fixos de dimensões por classe. Features da VLA entram somente nas variantes
designadas. A comparação principal é com um MLP sem a estrutura geométrica,
usando exatamente as mesmas informações e orçamento de treino.

O módulo torna a resposta monotônica em relação ao horizonte: se uma curva
atinge a região de proximidade antes de 2 s, continuar a mesma consulta até 4 s
não apaga esse evento. Essa propriedade resulta da construção do predicado.
Não apresentamos isso como um novo teorema. Agregação de várias restrições e
aprendizagem sob remoções compostas ainda não foram implementadas.

## Dados e alvo

Foram projetadas as anotações 3D de nuScenes para as imagens locais. A associação
exige classe compatível, IoU >= 0,5 e diferença >= 0,15 para a segunda candidata.
Há 953 correspondências automáticas candidatas, não identidades de remoção
verificadas. Destas, 584 objetos em 150 cenas têm os 6,4 s de futuro anotado sem
extrapolação nem interpolação através de lacunas acima de 0,75 s.

O alvo é saber se uma curva de consulta passa a até 2 m da caixa orientada de
**um agente específico**, seguindo o movimento desse agente no log. Isso é
um predicado operacional preciso, mas não um rótulo de colisão do veículo:
não inclui o footprint completo do ego, regras de trânsito nem reação de outros
agentes à ação consultada. As curvas podem não respeitar a velocidade inicial
ou os limites dinâmicos do veículo.

As anotações 3D foram usadas para construir os rótulos binários e a avaliação.
Portanto, **não alegamos ter criado o dataset sem anotações 3D** ou ter medido
economia de anotação humana. Estamos estudando aprendizagem a partir de rótulos
parciais em um ambiente onde existe uma referência geométrica independente das
previsões da VLA.

Há 49.056 consultas, mas somente 150 cenas. Consultas do mesmo objeto não são
observações independentes. Todas as consultas e objetos de uma cena permanecem
na mesma divisão. Curvas e horizontes adicionais são reservados para testar
generalização dentro do modelo geométrico adotado.

## O que diferencia este teste dos anteriores

O estado global do token de trajetória não venceu a baseline 2D. Features
espaciais foram extraídas novamente no modelo real; não reutilizamos
`original_lm_hs.npz` como se contivesse um ROI original para cada objeto.
O código externo disponível salva esse arquivo uma vez por cena para um ROI;
ele não fornece, por esse motivo, todos os ROIs originais necessários.

As features novas vêm das camadas 8, 16, 24 e 36, escolhidas conjuntamente antes
da extração, com redução para 16 componentes ajustada apenas nas cenas de
treino de cada divisão. Não escolhemos a melhor camada olhando a avaliação.
O backbone ficou congelado. Estamos treinando readouts/decoders pequenos,
não fazendo fine-tuning do Alpamayo nem demonstrando melhora de sua política.

## Evidência inicial e controles

Com 1.500 updates, três divisões por cena e três seeds, o modelo geométrico com
features espaciais teve erro balanceado de 15,73% nas consultas novas. O readout
direto com as mesmas features ficou em 18,68%; a baseline direta com metadados e
priors ficou em 17,59%. O ganho contra o readout espacial apareceu nas três
divisões; contra metadados, em duas das três. São resultados exploratórios.

O controle de 6.000 updates no último checkpoint apagou a vantagem inicial.
Com checkpoint e limiar escolhidos somente em validação interna por cena,
o modelo geométrico espacial atingiu 15,03% contra 18,79% do direto espacial e
18,43% do direto com metadados. Esses resultados são exploratórios e têm tradeoff
entre falsos negativos e falsos positivos. Consulte o relatório consolidado
para todos os regimes e o controle adicional por log de gravação.

O controle por gravação manteve o ganho: 13,93% no geométrico espacial contra
18,79% no direto espacial e 18,98% no direto com metadados, após calibração interna.
São 50 logs sem compartilhamento entre treino/validação/teste em cada divisão.
Reutiliza a mesma amostra exploratória; não é uma confirmação independente.

O erro nas posições 3D latentes continua sendo alto. Bons rótulos de consulta
não identificam necessariamente a geometria física correta, sobretudo para
objetos que nenhuma consulta alcança. A interpretação legítima é uma
representação das restrições observáveis sob o conjunto de consultas, não uma
reconstrução causal identificada do mundo.

## A régua de novidade continua alta

- [FCSRL, ICML 2024](https://proceedings.mlr.press/v235/cen24b.html) já aprende
  representações orientadas à viabilidade para safe RL. Mudar o domínio não basta.
- [CabiNet](https://cabinet-object-rearrangement.github.io/) e SceneCollisionNet
  já aprendem consultas de colisão a partir de representações de cena/objeto.
  O nosso uso de imagem RGB, tokens de VLM e um bottleneck geométrico precisa
  demonstrar uma vantagem além de recombinar esses componentes.
- [CausalDriveBench](https://arxiv.org/abs/2609.32157) já oferece avaliação causal
  em nuScenes, incluindo trajetórias contrafactuais de referência. Uma nova
  tabela de sensibilidade não é uma contribuição suficiente.
- [SafeAlign-VLA](https://arxiv.org/abs/2605.19524) e
  [FailBank](https://arxiv.org/abs/2609.39820) já usam sinais negativos/corretivos
  para melhorar políticas. Treinar um head de restrições ainda não compara com
  o resultado comportamental desses métodos.
- [Safety Certificate against Latent Variables, ICML 2025](https://proceedings.mlr.press/v267/jing25b.html)
  já trata segurança com dinâmica parcialmente não identificável. Não podemos
  chamar um readout geométrico de certificado sem hipóteses e demonstrações.

## O que poderia tornar isso um candidato de paper

Um método que aprenda **restrições suficientes para decisões sob intervenções**
com supervisão parcial, mantenha a incerteza que os dados não resolvem e
generalize para remoções/combinações não vistas. Precisaria superar aprendizagem
direta de risco, supervisão 3D disponível, representações espaciais sem o método
proposto e regras geométricas, com os mesmos dados e custos claramente medidos.

Faltam: resultado construtivo original de aprendizagem/identificação ou
generalização; validação independente dos pares; transferência para outro
backbone e outras cenas; composição entre objetos; teste de resposta a edições
independentes; e integração com geração/seleção de ações. A avaliação final
precisa verificar comportamento, incluindo tráfego reativo quando se alegar
efeito da ação sobre o ambiente.

O controle de edição imperfeita permanece útil para não converter mudanças do
processamento da imagem em rótulos de relevância física. Ele sustenta essa
necessidade metodológica; não resolve sozinho os itens acima.
