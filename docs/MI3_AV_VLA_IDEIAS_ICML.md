# Mi³, AV/VLA e Counter-nuScenes: pesquisa e escolhas de projeto

Consulta: 20/10/2026. Este documento incorpora a preferência do usuário por
reaproveitar o dataset de **What Do They See?**. As propostas abaixo ainda não
foram validadas experimentalmente. Os resultados anteriores continuam nos seus
respectivos relatórios; esta rodada acrescenta pesquisa e inventários reais.

## Recomendação

Usar **Counter-nuScenes + doScenes** para investigar aprendizagem com intervenções
que podem preservar, modificar ou invalidar a própria tarefa. A pergunta é:
**como aprender respostas corretas a mudanças visuais quando nem todo par
editado admite a mesma instrução e nem toda edição fornece uma ação correta?**

Essa aposta aproveita diretamente os ativos do laboratório. Não estabelece
novidade por combinar datasets, acrescentar uma loss ou permitir perguntas ao
usuário. A contribuição precisa sobreviver a IVA, ECT, CofactVLA e aos controles
simples que já explicaram quase todo o ganho do nosso piloto sintético.

## O que o laboratório já oferece

O [site do Mi³](https://mi3-lab.github.io/) destaca direção segura, interação
humana verbal/não verbal, acessibilidade, generalização e explicabilidade. A
[lista de publicações](https://mi3-lab.github.io/publications) sugere quatro
conexões particularmente úteis para este projeto:

| Ativo/trabalho | Capacidade que podemos aproveitar | Limite para o novo projeto |
|---|---|---|
| [doScenes](https://arxiv.org/abs/2412.05893) | Instruções ligadas a cenas nuScenes e referencialidade | Anotações retrospectivas; não são instruções prospectivas que causaram as trajetórias |
| [Planejamento com instruções, 2026](https://arxiv.org/abs/2602.04184) | Integração reproduzível de doScenes com OpenEMMA | ADE isolada não prova execução correta da instrução |
| [Gestos de trânsito](https://arxiv.org/abs/2504.10873) | ATG/ITGI, comunicação não verbal e contexto de destinatário | O experimento inicial de ATG usa um ator e vídeos curtos; generalidade exige extensão |
| [Work-zone intelligence](https://arxiv.org/abs/2606.08860) | Evidência visual, persistência temporal e saída para controle | Histerese/percepção embarcada não é, sozinha, uma nova contribuição de ML |
| [VisLED](https://arxiv.org/abs/2404.12856) | Seleção de dados e embeddings para objetos raros | Diversidade e active learning já são uma linha do próprio laboratório |
| [MTR-VP](https://arxiv.org/abs/2511.22181) | Contexto visual e múltiplos futuros | A distribuição de trajetórias não deve ser reduzida arbitrariamente a uma única ação |

Não inferimos acesso a dados privados, disponibilidade de participantes ou
capacidade de anotação a partir de projetos e financiamentos listados no site.

Uma observação metodológica útil do paper de planejamento: a Tabela I apresenta
ADE filtrada média de 2,929 com instruções contra 2,879 sem instruções; o resultado
de 2,732 corresponde à melhor instrução por cena. Os autores explicitam essa
distinção e discutem outliers. Portanto, a extensão precisa avaliar a instrução
efetivamente recebida, sem escolher a melhor frase olhando o resultado.

## O dataset indicado pelo usuário

[What Do They See? / CVAA](https://arxiv.org/abs/2607.16938) introduz
Counter-nuScenes: o artigo reporta **3.062 pares em 210 cenas**. Há imagens
originais e remoções individuais por inpainting; o trabalho estuda deslocamento
das trajetórias e propagação interna. Ele já emprega múltiplas amostras e seeds;
simplesmente repetir inferências ou substituir uma métrica não é uma extensão
suficiente. O protocolo publicado usa imagem frontal estática, com demais
câmeras zeradas: são efeitos no sistema avaliado, com alcance restrito.

Nosso arquivo local previamente auditado contém **2.975 pares/210 cenas**, uma
trajetória por entrada e seed registrada 123. Isso descreve esse arquivo, não
todos os experimentos do artigo. Os 87 pares de diferença precisam ser
reconciliados; ainda não sabemos se são filtros, versões ou arquivos ausentes.

O dataset fornece **intervenções visuais e respostas da política**. Não fornece,
por esse motivo, uma trajetória correta para todo mundo editado. A trajetória
logada original também não se torna verdade contrafactual após uma remoção.

## Conexão local efetivamente verificada

Examinamos os CSVs locais de doScenes, normalizando espaços nos nomes das
colunas. Encontramos 2.450 linhas não vazias e 2.421 pares únicos de cena e
instrução, envolvendo 732 identificadores de cena. São números dessa cópia
local; não substituem os totais de versões publicadas.

O cruzamento pelo identificador numérico nuScenes encontrou:

- 146 das 210 cenas locais de CVAA com alguma instrução do doScenes;
- 448 linhas correspondentes;
- **442 pares únicos de cena e instrução**.

Isso é uma junção de identificadores, não uma validação semântica. A instrução
pode se referir a outro instante ou a um objeto que não corresponde à máscara
editada. Ainda não validamos frames, referentes, viabilidade ou escopo temporal.
Esses 442 pares não são exemplos contrafactuais prontos para treinar.

Inventários: `artifacts/mi3_literature_20261020/local_doscenes_inventory.json`
e `cvaa_doscenes_overlap.json`.

## Ideia A — Aprender com intervenções de validade incerta

**Prioridade principal se Counter-nuScenes for o ponto de partida.**

Exemplos propostos, ainda sem anotação nos dados:

1. Instrução “siga o carro branco”; a edição remove o referente. A tarefa pode
   ficar indefinida. Não há motivo para supervisionar uma nova trajetória como
   se a instrução ainda pudesse ser executada normalmente.
2. A mesma instrução, removendo um veículo estacionado sem papel na tarefa.
   A instrução pode continuar válida, mas a alegação de invariância ainda exige
   verificar que a edição não alterou outras restrições relevantes.
3. Remover um obstáculo libera várias trajetórias aceitáveis. Não é necessário
   que a política mude, nem há uma única trajetória correta a imitar.

**Formulação candidata:** representar a supervisão como um conjunto de ações ou
restrições admissíveis e uma variável de validade da instrução. Aprender quando
a evidência permite usar a restrição, quando ela fica ambígua e quando a tarefa
precisa ser reformulada. A política não recebe como verdade a própria resposta
congelada após a edição. A implementação precisa distinguir pares apenas
perceptuais de intervenções físicas consistentes.

**O que poderia alcançar a régua de ICML:** um método de aprendizagem que
aproveite rótulos parciais e validade incerta, com análise explícita do erro
introduzido por pares inválidos, transferência entre mecanismos e melhora
comportamental com orçamento de anotação medido. A contribuição não pode ser
apenas `loss de sinal + classificador de validade`. A baseline ancorada simples
continua obrigatória; já reproduziu quase todo o ganho no toy.

**Concorrentes que tornam a distinção difícil:**

- [IVA / Do What?](https://aclanthology.org/2025.findings-emnlp.635/) já trata
  comandos com premissa falsa e esclarecimento. Logo, “perguntar se o objeto
  sumiu” não é novidade.
- [ECT](https://arxiv.org/abs/2609.39971) já aprende com demonstrações
  contrafactuais válidas. Precisamos demonstrar por que supervisão parcial e
  pares incertos mudam o problema e como o método resolve essa diferença.
- [CofactVLA](https://arxiv.org/abs/2608.04396) já propõe intervenções e redução
  de confundimento visual; uma nova loss genérica de alinhamento colide com ele.

**Primeiro experimento que decide se a ideia vive:** um lote pequeno, escolhido
sem olhar a resposta da VLA, com dois anotadores independentes verificando
alinhamento temporal, referente, validade da edição e conjunto de respostas
aceitáveis. Comparar factual, augmentação nula, restrição ancorada, uso ingênuo
de todos os pares e uso com validade. Rótulos duvidosos devem permitir abstenção.
Se o ganho desaparecer contra a restrição ancorada ou depender de anotações tão
caras quanto trajetórias completas, não escalar essa narrativa.

## Ideia B — Distinguir efeito do objeto e efeito da edição

**Melhor opção para aproveitar imagens e máscaras com pouca nova anotação de
ações, mas com necessidade de novas edições de controle.**

O objeto removido revela um fundo que não foi observado. Diferentes fundos
plausíveis e alterações colaterais podem gerar respostas distintas. Medir a
diferença entre dois inputs identifica a resposta àquela edição; atribuí-la à
ausência semântica do objeto requer hipóteses adicionais.

Experimento candidato: múltiplas edições independentes por objeto, no-op
reconstruindo a mesma região e preservando o objeto, controles de área/localização,
e referência controlada com renderizações limpas onde a ausência é conhecida.
Separar variabilidade da política, variabilidade da edição e incerteza sobre o
que conta como intervenção válida. Reportar estabilidade e abstenção em rankings.

**Potencial contribuição:** limites de identificação/sensibilidade para
atribuições sob intervenções imperfeitas, com hipóteses verificáveis e uma
decisão útil que melhore. A média de vários inpaintings ou uma subtração de
no-op não é, por si, um estimador identificado do efeito semântico. No-ops podem
ter distribuição de artefatos diferente da remoção.

[Explaining by Removing](https://www.jmlr.org/papers/v22/20-1316.html) é uma base
obrigatória sobre escolhas de remoção; [Embodied Interpretability](https://arxiv.org/abs/2605.00321)
já formaliza atribuição interventional e propriedades de estimadores. Esta busca
não certifica uma lacuna inédita em identificação parcial.

**Gate:** demonstrar que controles mudam atribuições de forma reproduzível e
correta na referência controlada, generalizando entre editores e políticas.
Se só mostrarem que imagens diferentes causam outputs diferentes, falta novidade.

## Ideia C — Explicar a propagação até o decoder de trajetórias

**Aproveita os pares e ativações, com acesso ao modelo necessário.**

O CVAA relata casos em que o output muda sem uma assinatura proporcional no
token analisado. Uma hipótese testável é que distâncias geométricas genéricas
não capturam direções às quais o decoder é sensível, ou que a rota observada
não cobre os mediadores relevantes.

Comparar cosine/L2, readout treinado, aproximação Jacobiana e intervenções reais
por activation patching. Testar necessidade/suficiência dos mediadores propostos
em cenas não usadas para descobri-los, com placebos e seeds acopladas. A meta é
explicar a política existente, não declarar que seu comportamento é correto.

**Risco de novidade alto:** [Depth-Wise Probing](https://arxiv.org/abs/2608.07361)
já usa o planner como lens no espaço de trajetórias;
[Encoded but Not in Control](https://arxiv.org/abs/2610.06235) já mostra uma
action lens e dissociação entre informação codificada e controle;
[IMPACT-VLA](https://arxiv.org/abs/2609.15005) já investiga propagação e interações.
Trocar cosine por Jacobiano ou por uma probe não basta. Avançar apenas se surgir
um mecanismo causal novo e um método de intervenção transferível.

## Outras linhas alinhadas ao lab, com menor reaproveitamento direto

**Validade temporal das instruções:** aprender estados de instrução pendente,
ativa, concluída ou revogada. doScenes já discute que uma instrução pode ser
relevante por apenas parte do vídeo. Counter-nuScenes tem edições estáticas;
seria preciso estendê-las no tempo. A ideia compete com
[SwitchVLA](https://arxiv.org/abs/2506.03574),
[TIC-VLA, ICML 2026](https://proceedings.mlr.press/v306/huang26cd.html),
[LIMP](https://arxiv.org/abs/2402.11498) e
[PAV](https://arxiv.org/abs/2610.21434). Em particular, PAV já aprende validade
de prefixos com supervisão de sobrevivência. Não vender hazard head ou horizonte
adaptativo como contribuição nova. Validade da instrução e confiabilidade do
prefixo da política são alvos diferentes; demonstrar essa diferença é trabalho.

**Esclarecimento orientado ao custo de agir:** escolher quando perguntar pelo
impacto das interpretações sobre ações futuras e pelo custo de esperar. Combina
com acessibilidade e gestos, mas
[KnowNo](https://arxiv.org/abs/2307.01928),
[UPS](https://arxiv.org/abs/2602.22474),
[BAL](https://arxiv.org/abs/2609.34270) e
[Conformal Decision Theory](https://arxiv.org/abs/2310.05921) cobrem grande parte
do espaço. “Agir, perguntar ou parar” não é uma nova proposta suficiente.

## Ideias que esta pesquisa enfraqueceu

- Um benchmark somente trocando instruções na mesma cena: [Language2Drive](https://arxiv.org/abs/2610.16634),
  publicado em 12/10/2026, já faz avaliação contrafactual pareada, incluindo
  persistência, contexto e critérios temporais. Essa colisão é direta.
- Simplesmente testar se raciocínio causal acompanha trajetória:
  [CausalDriveBench](https://arxiv.org/abs/2609.32157) já aborda QA causal e ações.
- Uma action lens ou decodificação por camada: concorrentes da ideia C já existem.
- Uma loss direcional apresentada como novo alinhamento de representação:
  nossos controles e literatura de feedback parcial não sustentam essa novidade.
- Selecionar pelo erro de efeito: falhou no diagnóstico executado; VisLED,
  DataMIL e QoQ também impedem reivindicar seleção genérica como novidade.

## Sequência de execução recomendada

1. Reconciliar os 3.062 pares publicados e os 2.975 locais, preservar proveniência
   e alinhar máscaras por identidade. Os joins ambíguos não viram rótulos.
2. Auditar um lote dos 146 scene IDs coincidentes: conferir instante, instrução
   e referente antes de formar quartetos mundo × linguagem. Separar por cena,
   nunca por imagem/edição da mesma cena.
3. Medir concordância e tempo de anotação da validade e das restrições. Obter
   referências independentes; não inventar alvos normativos a partir da VLA.
4. Executar baselines simples e medir o custo completo. Comparar com IVA/ECT
   onde as tarefas e interfaces permitirem uma adaptação fiel, documentando-a.
5. Exigir efeito em comportamento e generalização por mecanismo/backbone.
   Inferência frontal com padding não demonstra desempenho no input nativo.

CARLA não é requisito para o inventário, curadoria ou diagnóstico. Uma alegação
de melhoria de direção em malha fechada exigirá um ambiente que a avalie;
escolher CARLA ou outro depende dessa alegação, não do nome do método.

## Material preservado e limites desta leitura

`artifacts/mi3_literature_20261020/` contém nove PDFs de fontes primárias, texto
extraído, URLs/hashes e os inventários. Foram consultadas seções de método,
avaliação e limitações dos trabalhos centrais; vários concorrentes foram
triados pelo resumo. Isso não equivale à revisão técnica integral de todos.
Nenhum novo treinamento ou resultado comportamental é reivindicado nesta rodada.
