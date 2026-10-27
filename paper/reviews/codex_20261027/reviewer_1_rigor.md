# Revisor 1 — rigor matemático, experimental e estatístico

**Parecer interno independente, não avaliação oficial do ICML. Data: 27/10/2026.** Versão avaliada: snapshot congelado `paper/reviews/codex_20261027/snapshot/main.tex` (SHA256 `47c930ee22a667cdeec6e60054fa0bfb12babf5aa621be3367f8d4366eaf7f97`) e PDF correspondente, 9 páginas. Não consultei os pareceres anteriores nem a auditoria anterior para formular esta revisão. Consultei os scripts e artefatos citados abaixo; não executei modelos nem usei GPU.

## Resumo e recomendação

O paper propõe controlar a escala de ruído inicial ao avaliar acelerações de políticas de flow matching: escolhe-se a escala que iguala a dispersão entre seeds do sampler candidato e de um solver de referência, e comparam-se ADE e taxas de mudança de decisões. Em três Alpamayos, Euler4/10 tende a contrair as trajetórias em direção ao endpoint de um passo e diversos efeitos agregados se aproximam da curva obtida por redução de ruído. Heun barato não segue essa curva; o Qwen-Drive mostra que casar dispersão não garante reproduzir o efeito sobre a qualidade. Um cache de velocidades fornece um exemplo aplicado.

**Recomendação sobre a versão congelada: ainda não recomendaria aceitação no ICML.** O problema é relevante e a investigação tem mérito, mas o resultado central ainda é exploratório, quase todo o conjunto principal está na partição de treino alinhada ao Alpamayo, uma proposição está incorreta como escrita, e a ponte entre o mecanismo ideal e os campos aprendidos não sustenta algumas frases gerais. A recomendação pode mudar com a confirmação fora de treino e um argumento mais preciso sobre o que o controle identifica. Uma contribuição de análise não precisa inventar outra loss; precisa demonstrar um insight que sobreviva a controles alternativos e tenha alcance definido.

## Pontos fortes

1. Pergunta operacional clara: ao reduzir custo numérico, preservou-se a distribuição relevante ou apenas se mudou a dispersão? A comparação pode corrigir uma atribuição equivocada de ganhos em avaliações de políticas.
2. Emparelhamento dos samplers no mesmo contexto e ruído; bootstrap por clipe e não por seed. O script `src/cavla/paper_numbers.py:95–111` usa reamostragem conjunta entre modelos e recalcula o matching da escala de ruído.
3. O texto distingue ausência de diferença detectável de equivalência e declara que as margens foram escolhidas após observar o conjunto exploratório (`snapshot/main.tex:154`). Também relata falhas da previsão no Qwen e não transforma colisões raras em evidência de segurança.
4. O controle de contração linear e os samplers Heun são tentativas sérias de refutar a interpretação. O resultado não é apresentado como uma garantia amostra a amostra.
5. O Qwen v2 corrige o domínio temporal da referência e verifica Heun32 contra Heun64. Não considero válidas, contra esta versão, críticas baseadas apenas no Qwen v1 ou na alegação literal de que redes aprendidas têm diversidade zero em um passo: o texto atual já reconhece o campo ideal.

## Fragilidades principais

### 1. A Proposição 1 é falsa para a classe de integradores enunciada — prioridade alta

**Local:** `snapshot/main.tex:141–146`. O enunciado quantifica sobre todo Runge–Kutta explícito e usa somente `h Lip(v)<1` para concluir que cada passo é um homeomorfismo que preserva semiespaços. O limite precisa depender dos coeficientes de RK. O argumento apresentado é justificável para Euler, mas não para a classe inteira.

Contraexemplo que calculei independentemente: considere o fluxo retificado exato com ruído e alvo ambos gaussianos padrão,

`v(x,t) = (2t−1)/((1−t)^2+t^2) · x`.

O campo é ímpar, a reflexão é `g(x)=−x`, o hiperplano é `{0}`, e o Lipschitz espacial global em `[0,1]` é 1. Tome `x=1`, `t=0,4`, `h=0,1`. Um RK explícito de três estágios com `a21=0,5`, `a31=1`, `a32=0` e pesos `b=(−4999,5; 10000; −4999,5)` é consistente de **ordem 2**, pois `sum(b)=1` e `sum(b*c)=1/2`. Ele retorna **−0,9232673267**, enquanto o fluxo exato retorna **+0,9805806757**, apesar de `hL=0,1<1`.

Os coeficientes são deliberadamente extremos: isso refuta a quantificação universal, não Euler/Heun/RK4 concretos do experimento. A correção mínima é restringir a proposição a Euler, explicitar hipóteses do campo ótimo e demonstrar a preservação por distância ao hiperplano. A frase “decision flips therefore need an asymmetric context” também deve ser limitada ao domínio em que o resultado realmente se aplica. Uma rede aprendida pode quebrar a simetria mesmo que a distribuição-alvo a possua.

### 2. O conjunto central ainda não estabelece generalização ou a equivalência anunciada — prioridade alta

**Locais:** `snapshot/main.tex:154,159,184`; `snapshot/tables/tab_matched.tex`; `artifacts/split_overlap_v1_20261027/summary.json`.

O próprio paper informa que **57/60 clipes** estão na partição train de `alpamayo_aligned`; o artefato confirma 57 train e 3 test. Isso não invalida uma comparação mecanística dentro do modelo, mas limita fortemente a narrativa de desempenho sobre logs de teste e a generalidade do resultado. A nova avaliação de 120 clipes fora dessa partição é o teste decisivo; não deve ser contada como concluída no parecer sobre o snapshot.

Nas seis células centrais de Euler4/10, apenas R1/Euler10 passa simultaneamente os critérios de equivalência de ADE e flips. A tabela contém, por exemplo, resíduo de flips do A2/Euler10 de **+2,0 pt [0,03; 4,4]**, e A1.5/Euler4 de **+1,5 pt [−1,3; 5,3]**. O corpo reconhece isso, mas expressões como “reproduces” no abstract, “land on the curve” nas legendas e “accounts for most of what they change” ainda sugerem uma conclusão mais uniforme do que os dados suportam.

Correção mínima: usar o conjunto fora de treino como principal assim que estiver completo, manter células/modelos separados e dizer “aproxima os efeitos médios no conjunto exploratório” enquanto não se demonstrar equivalência. Não substituir equivalência por um intervalo que simplesmente contém zero.

### 3. O controle identifica uma família de efeitos agregados, não um mecanismo de temperatura único — prioridade alta

**Locais:** `snapshot/main.tex:151–154,189,211–213`; `src/cavla/paper_numbers.py:68–85`.

O tau é escolhido para igualar um único escalar: distância média entre pares de trajetórias. O resíduo de flips compara duas **taxas de desacordo contra uma terceira referência**; duas políticas podem ter a mesma taxa e trocar decisões completamente diferentes, em clipes diferentes. A aproximação de taxas de flips e ADE, portanto, não equivale a uma aproximação das distribuições de ação ou de decisão.

O paper reconhece essa limitação e o controle de contração linear também passa parte do teste. Isso é uma força da investigação, mas reduz o que se pode chamar de explicação mecanística específica. A aproximação da mediana de contração por uma fórmula gaussiana também não identifica a distribuição: o texto admite que ela não prevê o ranking de clipes. O termo “temperature” aqui deve significar explicitamente escala do ruído inicial, e não uma afirmação geral de densidade final proporcional a `p(x)^(1/T)`.

Para tornar a contribuição forte: calibrar tau em dados separados, avaliar em clipes novos e acrescentar um endpoint capaz de separar as alternativas que já passam no conjunto atual — por exemplo, frequências condicionais de decisões, distância entre distribuições ou um teste de assinatura previamente congelado. O contraste entre Euler e Heun é útil, mas a ausência de contração no Heun não demonstra que temperatura seja a única explicação do Euler.

### 4. Teoria gaussiana não se transfere automaticamente à métrica usada — prioridade média/alta

**Locais:** `snapshot/main.tex:118–125,130–137,180`; `src/cavla/paper_numbers.py:25–33`.

A identidade de um passo foi corretamente qualificada para campo exato. Entretanto, o ponto de referência das medidas empíricas é `E1(z)` **dependente da seed**, e o modelo decodifica ações de aceleração/curvatura por uma transformação não linear para trajetórias. A expressão exata para contração em coordenadas de uma gaussiana no espaço de ações não prova contração radial, aproximação da média condicional ou equivalência em espaço de trajetórias. O artigo pode usar isso como analogia explicativa, mas deve evitar chamar todo endpoint E1 aprendido de média condicional real.

A frase final da linha125 também precisa de restrição: coordenadas com `s>3,3` podem carregar justamente a maior parte da variância; estar fora do intervalo não implica “carry little of the spread”. Além disso, a notação da equação usa `Aref` definido como Heun32, quando a identidade teórica exige o fluxo **exato**. Distinguir `F` exato de referência numérica.

Correção mínima: separar formalmente o resultado ideal, a aproximação no espaço de ações e a observação empírica em trajetória; reportar erro da predição ou ganho contra uma constante, não apenas proximidade da mediana. A mediana ajustada/semelhante não é uma lei preditiva fora da amostra.

### 5. Referência, calibração e bootstrap ainda precisam ser fechados — prioridade média

**Locais:** `snapshot/main.tex:110,154`; `src/cavla/paper_numbers.py:39–49,58,95–122`.

Fiz uma checagem parcial de convergência usando os artefatos existentes: nos 60 clipes e seeds9201/9202 em comum, Heun32 do probe de temperatura contra Heun160 do probe anterior tem ADE entre referências média **0,02142 m**, mediana **0,00891 m**, máximo **0,17803 m** e **0/120** mudanças nas decisões por thresholds atuais. Como controle de compatibilidade, Euler10 entre esses dois pipelines difere em média **0,00145 m** (máximo **0,01098 m**) e também 0/120 decisões. Isso favorece estabilidade do resultado de decisão em tau=1, mas não é prova de convergência uniforme nem cobre os demais taus/modelos; o erro médio de referência já é uma fração material da margem de ADE de 0,05 m.

Outros detalhes:

- A escala do controle linear é ajustada uma vez sobre os 60 clipes (`add_shrinkage`) e não reajustada dentro de cada bootstrap, ao contrário do tau. Os intervalos comparativos tratam esse nuisance como fixo. Se o alvo é uma família calibrada no conjunto, reajustar no bootstrap ou calibrar separadamente.
- A interpolação linear de métricas de decisão entre taus não é a execução do solver no tau estimado. Essas taxas podem ter degraus por thresholds e seis seeds. Validar a interpolação em um tau intermediário, ou contabilizar sua incerteza.
- O texto diz 1000 reamostragens; o script tem default2000. O manifest/summary deve determinar o número efetivo. Não é problema científico central, mas é uma inconsistência de reprodutibilidade.
- A figura/teoria gaussianas usam amostra separada e espaço de ações; isso é declarado. Não misturar novamente esses resultados com as contagens de contração da amostra principal em trajetória.

## Perguntas para os autores

1. Qual é exatamente a alegação primária: igualdade de ADE/flip-rate médios, proximidade de distribuições condicionais, ou uma explicação do mecanismo? Que resultado distinguiria temperatura de outras transformações com a mesma dispersão?
2. A escala de ruído será estimada numa população e transferida para outra ou reajustada na própria avaliação? Os dois protocolos respondem perguntas diferentes.
3. Heun32 converge nos contextos difíceis e taus reduzidos dos três Alpamayos? O ponto de corte da decisão permanece estável com referência mais precisa?
4. Por que o pooling das gerações é o endpoint adequado quando a heterogeneidade entre elas é parte do resultado? Favor reportar modelos individualmente mesmo que o teste primário seja agrupado.
5. Como a mudança de dispersão se manifesta em ações executadas e no seletor de trajetórias? Se não houver efeito fechado, qual consequência metodológica quantitativa permanece demonstrada?

## Mudanças mínimas que alterariam minha recomendação

1. Corrigir ou remover a proposição universal e separar campo ideal, rede aprendida e decoder.
2. Concluir confirmação em clipes realmente fora da partição de treino, com critérios e margens congelados; apresentar também falhas, heterogeneidade e efeito por modelo.
3. Adotar uma alegação identificável e avaliá-la com pelo menos uma medida que o matching de dispersão não force e que o controle linear não reproduza trivialmente. Se isso não acontecer, limitar a conclusão a um diagnóstico de atribuição de ganhos em métricas médias.
4. Demonstrar convergência suficiente da referência para a precisão das alegações e dar o protocolo completo no apêndice (hoje ainda TODO, linha246).
5. Fechar o exemplo FlashDrive com decisões e, se o paper alegar consequências comportamentais, resultados fechados com incerteza adequada. Uma melhoria clara do protocolo de avaliação pode sustentar um paper de análise; não é necessário que todos os efeitos melhorem segurança.

## Atualização de contexto posterior ao snapshot

Durante esta revisão o coordenador informou que a avaliação A1.5 em 120 clipes fora de treino terminou e uma análise CPU encontrou equivalência de E10 em ADE e flips. Esse resultado é promissor e fortalece a linha; não o tratei como evidência já contida no PDF congelado, nem o validei integralmente neste parecer. A conclusão final deve incorporar o artefato dessa nova rodada e a revisão do desenho pelo auditor, sem reescrever retrospectivamente o que o snapshot demonstrava.

## Nota interna simulada — solicitada após o parecer

Esta nota avalia **a mesma versão congelada acima**, sem releitura de versão nova nem novos experimentos. É uma simulação interna para orientar o trabalho, não um formulário oficial, uma previsão de aceitação ou uma avaliação emitida pelo ICML.

Escala utilizada: 1 = rejeição forte; 3 = rejeição; 4 = rejeição leve (*weak reject*); 5 = limítrofe; 6 = aceitação leve; 8 = aceitação forte; 10 = excepcional. A confiança usa escala separada de 1 a 5.

| Dimensão | Nota |
|---|---:|
| **Geral** | **4/10** |
| **Recomendação atual** | **Rejeição leve / weak reject** |
| **Confiança** | **4/5** |
| Rigor matemático e estatístico | 4/10 |
| Evidência experimental | 4/10 |
| Contribuição científica demonstrada | 4/10 |
| Clareza e transparência | 6/10 |

Portanto, **sim: minha recomendação simulada para submissão desta versão é reject**, na categoria leve. Isso não é uma recomendação de abandonar a pesquisa. O projeto tem uma pergunta relevante, implementação funcional e tentativas honestas de refutação; o manuscrito ainda não fecha a contribuição que anuncia.

Os três motivos de maior peso são:

1. **Uma afirmação formal está errada como enunciada.** A condição apresentada não cobre todo RK explícito. A correção parece localizada, mas precisa entrar no paper; não posso pontuar como demonstrada uma proposição refutada por contraexemplo.
2. **A demonstração empírica principal ainda é exploratória.** O conjunto congelado é pequeno, 57/60 clipes pertencem à partição train alinhada, e apenas uma das seis células centrais passa ambos os critérios de equivalência. O texto é relativamente transparente sobre isso, mas ainda usa linguagem conclusiva em pontos importantes.
3. **O mecanismo específico não está identificado.** Dispersão pareada e taxas agregadas também podem ser reproduzidas por contração linear. A contribuição precisa dizer exatamente que inferência válida esse protocolo permite e demonstrar uma consequência que sobreviva a esse controle; não basta trocar o nome de contração por temperatura.

### O que me levaria a 6/10

Eu consideraria **aceitação leve**, sem prometer uma mudança automática de nota, se uma versão revisada entregasse conjuntamente:

- Teoria correta e com escopo explícito: Euler ou condições específicas por método; distinção entre campo ideal, rede aprendida e trajetória decodificada.
- Confirmação fora de treino, com margens congeladas, estatística por clipe e resultados separados por modelo. Para a alegação de replicação nas gerações, o critério registrado de equivalência de E10 em pelo menos dois dos três modelos seria evidência relevante, acompanhado de análise honesta do modelo que eventualmente falhar.
- Uma contribuição delimitada que os controles não esvaziem: uma previsão mecanística transferível, uma diferença distribucional relevante previamente definida, **ou** demonstração convincente de que um controle de ruído muda a conclusão de uma comparação prática de aceleração. Para um paper de análise, não exijo uma nova loss nem melhora universal de segurança.
- Referência numérica validada na precisão das margens e alegações do artigo; fechamento dos protocolos e correções de bootstrap/interpolação que afetam os intervalos reportados.

### O heldout A1.5 posterior muda a nota?

**Não altera a nota 4/10 nem a recomendação sobre o snapshot congelado.** Fortalece a justificativa de continuar: é evidência positiva justamente no ponto de generalização que faltava. Para atribuir uma nova nota ao paper, seria necessário incorporar e verificar o resultado, observar o restante da confirmação prevista e avaliar as correções da teoria e da contribuição. Um único resultado posterior favorável não resolve, sozinho, os três motivos acima.
