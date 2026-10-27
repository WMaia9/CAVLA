# Parecer independente 2 — novidade, significado e posicionamento

**Data:** 27/10/2026. **Natureza:** revisão interna simulada; não é uma avaliação oficial do ICML e não estima probabilidade de aceite.

**Versão avaliada:** cópia congelada `snapshot/main.tex` e `snapshot/main.pdf`, SHA256 do TeX `47c930ee22a667cdeec6e60054fa0bfb12babf5aa621be3367f8d4366eaf7f97`. Não consultei pareceres anteriores, o parecer do outro revisor nem a auditoria anterior. Li os artefatos indicados abaixo e fontes primárias. Uma nota posterior separa um resultado concluído depois do snapshot. Não usei GPU nem alterei paper, código ou jobs.

## Resumo e recomendação

O paper investiga quanto do efeito de samplers Euler grosseiros sobre métricas de políticas de direção é reproduzido ao reduzir a escala do ruído inicial de um solver de referência. Propõe comparar métodos após parear sua dispersão entre seeds e apresenta Alpamayo, um controle Heun, contração linear e Qwen-Drive, além de uma reprodução do cache de velocidades do FlashDrive.

**Minha recomendação para a versão congelada é weak reject, com interesse real na continuação.** A pergunta é relevante e um paper de análise desse tipo pode ser adequado ao ICML sem inventar outra loss ou arquitetura. O impedimento principal é a distância entre uma observação útil de contração/risco de avaliação e uma contribuição nova suficientemente identificada. O problema geral de temperatura numérica confundindo qualidade/diversidade já tem um antecedente próximo não citado; a evidência presente não identifica temperatura de maneira única; e os resultados mais promissores de confirmação ainda não integram o paper congelado. Não vejo motivo para abandonar a candidata, mas tampouco trataria um ganho adicional de uma tabela como suficiente para resolver esses pontos.

## Forças

1. **Uma pergunta concreta e útil.** Separar distribuição aprendida de distribuição implementada pelo sampler é relevante para atribuir melhorias de trajetória. Os testes compartilham observação, raciocínio e ruído dentro de um clipe, o que é um desenho forte para estudar alterações do solver (`main.tex:108`, `:110`, `:158`).
2. **O controle negativo Heun discrimina.** Não é uma comparação em que qualquer sampler produz a conclusão desejada. Heun3/5 têm resíduo de inversões grande apesar de dispersão pouco alterada (`main.tex:186`, `tables/tab_matched.tex`).
3. **O controle de contração simples é honesto e valioso.** O próprio paper reconhece que mover linearmente amostras em direção a E1 reproduz as taxas, sem um novo integrador (`main.tex:189`, `:212`). Isso reduz o risco de apresentar uma correlação como identificação de mecanismo.
4. **A versão atual explicita limitações difíceis.** Somente R1/E10 passa os dois testes principais no conjunto exploratório; 57/60 clipes estão no treino alinhado do Alpamayo; resultados do Qwen não satisfazem todas as previsões; resultados raros de parada/colisão não são promovidos (`main.tex:70`, `:154`, `:159`, `:184`, `:196`, `:213`). Essas qualificações tornam a investigação confiável.
5. **Reprodução de um componente publicado.** O cache do FlashDrive oferece uma oportunidade de consequência externa concreta, em vez de apenas um solver artificial (`main.tex:191`). A separação explícita entre implementação própria do componente no checkpoint NVIDIA e stack otimizado ainda em execução é correta.

## Fragilidades prioritárias

### P1 — A novidade genérica de “erro numérico esconde temperatura e engana métricas” já foi antecipada

Falta o trabalho de Zheng et al., **Masked Diffusion Models are Secretly Time-Agnostic Masked Models and Exploit Inaccurate Categorical Sampling**, ICLR 2025, versão arXiv v6: https://arxiv.org/abs/2409.02908 . O projeto mantém também um título anterior: https://research.nvidia.com/labs/cosmos-lab/mdm-equal-mm/ .

Esse trabalho demonstra erro de precisão na amostragem categórica que reduz temperatura efetiva e diversidade, favorece uma métrica incompleta de qualidade e altera comparações; sua seção 6.2 compara a curva qualidade/entropia com temperatura explícita. Portanto, a ideia geral de diagnosticar um ganho numérico como mudança escondida de temperatura e exigir controle de diversidade não é nova. A distinção possível é real: aqui se trata de discretização Euler de um fluxo contínuo, ação/decisões e um cache VLA, não truncamento de Gumbel em texto. O paper precisa desenvolver e demonstrar essa diferença, e não apenas transportar a recomendação de avaliação.

Também está ausente **Temporal Score Rescaling for Temperature Sampling in Diffusion and Flow Models**, Xu et al., ICML 2026: https://arxiv.org/abs/2510.01184 ; texto: https://arxiv.org/html/2510.01184v2 . É especialmente relevante para separar controle de diversidade de temperatura probabilística; inclui políticas de manipulação. Não demonstra a mesma hipótese de discretização do Euler, logo não elimina automaticamente a contribuição. Mas omiti-lo deixa o posicionamento incompleto.

Outro confronto conceitual útil é **Diversify Diffusion with Temperature Sampling and Variance-Corrective Time Shifting**, Li et al., 2026: https://arxiv.org/abs/2607.10853 . Distingue alterações de pesos entre modos e variâncias locais. Não é um estudo do nosso solver; ajuda a evitar que “temperatura” seja usado como sinônimo de qualquer redução da dispersão.

### P2 — O teste principal detecta contração compatível com escala de ruído; não identifica temperatura de forma única

A definição ajusta tau para igualar a dispersão média (`main.tex:152`). A tabela mostra que o controle Shrink também obtém taxas próximas. No Alpamayo1.5/E4, por exemplo, Euler e Shrink têm ADE1 2,58 e 2,53 m; ambos têm pequena diferença agregada na taxa de inversões contra a curva de ruído. Isso não permite atribuir exclusivamente à temperatura o mecanismo gerador da melhoria.

O paper reconhece o problema (`main.tex:189`, `:211–212`), mas continua usando “compare distributions” (`:154`) para testes de poucos funcionais agregados. Duas distribuições podem compartilhar dispersão, ADE e taxa de desacordo com uma terceira referência, e ainda diferir muito em probabilidades condicionais e caudas. A proximidade de uma mediana de contração também não resolve essa não identificação.

Não exijo igualdade amostra por amostra. Exijo escolher a afirmação: **controle operacional de dispersão explica certas métricas médias** é defensável; **Euler equivale distribucionalmente a temperatura menor** requer evidência muito mais forte. A recomendação atual de relatar um controle de dispersão pode sobreviver mesmo que a hipótese distribucional falhe.

### P3 — A teoria explica uma tendência conhecida; ainda não entrega previsão nova dos limites observados

A base gaussiana é atribuída corretamente a Claustre et al., **Gaussian Flow-Matching Schedules**, https://arxiv.org/abs/2609.25839 . O fato de o valor típico aproximar a mediana não torna o fator um preditor novo de contextos: o texto admite que não ordena os clipes (`main.tex:180`).

Há uma oportunidade mais substantiva que não está desenvolvida: o próprio fator da equação 1 depende da direção d e do desvio s_d. Assim, para uma Gaussiana anisotrópica, Euler contrai cada direção de forma diferente, enquanto multiplicar o ruído inicial por um único tau contrai todas igualmente no fluxo exato linear. A teoria prevê que uma equivalência escalar geral falha. Em vez de usar a Gaussiana só para explicar 0,83, seria útil prever **quando a aproximação escalar funciona e qual resíduo deve permanecer**. Uma lei quantitativa validada fora da amostra diferenciaria o paper de uma aplicação de conhecimento clássico.

O resultado sobre simetria (`main.tex:144–146`) também está incompleto e não é conectado a uma classe de decisões reais que satisfaça suas hipóteses. Não o contaria, por enquanto, como contribuição teórica consolidada. Deixo a validade exata da proposição para a revisão técnica; sua presença isolada não estabelece novidade significativa.

### P4 — O ganho atribuído a uma aceleração publicada precisa ser decomposto no sistema certo

O FlashDrive combina streaming, ajuste de pesos, especulação de linguagem, quantização e cache. Seu artigo atribui a melhora de erro a mais de uma possível causa; o componente de cache isolado é só uma parte. Fonte primária: https://arxiv.org/html/2608.12932v1 , §§3.3 e 4.3.

A versão congelada faz corretamente uma afirmação restrita sobre cache próprio no checkpoint original (`main.tex:191`), mas ainda anuncia “a published acceleration behaves like a lower temperature”. O efeito em ADE1 é interessante: resíduo 0,008 m, IC90% [-0,034; 0,051], tau≈0,64. Ainda não passa a margem de equivalência, faltam decisões nesse experimento e o checkpoint otimizado não está analisado. O resultado atual não reinterpreta todo o ganho do FlashDrive.

Também convém separar **atribuição do ganho em uma métrica** de **mérito de engenharia do sampler**. Um sampler pode obter uma boa fronteira custo/qualidade justamente por realizar uma contração implícita; um controle com 64 avaliações não retira seu ganho de latência. A frase de introdução de que a melhora não deve ser creditada ao sampler (`main.tex:77`) é forte demais. A inferência suportada é que melhorar ADE1 não demonstra preservar a distribuição ou melhorar a aproximação do fluxo.

### P5 — O escopo experimental positivo ainda é estreito para a força da recomendação

No snapshot há 60 clipes compartilhados entre três membros da mesma família, com 57 clipes no treino alinhado. Isso não invalida o contraste intramodelo, mas limita generalidade. O Qwen v2 está mais bem controlado que uma integração ingênua até t=1: o paper usa o domínio temporal do modelo, endpoint final, precisão do expert e teste Heun32/64 (`main.tex:213`). Esse é um bom controle, mas não oferece uma segunda replicação positiva do vínculo entre menos passos, resfriamento e ganho de ADE.

As figuras e a introdução fazem a melhora de ADE1 acompanhada de piora de minADE6 parecer uma consequência geral. Tal troca é plausível, porém depende da relação entre a política e o humano registrado: reduzir dispersão pode melhorar ambas as métricas ou nenhuma. O texto deve distinguir previsão do mecanismo de regularidade observada na amostra.

O apêndice ainda deixa em TODO o protocolo, limiares de decisão, prova e resultados secundários (`main.tex:235–248`). Isso impede uma avaliação científica completa, além da apresentação. Não trataria ausência de closed-loop como rejeição automática de um paper de análise; exigiria, porém, alguma consequência relevante e robusta do viés de avaliação ou um resultado teórico preditivo que dispense essa promessa.

## Fronteira da novidade

| Elemento | Situação frente ao estado da arte |
|---|---|
| Ajustar diversidade pelo ruído inicial | Conhecido; não é contribuição isolada. |
| Euler em fluxo gaussiano contrai | Base já estudada; o próprio paper a credita. |
| Erro numérico produz temperatura implícita e qualidade enganosa | Antecedente conceitual claro em Zheng et al., ICLR 2025. |
| Menos passos perde cobertura de modos | Relacionado a GMFlow/SubFlow; esses trabalhos já constam na bibliografia. |
| Lei quantitativa de Euler→ruído efetivo e suas falhas em ações VLA | Potencialmente nova, mas ainda não estabelecida além de métricas agregadas. |
| Demonstração de que um ganho publicado de ação desaparece ao controlar dispersão | Pode ser contribuição forte de análise, desde que reproduzida no componente/sistema correto e em dados adequados. |
| Protocolo de controle de dispersão com incerteza e equivalência | Útil, mas precisa mostrar utilidade adicional ao precedente de matching por entropia e aos controles elementares. |

Não encontrei nas fontes primárias consultadas uma antecipação exata do resultado Alpamayo/FlashDrive. Isso não é prova de novidade. O espaço defensável é a **causa numérica contínua, as condições de validade e a consequência demonstrável para avaliar ações**, não a noção genérica de temperatura escondida.

## Perguntas aos autores

1. Qual quantidade não usada para ajustar tau distinguirá ruído reescalado de contração genérica? Ela será predita antes do teste?
2. O tau será calibrado em um conjunto e transferido para outro, ou sempre ajustado no próprio conjunto avaliado? Ambos podem ser úteis, mas respondem a perguntas diferentes.
3. Qual resultado muda a conclusão de alguém que hoje avaliaria o FlashDrive pelos números publicados? O que permanece após separar pesos, cache, quantização e custo?
4. A teoria consegue prever em quais modelos/cenas tau escalar falha, incluindo Qwen e resíduos em decisões raras?
5. Qual recomendação prática mantém a comparação de custo/qualidade justa sem confundir fidelidade ao fluxo com acerto da tarefa?

## Evidência que mudaria minha avaliação

Eu não pediria outra arquitetura. Pediria um conjunto enxuto e convincente:

1. Posicionamento explícito frente a Zheng2025 e TSR, com terminologia operacional de tau e escopo da novidade.
2. Uma confirmação em dados novos/alinhados, com margens congeladas, negativos, teste de caudas e alguma medida não usada no ajuste de dispersão. Clusters e origem dos dados precisam estar claros.
3. Resultado completo do cache/checkpoint oficial, com uma conclusão que sobreviva a controles elementares e limite corretamente a atribuição ao sistema publicado.
4. Uma previsão nova de mecanismo ou limite, por exemplo um resíduo relacionado à anisotropia, que se sustente fora da amostra. Alternativamente, consequência de avaliação/execução suficientemente importante e replicada que torne essa teoria adicional dispensável.
5. Apêndice, prova e protocolos completos; simplificação das frases que transformam concordância de taxas em igualdade de distribuição.

O pacote acima poderia transformar a candidata em um bom paper de análise. Mais modelos ou um p-valor menor, sem resolver a contribuição, não bastariam.

## Nota posterior ao snapshot: A1.5 held-out concluído

Após o parecer inicial, o coordenador apontou `artifacts/audit_agent_20261027/heldout_a15_summary.json`; conferi esse arquivo. Nos 120 clipes, E10 tem tau≈0,849; resíduo ADE1 +0,0075 m, IC90% [-0,0022; +0,0176]; resíduo de inversões -0,38 ponto, IC90% [-1,45; +0,75]. Ambos passam as margens declaradas. **Isso fortalece materialmente a sustentação empírica do resultado E10**, e a futura versão deve incorporá-lo. E4 ainda não estabelece equivalência nas mesmas margens.

Essa confirmação não resolve sozinha identificação ou novidade. Também há uma nuance útil para a apresentação: nesse conjunto, minADE6 do E10 é 0,853 m contra 0,872 da referência, efeito pontual -0,019 m. Não concluo melhora significativa; apenas noto que a piora universal de best-of-six não decorre da hipótese. O status final do paper deve ser reavaliado com o resultado conjunto das confirmações planejadas, sem alterar endpoints para selecionar apenas o E10 positivo.

## Nota interna solicitada após o parecer

Esta nota se refere à **mesma versão congelada** revisada acima. Não reli uma nova versão nem executei novos experimentos para atribuí-la. A escala é uma convenção interna solicitada pelo coordenador, **não um formulário oficial do ICML**: 1 = rejeição forte; 3 = rejeição; 4 = rejeição leve (weak reject); 5 = limítrofe; 6 = aceitação leve; 8 = aceitação forte; 10 = excepcional. Valores intermediários têm interpretação gradual.

**Nota geral: 4/10. Recomendação: rejeição leve (weak reject). Confiança: 4/5.**

| Critério | Nota interna | Justificativa |
|---|---:|---|
| Novidade | 4/10 | A relação genérica entre erro numérico, temperatura implícita e comparação enviesada já tem precedente direto. A diferença específica no fluxo de ações ainda precisa tornar-se uma contribuição demonstrada. |
| Significado potencial | 6/10 | Uma explicação convincente de ganhos aparentes em acelerações VLA pode mudar a prática de avaliação. O problema tem importância mesmo sem outra arquitetura. |
| Evidência da versão congelada | 4/10 | Controles úteis e resultados coerentes, mas maioria das células sem equivalência estabelecida, conjunto exploratório majoritariamente no treino alinhado e teste que não identifica exclusivamente temperatura. |
| Clareza | 6/10 | A estrutura e as qualificações permitem entender a tese; ainda há tensão entre o título/linguagem de mecanismo e a conclusão limitada a taxas, além de apêndices incompletos. |

As subnotas não são uma média aritmética que determina a nota geral. Novidade e sustentação da contribuição central são decisivas.

**Três motivos principais para o 4:**

1. **A fronteira da novidade não está fechada.** A hipótese específica pode ser nova, mas o argumento geral de avaliação já aparece em Zheng et al.; TSR e outras definições de temperatura precisam ser enfrentados.
2. **O diagnóstico não distingue o mecanismo de controles simples.** O Shrink também reproduz taxas. Mostrar que duas ou três métricas médias coincidem não basta para identificar uma equivalência distribucional ou um mecanismo exclusivo de temperatura.
3. **A consequência externa ainda é limitada.** O cache no checkpoint NVIDIA é uma boa prova de conceito, mas não estabelece a reinterpretação do sistema completo nem uma consequência robusta sobre execução; a confirmação e os detalhes reprodutíveis ainda não estão incorporados ao texto congelado.

**O resultado held-out A1.5 muda a nota?** Neste julgamento, **não altera a nota geral de 4 nem a recomendação de rejeição leve**. Ele melhora minha avaliação da evidência de E10 e torna mais justificado continuar a investigação. Se fosse atualizado isoladamente o critério de evidência com essa nota posterior, eu o elevaria de 4 para 5/10. O obstáculo principal permanece a contribuição identificada e diferenciada, e não apenas o intervalo de confiança de E10. Isso não é uma previsão de rejeição da futura submissão.

**Condições concretas para eu considerar 6/10 (aceitação leve):**

1. Delimitar e sustentar uma afirmação nova frente a Zheng2025/TSR: por exemplo, uma previsão quantitativa de quando a discretização contínua é aproximada por escala de ruído e quando deixa um resíduo estruturado. Uma consequência de avaliação suficientemente forte e replicada também pode cumprir esse papel; não exijo ambos nem um método novo.
2. Incorporar a confirmação planejada, com endpoints e margens congelados, referências verificadas, agrupamento adequado dos dados e resultado de algum funcional não usado para ajustar a dispersão. Relatar E4 e modelos/células negativos junto de E10 positivo. Um teste que diferencie a hipótese de Shrink precisa existir ou a contribuição deve ser explicitamente a identificação de contração genérica.
3. Concluir ao menos uma demonstração externa convincente: efeito do cache publicado na comparação correta, separando o checkpoint dos outros componentes, ou consequência de execução relevante e bem controlada. A conclusão deve sobreviver aos controles simples e não confundir ADE melhor com maior fidelidade/safety.
4. Corrigir a teoria que sustenta a tese, completar protocolos/apêndice e alinhar título, abstract e recomendações à evidência final. Esta é uma condição de correção e reprodutibilidade, não uma demanda por mais GPU.

Cumprir essas condições exige reavaliação; não garante automaticamente nota 6 ou aceite por revisores externos. O diagnóstico atual é **paper promissor, ainda abaixo da régua**, não ideia descartada.
