# Aquisição contrafactual: protocolo exploratório v1

Registrado antes de executar esta rodada. Não é uma alegação de novidade nem um
protocolo confirmatório de submissão. O teste separa ganho por aquisição de ganho
por mais treino e contabiliza o custo de observar a política.

## Hipótese

Uma regra que escolhe entre repetir uma comparação ruidosa e solicitar um novo
quarteto de referência pode produzir uma política melhor que seleção aleatória,
diversidade e incerteza com amostragem fixa, sob orçamento marginal equivalente.

## Cenário e referência

Usamos o problema visual de duas faixas já implementado. A referência é a função
determinística do estado controlado. O modelo inicial aprende apenas as entradas
factuais (duas instruções); um pool independente oferece quartetos factual/editado
para aquisição. Os rótulos do pool ficam ocultos aos seletores. Todos usam BC
para a adaptação; nenhuma loss causal adicional é introduzida nesta rodada.

Política: mesma CNN pequena com IDs de duas instruções, sem VLA pré-treinada.
Três modelos treinados por bootstrap fornecem desacordo epistêmico aproximado.
A resposta de cada modelo é tornada estocástica por ruído gaussiano **sintético**
aditivo na diferença de ação. Isso é um stress test controlado, não uma reprodução
da distribuição nativa do Alpamayo. Há três regimes pré-definidos: sem ruído,
ruído constante e ruído heterogêneo independente dos rótulos e dos estados.

## Desenho fixado

- Cinco replicações com seeds 101, 202, 303, 404 e 505. Cada uma muda dados,
  inicialização, bootstrap e amostras de aquisição, não só o minibatch.
- 224 cenas factuais iniciais; 96 cenas de validação factual; 96 candidatos no
  pool. O pool contém ID, maior severidade e fundo invertido em partes iguais.
- Três testes independentes de 256 cenas cada: ID, severidade e fundo invertido.
  Nenhum resultado de teste participa da escolha de exemplos ou checkpoints.
- Pré-treino: 1.500 passos por membro. Adaptação: 800 passos, mesmo modelo e
  otimizador por estratégia, com replay 50/50 de factual e adquirido.
- Checkpoint: menor MSE de ação na validação factual, avaliada a cada 100 passos.
- Orçamentos marginais: 1.024 e 2.048 créditos; referência exata de um quarteto
  custa 32 créditos; uma amostra da resposta de um membro custa um crédito.
  Uma resposta envolve as duas instruções e dois mundos. A razão de custos é
  simulada e não está calibrada para tempo humano/GPU real.
- Custo inicial comum (treino, rótulos factuais e validação) é idêntico e reportado
  separadamente. Incerteza não recebe gratuitamente inferências sobre todo o pool.
- Aleatória não paga por sondas que não usa e pode, portanto, adquirir mais
  rótulos. Diversidade paga uma extração de features por candidato. Inferências
  determinísticas pré-calculadas serão usadas em replay, cobradas no ledger
  quando acessadas logicamente; créditos não são chamadas reais ao Alpamayo.

## Comparações

1. Aleatória, sem sondas da política.
2. Diversidade no espaço latente inicial, via farthest-first e distância aos
   exemplos factuais já conhecidos.
3. Incerteza: desacordo de respostas dos membros, duas amostras por membro/cena.
4. Incerteza corrigida: mesmo desenho, desconta a contribuição estimada do ruído
   dentro de cada membro à variância das médias.
5. Magnitude da resposta média: controla se basta escolher efeitos grandes.
6. Regra adaptativa corrigida: pode repetir a amostragem de candidatos com
   scores imprecisos ou adquirir novo quarteto. O critério de imprecisão é uma
   aproximação heurística por variância, sem garantia de cobertura estatística.

Também continuamos o treino sem rótulos novos para cada replicação. A regra
adaptativa limita repetições extras a 25% do orçamento restante após as sondas
iniciais e nunca consome o crédito reservado ao próximo rótulo. O seletor não
recebe a intensidade de ruído verdadeira nem os erros contra a referência.

## Métrica e decisão

Primária: média igualmente ponderada do MSE de ação nos três testes, após
adaptação. Reportar cada teste, cada seed e o número de rótulos/sondas também.
ICs por bootstrap das cinco diferenças pareadas são apenas exploratórios e
têm baixa resolução; não substituem novas replicações de confirmação.

Gate gerencial para continuar: ganho em todas as cinco replicações contra
aleatória e incerteza corrigida fixa no regime heterogêneo, com pelo menos 5%
de redução média na métrica primária em ao menos um orçamento; verificar que
isso não se explica por degradação severa em um dos testes. Este limiar não é
previsão de resultado nem critério imposto pelo ICML.

Se a regra adaptativa não vencer os controles mais simples, registrar como
resultado negativo e não trocar retrospectivamente a métrica principal. Somente
um sinal promissor justificaria varrer razões de custo, usar novos cenários e
passar para uma VLA real. Toda atualização do protocolo deve preservar esta versão.
