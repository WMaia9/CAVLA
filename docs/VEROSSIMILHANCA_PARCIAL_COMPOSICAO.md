# Verossimilhança dos dois bits observados

Este é um baseline probabilístico de observação parcial, não uma nova fórmula
de identificação causal. A relação com regressão a partir de respostas de
grupos e estados individuais latentes é antiga; veja
[Zhang, Bilder e Tebbs, 2013](https://pmc.ncbi.nlm.nih.gov/articles/PMC3841014/)
e [Group Testing, segunda edição](https://arxiv.org/abs/1902.06002).

Para uma cena/curva, o modelo prevê probabilidades individuais `p_i`. A
independência condicional dos Bernoullis é uma hipótese de modelagem, não uma
propriedade demonstrada do tráfego. Seja `j` o agente excluído e `b0,b1` as
respostas original/após exclusão. Sob o operador OR e outros agentes congelados:

| Observação | Evento compatível | Probabilidade |
|---|---|---|
| 0,0 | Nenhum agente bloqueia | produto de `(1-p_i)` para todos os agentes |
| 1,0 | Só o agente j bloqueia | `p_j` vezes produto de `(1-p_i)` para i diferente de j |
| 1,1 | Algum outro agente bloqueia; j é desconhecido | `1 - produto_{i diferente de j}(1-p_i)` |
| 0,1 | Incompatível com o modelo exato | 0 |

As três probabilidades possíveis somam 1. No caso 1,1, a probabilidade não
depende de `p_j`: esse par não supervisiona seu estado individual naquela
consulta. No caso 1,0, ele identifica um bloqueador único. Compartilhar a rede
e a estrutura geométrica entre consultas acrescenta hipóteses e pode permitir
generalização; não torna a geometria física identificável por definição.

O treino usa a NLL do evento observado dividida por dois. O baseline BCE calcula
a média das duas perdas marginais. **A BCE marginal não é automaticamente
inválida:** trata-se de um objetivo composto que pode aprender marginais
corretas. A dependência entre observações, isoladamente, não prova viés nem
superioridade da NLL conjunta. A comparação empírica deve decidir sua utilidade
neste regime e com esta capacidade de modelo.

Noisy-OR é calculado no domínio log por eventos disjuntos de primeiro sucesso.
Isso evita zerar gradientes quando todos os logits individuais são muito
negativos. Os testes verificam valores contra enumeração de atribuições,
gradientes em logits extremos, ausência de gradiente para o agente desconhecido
no caso 1,1 e probabilidade zero para a observação impossível 0,1.

Com pares editados reais, a hipótese OR exata pode falhar: identidade de remoção,
preservação dos demais elementos e movimento reativo não estão garantidos.
Não se deve aplicar esta loss descartando silenciosamente observações 0,1.
O piloto atual usa exclusões algébricas com rótulos consistentes. Um modelo de
ruído/contaminação ou uma referência independente ainda precisa ser estudado
antes de transportar essa formulação aos pares físicos/visuais imperfeitos.
