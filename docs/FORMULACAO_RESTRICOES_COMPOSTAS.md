# Formulação operacional: restrições compostas e informação parcial

Esta nota fixa o problema e seus limites; não apresenta um teorema original.

Atualização: a [correção de duplicatas](ADENDO_DUPLICATAS_COMPOSICAO.md) exige
distinguir 584 ROIs de 578 instâncias únicas. O
[piloto aprendido e o holdout](RESULTADOS_COMPOSICAO_PARCIAL_E_HOLDOUT.md)
foram executados; uma regra estacionária sem rede superou os modelos aprendidos.

Seja `r_i(q)` o predicado de proximidade de uma curva `q` ao agente `i`, seguindo
seu movimento registrado. Para o subconjunto de agentes anotados `S`, a resposta
composta após excluir o conjunto `A` é

`B_A(q) = OR_{i em S, i fora de A} r_i(q)`.

Isso considera exclusão de restrições no modelo geométrico, mantendo todo o
restante congelado. Não representa reação de outros agentes nem segurança
completa. Mesmo `B_vazio=0` não certifica uma curva segura na cena real.

## Por que a remoção isolada não basta

Com três agentes e uma curva, os estados `r=(1,1,0)` e `r=(0,1,1)` dão a mesma
resposta original e as mesmas respostas após cada remoção isolada: todas são 1.
Mas retirar os agentes 1 e 2 libera a curva no primeiro estado e não no segundo.
Com dois agentes, observar ambas as remoções pode identificar os dois: não se
deve afirmar ambiguidade em qualquer caso de redundância.

A auditoria executada enumera todas as atribuições binárias compatíveis com
a original e todas as remoções isoladas. Em 12.600 consultas por cena (150
cenas × 84 curvas/horizontes), 3.854 são positivas; 1.052 delas (27,30%) têm
dois ou mais bloqueadores no subconjunto anotado. Dos 5.087 predicados positivos
por agente único, 2.285 (44,92%) não mudam a resposta composta quando o próprio
agente é removido isoladamente, porque outro agente continua bloqueando.

Esses números vêm de aritmética sobre as anotações, não de novas edições
múltiplas ou de respostas observadas da VLA. Dependem da grade de consultas,
da margem de 2 m e da seleção de objetos. Não são prevalências gerais de
redundância no trânsito.

## Referências e baseline de identificação

Uma observação que retorna OR dos itens presentes tem relação direta com
[group testing](https://arxiv.org/abs/1902.06002), cuja literatura trata
identificação, ruído e desenho de testes. A equivalência de restrições sob
consultas parciais não deve ser apresentada como uma descoberta nova apenas
por aplicar esse vocabulário a direção. O exemplo acima é uma dedução elementar
da definição, verificada por enumeração no código.

Definir o conjunto de atribuições compatíveis `Theta(D,q)` com as observações
disponíveis. Para uma remoção composta nova, os limites de identificação são

`L_A(q) = min_{r em Theta(D,q)} B_A(q)`

`U_A(q) = max_{r em Theta(D,q)} B_A(q)`.

Se os limites diferem, as consultas observadas não identificam a resposta sem
hipóteses adicionais. Um ponto estimado pela rede não elimina essa ambiguidade.
Geometria compartilhada entre consultas pode acrescentar hipóteses e reduzir o
conjunto; a enumeração por consulta não prova impossibilidade para todos os
modelos geométricos. É um baseline sem estrutura entre consultas.

## Papel dos próximos experimentos

O piloto de sobreviventes responde primeiro se a aparência dos pares ajuda a
prever restrições preservadas. Não aprende ausência nem composição, e seus
rótulos binários já usam anotações 3D. Um ganho de augmentação seria evidência
de utilidade dos pares para robustez, ainda insuficiente como contribuição ICML.

A etapa seguinte precisa separar: aprendizado de restrições por objeto com
rótulos individuais; aprendizado somente com respostas compostas; informação
adicional dos pares de imagem; e efeito de consultas de remoção de vários
agentes. Comparar com OR/max, DeepSets e identificação por enumeração/grupo,
mantendo os mesmos dados, entradas e custos. Remoções múltiplas terão de ser
verificadas independentemente quando forem interpretadas como mudanças visuais
ou físicas, em vez de simples operações em uma lista de restrições conhecidas.

O avanço de nível ICML, se houver, deverá estar em aprendizagem/generalização
sob supervisão parcial e edições imperfeitas além desses baselines conhecidos.
Não basta compor caixas, acrescentar uma loss de invariância ou chamar a
enumeração de um certificado de segurança.
