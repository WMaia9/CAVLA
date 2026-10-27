# Relações de efeito como supervisão fraca — protocolo exploratório v1

Registrado antes da primeira execução deste experimento. Motivação: os pilotos
anteriores já possuíam os quatro rótulos exatos do quarteto, tornando a loss de
efeitos uma combinação de supervisão existente. Aqui a nova informação é apenas
uma relação categórica sobre a mudança, sem o valor correto da nova ação.

## Pergunta e controle principal

Relações condicionadas à instrução (diminuir, manter, aumentar) melhoram a ação
em intervenções não vistas, além do que já se obtém transferindo rótulos factuais
para alterações comprovadamente irrelevantes? Este último controle é obrigatório:
um ganho de consistência a fundo pode ser apenas augmentação válida comum.

Não alegamos que supervisão direcional, monotonicidade ou invariância sejam novas.
São hipóteses a confrontar com weak/coactive learning, ECT/CRAFT e métodos de
abstração causal. Um resultado positivo seria um sinal para investigação, não
uma contribuição de nível ICML pronta.

## Referência e dados

Mesmo ambiente visual de duas faixas, CNN e dois IDs de instrução. O gerador
controlado fornece a referência independente do modelo. O treino factual tem
224 cenas, duas instruções e 448 rótulos de ação conhecidos. Escolhemos 96 dessas
cenas, sem observar erros do modelo, e geramos três tipos de intervenção em
partes iguais: aproximar o obstáculo; trocar o fundo; trocar o billboard.

Para cada intervenção, um anotador simulado retorna apenas a relação por
instrução/componente: -1, 0 ou +1. O trainer fraco nunca recebe os valores exatos
de ação contrafactual. Margem 0,01 em unidades normalizadas; o gerador verifica
que toda mudança não nula é maior que essa resolução. Esta é uma hipótese forte
do piloto, não um custo/qualidade de anotação humana já medido.

Pares de fundo/billboard com efeito nulo podem herdar os rótulos factuais
previamente conhecidos. Para efeitos não nulos isso é inválido. Rótulos exatos
contrafactuais são usados apenas em comparadores explicitamente supervisionados
e na avaliação, nunca escondidos no método fraco.

Testes independentes de 256 cenas por grupo:

- ID: intervenções da distribuição de treino, para verificar retenção.
- Severidade: aproximações maiores que no treino.
- Troca de faixa do obstáculo: mecanismo não supervisionado nas relações.
- Composição: troca de faixa e fundo juntas, ausente do treino de relações.

Cada grupo contém duas instruções e mundos factual/editado. A ação de referência
é de um passo, sem dinâmica física ou avaliação de segurança real.

## Métodos

1. Continuação factual sem novos pares.
2. Augmentação válida nula (`null_copy`), com rótulos factuais transferidos apenas
   nos pares conhecidos como nulos — controle principal.
3. Consistência só nos componentes com relação zero (`weak_null`).
4. Relações categóricas completas, zero e direção com margem (`weak_relation`).
5. Augmentação nula mais restrições direcionais (`null_copy_relation`).
6. Invariância em todos os pares, incluindo efeitos relevantes — controle negativo.
7. Dezesseis quartetos contrafactuais com ações exatas (`full_cf16`).
8. Noventa e seis quartetos com ações exatas (`full_cf96`) — referência forte,
   não comparador com o mesmo tipo/quantidade de supervisão.

Todos usam a mesma CNN inicial, replay factual e 1.200 passos de adaptação.
Pré-treino factual: 2.000 passos. Peso das relações: 0,5; sem busca após resultados.
Checkpoints são selecionados somente na validação factual independente (96 cenas),
a cada 100 passos. Reportar tempo efetivo porque losses com dois mundos custam mais
por passo. Nenhum custo humano em créditos é inventado: reportamos separadamente
rótulos factuais, relações e ações contrafactuais exatas.

Cinco seeds novas de dados e modelo: 616, 717, 818, 919, 1020. Nenhuma dessas
cenas usa os conjuntos de teste dos pilotos anteriores. O smoke técnico usa
outra seed e não participa das conclusões científicas.

## Métricas e gate de continuação

Primária: MSE de ação médio nos três grupos não ID (severidade, troca de faixa,
composição). Secundárias: cada grupo separadamente; fidelidade da mudança de ação;
taxa de direção correta para efeitos não nulos; resposta indevida em efeitos nulos;
retenção ID; e tempo/supervisão utilizados.

Gate gerencial: ao menos uma das duas variantes com direção reduzir a média
primária em >=10% contra `null_copy` e continuação factual, vencendo ambos em
ao menos quatro das cinco seeds, sem aumentar o MSE ID médio em mais de 10%
contra `null_copy`. Reportar todas as variantes, sem trocar a métrica após ver
os resultados. Cinco seeds são exploratórias, não uma garantia estatística.

Se houver sinal, confirmar sem alterar a configuração em cinco novas seeds
(1121, 1222, 1323, 1424, 1525), comparar custo real de treino e depois decidir
se outro ambiente/backbone é justificado. Se o ganho vier apenas de mudanças
nulas já cobertas por augmentação comum, não promover isso como nova causal loss.

## Segunda investigação: prever utilidade de dados

Em paralelo à triagem da literatura, planejamos medir se proxies de erro e
gradiente predizem o ganho real de micro-adaptações em calibração independente.
DataMIL, Quality over Quantity e Learning Loss for Active Learning são trabalhos
próximos e devem ser tratados como baselines/ameaças de novidade. Esse diagnóstico
não será anunciado como aquisição sem rótulos quando consultar rótulos para
medir utilidade; custos de sondagem precisam permanecer explícitos.
