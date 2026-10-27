# Adendo: controles com o mesmo orçamento de repetições

Adicionado durante a execução exploratória v1, depois de inspecionar a primeira
replicação (seed 101). O protocolo original permanece inalterado e preservado.
Não é uma comparação confirmatória pré-registrada.

Motivo: a regra adaptativa usou todo o limite de repetições nos casos ruidosos
da primeira replicação. Compará-la apenas com sondas fixas de duas amostras não
separa o benefício de mais repetições do benefício de escolher onde repeti-las.

Dois controles adicionais receberão **exatamente o mesmo número de repetições
extras efetivamente gasto pela regra adaptativa em cada caso**, sem consultar
nenhum resultado de teste para escolher esse número:

- Repetições uniformes entre candidatos/membros do ensemble.
- Repetições com candidatos/membros escolhidos aleatoriamente.

Ambos usam o score corrigido após as repetições e solicitam os rótulos restantes.
Como o pool é estático e os modelos congelados durante aquisição, a comparação
isola a localização das novas medições, com o mesmo número de referências,
sondas iniciais e repetições extras. O treino e os testes são os mesmos do v1.

Os ensembles iniciais e conjuntos do estudo principal serão reutilizados,
verificando hashes, sem novo pré-treino. Não contamos os controles como novas
replicações independentes: são comparações pareadas nas mesmas cinco seeds.

Mesmo que passe o gate original, uma regra que não supere esses controles
não demonstra benefício da alocação adaptativa. A estimativa da variância
subtraída também não é automaticamente não enviesada sob parada adaptativa;
essa propriedade é verificada apenas para tamanhos fixos na unidade de teste.
