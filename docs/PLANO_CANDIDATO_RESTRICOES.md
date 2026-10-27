# Desenvolvimento do candidato: restrições por objeto sob intervenções imperfeitas

Decisão de trabalho autorizada em 21/10/2026. O candidato tem evidência
exploratória para um decoder geométrico com features espaciais; ainda não tem
novidade estabelecida nem melhora de política demonstrada.

**Revisão após o holdout:** [a versão atual falhou no controle físico sem rede](RESULTADOS_COMPOSICAO_PARCIAL_E_HOLDOUT.md).
Sua contribuição central foi rejeitada. Os itens iniciais abaixo são histórico;
as próximas decisões devem partir da regra estacionária forte e do diagnóstico
de que o alvo atual exige pouca informação de movimento. Não escalar uma nova
loss sem antes justificar um problema de aprendizagem que essa regra não resolva.

Gate obrigatório antes de novos treinos: avaliar regra estacionária, baseline
cinemática com entradas comparáveis e checkpoint zero. Fixar consultas usando
somente informação disponível no passado; mostrar em que decisões movimento
ou semântica acrescentam informação. A amostra de 16 logs já foi consultada e
não pode voltar a ser chamada de intocada após ajustar o método por seus erros.

## Pergunta central

Como aprender restrições úteis para decidir sob remoções de agentes quando
as consultas disponíveis não identificam todas as restrições individuais e
as imagens editadas também mudam aspectos que não correspondem à intervenção?

O objetivo é aprender respostas e incerteza sobre consultas de ação, evitando
exigir uma nova trajetória arbitrária a cada edição. As hipóteses de mundo
congelado devem ficar explícitas; o alvo atual ainda não inclui tráfego reativo.

## Frentes e critérios

| Frente | Evidência necessária | Estado |
|---|---|---|
| Representação original | Ganho contra mesmas entradas, priors, capacidade e seleção; divisão por gravação | Executado: 13,93% contra 18,79% de erro balanceado |
| Custo do backbone | Superar features visuais baratas, depois outro encoder pré-treinado | Controle sem pré-treino executado: 16,34%; outro encoder pendente |
| Utilidade dos pares | Comparar augmentação e consistência com original repetida e ruído de mesma magnitude | 36 modelos concluídos; usos simples sem ganho claro, rejeitados como argumento central |
| Informação parcial | Limites explícitos para respostas não identificadas; testar composição | 54 modelos corrigidos, sem ganho claro da NLL conjunta sobre BCE; regra sem rede venceu ambos |
| Validade visual | Revisão independente da identidade removida e da preservação dos demais agentes | Filtro de pixels e folha de revisão prontos; não verificado por anotadores |
| Contribuição original | Superar OR/max, modelos de conjuntos, identificação por enumeração/grupo e baselines de risco | Pendente; não chamar a formulação elementar de novo teorema |
| Utilidade de ação | Mostrar ganho na seleção/geração de ações com o mesmo orçamento | Pendente |
| Generalização | Novo conjunto congelado, outro backbone/editor; tráfego reativo para alegações comportamentais | 73 cenas/16 logs novos avaliados; composto usa 24 cenas/13 logs. Ganho específico da VLA não confirmado; outras transferências pendentes |

## Ordem após o piloto de sobreviventes

1. Fixar o veredito do piloto sem escolher somente seeds, divisões ou métricas
   favoráveis. Um ganho que empate com ruído não evidencia informação causal.
2. Construir o treino com rótulos compostos parciais: original e remoções
   isoladas; testar remoções de múltiplos agentes nunca observadas no treino.
   Distinguir operações conhecidas em uma lista de restrições de edições
   visuais/remoções físicas verificadas.
3. Comparar aprendizagem por objeto totalmente supervisionada, aprendizagem
   somente com rótulos compostos e uso adicional de imagens pareadas, com
   contagem explícita de consultas/rótulos. Usar os limites de identificação
   como baseline; imagens e geometria acrescentam hipóteses, não eliminam
   magicamente a ambiguidade dos rótulos.
4. Se houver vantagem específica das imagens pareadas, testar um editor ou
   alteração independente e outro encoder. Reservar uma amostra nova antes
   de ajustar a versão candidata; os 150 exemplos atuais já são exploratórios.
5. Integrar com escolha de trajetórias somente após vencer esses controles.

## Decisões de abandono

Se os pares não trouxerem ganho sobre originais/ruído, abandonar essa forma
simples de regularização como argumento central. Se somente a geometria
conhecida explicar o ganho, preservar o sistema como baseline de engenharia.
Se um modelo de conjuntos ou regras OR reproduzir o ganho composicional,
não promovê-lo como contribuição original para ICML.

CARLA não é necessário para executar as etapas atuais. Alegações sobre reação
de outros agentes e segurança de ações exigirão avaliação apropriada em uma
etapa posterior; os rótulos de movimento registrado não respondem a isso.

Protocolo: [pares de sobreviventes](PROTOCOLO_PARES_SOBREVIVENTES_V1.md).
Formulação e baseline: [restrições compostas](FORMULACAO_RESTRICOES_COMPOSTAS.md).
