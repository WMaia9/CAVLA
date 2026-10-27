# Próximos experimentos e decisões

Atualização mais recente: [180 adaptações e alinhamento real](RESULTADOS_SUPERVISAO_INCERTA.md)
rejeitaram a mistura conjunta simples. O controle detectou um atalho de rótulos,
que foi corrigido em rodada separada. A [pesquisa de método](PESQUISA_SUPERVISAO_INCERTA_ICML.md)
explicita concorrentes e informação normativa ainda ausente nos dados reais.

Nova atualização: relações parciais passaram a triagem e a confirmação em dez
seeds, mas uma baseline simples ancorada reproduz quase todo o ganho. O teste
de seleção por erro de efeito falhou. Os [resultados e próximos critérios](RESULTADOS_RELACOES_E_UTILIDADE.md)
registram 100 adaptações e 240 micro-adaptações novas; não há evidência de VLA
real ou novidade de ICML nesta rodada.

Atualização: o teste controlado de aquisição ativa v1 foi implementado. O
[protocolo](PROTOCOLO_AQUISICAO_V1.md), o
[adendo de controles](ADENDO_CONTROLES_AQUISICAO_V1.md) e os
[resultados](RESULTADOS_AQUISICAO_V1.md) registram o que foi efetivamente testado.
Isso não substitui as verificações de protocolo nativo e referências reais abaixo.

## Escopo autorizado e estado

O usuário autorizou pesquisar ideias e iniciar execução com objetivo de ICML.
Já foram executados os pilotos, a auditoria e inferências novas descritos nos
resultados. Nenhuma execução de treinamento grande ficou em segundo plano.
Os itens abaixo são a fila técnica, não resultados concluídos.

## 1. Verificar o protocolo antes de ampliar coleta

- Reproduzir uma entrada nativa com vídeo/múltiplas câmeras do modelo escolhido.
  Comparar com o protocolo frontal com padding que acabamos de reproduzir.
- Conferir temporalidade: uma edição de objeto precisa ser consistente entre
  frames e câmeras relevantes, ou ser descrita como perturbação do sensor.
- Confirmar o condicionamento de navegação de um backbone apropriado, sem
  assumir essa capacidade no Alpamayo 1 original.
- Preservar câmera, sample token, timestamps, instrução, seeds por estágio,
  checkpoint e hashes. Nenhuma chamada externa com dados do laboratório é necessária.

Critério: se a resposta for dominada pela adaptação incorreta do input, corrigir
o adaptador antes de produzir mais métricas de causalidade.

## 2. Estudo de suporte e orçamento de intervenções

Lote de desenvolvimento proposto: 8 cenas × 3 objetos × 16 seeds, compartilhando
a inferência original entre os objetos. Isso exige 8×(1+3)×16 = 512 inferências,
além de no-ops repetidos. A aproximadamente 2,3–2,7 s por inferência no smoke
frontal, isso representa cerca de 20–24 minutos de inferência bruta. O protocolo
nativo pode ter custo diferente; medir antes de prometer orçamento.

Escolher cenas por uma regra fixa que cubra parado/em movimento, objetos
próximos/distantes e diferentes classes, não por resposta observada do modelo.
Separar cenas de desenvolvimento, calibração e confirmação final.

Comparações necessárias: resposta por seed acoplada, diferença de médias,
distância distribucional, resposta de no-op e custo de amostragem. Usar U-estatísticas
e MMD como referências existentes. Não interpretar resultado negativo de MMD
não truncada como distância negativa da população, nem como prova de efeito nulo.

Critério: demonstrar que o estimando e o orçamento mudam de modo reproduzível
uma decisão útil (por exemplo, qual intervenção precisa de supervisão adicional).
Se a conclusão for apenas que poucas amostras têm variância alta, não há uma
contribuição de ICML.

## 3. Fonte independente do efeito correto

Antes de alinhar uma política a efeitos, decidir a origem dos alvos:

- Referência controlada com estado e dinâmica conhecidos: causalidade restrita
  ao ambiente; permite verificar efeito e intervenção.
- Anotação especialista/planejador independente em dados reais: explicitar
  ambiguidade, validade e possíveis erros da referência.
- Efeitos do próprio modelo congelado: usar apenas para auditoria/distilação,
  sem chamá-los de ground truth decisional correto.

Não usar a trajetória original logada como verdade após remover um objeto.
Não chamar nenhuma mudança de instrução de fisicamente realizável sem conferir
o estado, a rota e o horizonte.

## 4. Método candidato e comparação exigente

A hipótese de pesquisa é combinar estimativa confiável de efeitos, seleção
econômica de intervenções e aprendizagem da resposta relevante à tarefa.
Ainda precisa de uma contribuição algorítmica/formal distinta de ECT, CRAFT,
causal bisimulation e atribuição estocástica já existentes.

O primeiro método que merecer fine-tuning deve passar por: BC com os mesmos
dados/rótulos; contraste genérico; head auxiliar; ECT/CRAFT quando aplicáveis;
alinhamento com futuro; labels embaralhados; placebo; controle de magnitude
visual; e comparação de orçamento por consultas e tempo.

Critério: ganhos consistentes no comportamento e em intervenções novas, sem
atribuir à causal loss o que vier de mais rótulos ou de treino insuficiente da
baseline. O piloto de 400/4.000 passos já demonstrou por que este critério importa.

## 5. Decisão ICML

Só congelar a narrativa de submissão depois de demonstrar:

1. Distinção de novidade que sobreviva à leitura dos concorrentes diretos.
2. Estimando, hipóteses e fonte de referência definidos.
3. Sinal confirmado em dados que não selecionaram método ou hiperparâmetros.
4. Utilidade para a política, além de métricas auxiliares.
5. Generalidade além de um toy ou um único checkpoint, na medida necessária
   para sustentar a alegação do artigo.

Se qualquer ganho relevante desaparecer contra a melhor baseline simples,
registrar o resultado e reformular antes de consumir dias de fine-tuning.
