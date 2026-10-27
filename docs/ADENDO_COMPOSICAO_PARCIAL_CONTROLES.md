# Controles adicionais do piloto de composição

Adicionado após o início do piloto, antes de treinar/avaliar a variante
geométrica conjunta sem features espaciais. O protocolo principal congelado
foi preservado. Não é uma confirmação independente nem pré-registro público.

1. Treinar `geometry_joint` sem PCA/features da VLA, mantendo metadados,
   priors, arquitetura, dados parciais, splits, seeds, orçamento e seleção.
   O tamanho da entrada e a capacidade mudam ligeiramente; reportar parâmetros.
2. Compor as previsões dos nove modelos antigos treinados com rótulos
   individuais. Usar OR e máximo, calibrando o limiar composto apenas nos
   pares de validação. Esses modelos tiveram supervisão e seleção de checkpoint
   mais fortes; são referências, não baselines de orçamento igual nem limites
   superiores garantidos. Nenhum novo treino nessa referência.

A motivação é separar o ganho da geometria/objetivo daquele das features e
explicitar o efeito da informação disponível. Não selecionar um controle por
ter obtido um ponto favorável em alguma divisão.
