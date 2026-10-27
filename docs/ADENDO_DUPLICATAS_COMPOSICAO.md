# Correção de detecções duplicadas na composição

Após o primeiro piloto de composição, a auditoria encontrou seis pares de
detecções associados à mesma instância 3D, em seis cenas. Os 584 ROIs anteriores
correspondem a 578 instâncias distintas por cena. Isso não invalida o readout
por detecção, mas pode criar redundância artificial ao interpretar exclusões
como remoções de agentes distintos.

O dataset v2 mantém uma detecção por `(scene, instance_token)`: maior IoU
projetado, desempate por stem. A identidade de exclusão prévia é mapeada para
essa representante quando necessário. Todas as cinco variantes e o controle
sem features são retreinados nas mesmas divisões, com os mesmos hiperparâmetros
e seleção. Os números v1 ficam preservados e marcados como substituídos para
alegações sobre agentes únicos; a correção foi feita após ver resultados v1.

Os 584 vetores originais continuam no ajuste de PCA/normalização de treino,
igual em todas as variantes, preservando a referência anterior. Só 578 entram
nos conjuntos supervisionados e na composição. Nenhuma feature de validação
ou teste entra nesse ajuste. Portanto, não alegamos ter removido esses seis
vetores também da preparação não supervisionada das features.

Após a correção: 1.052/3.854 consultas positivas por cena têm redundância
(27,30%); 2.285/5.087 predicados positivos por agente ficam mascarados na exclusão
isolada (44,92%). Os números anteriores, 27,53% e 45,29%, eram por detecção.
O fenômeno permanece, mas sua magnitude deve ser reportada na versão corrigida.

Para o holdout, a mesma deduplicação é aplicada antes de compor os conjuntos.
Os resultados desse holdout ainda não foram consultados ao corrigir o piloto.
