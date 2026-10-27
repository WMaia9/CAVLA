# Correção técnica da mistura — v2

A primeira execução v1 foi interrompida depois de observar resultados parciais,
por erro na implementação das probabilidades extremas. O código fazia clipping
de p=1 para 1-1e-6; para perdas suficientemente grandes, isso permitia ao ramo
não informativo ignorar uma restrição **aprovada com certeza** pelo auditor.
Isso não implementava a fórmula registrada para p=1.

Correção: p=1 retorna exatamente a perda da restrição; p=0 retorna zero. Apenas
probabilidades interiores usam clipping numérico e logaddexp. Acrescentamos teste
do valor e do gradiente nos dois extremos. Não alteramos tau, pesos, arquitetura,
seeds, dados, auditorias ou checkpoint selection. Repetir todas as cinco seeds em
novo diretório v2. A execução v1 permanece arquivada com INVALIDATED.json e não
será usada para conclusões. Isto é correção de implementação, não confirmação
independente: resultados parciais de v1 já haviam sido observados.
