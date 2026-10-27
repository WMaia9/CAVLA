# Candidato em teste: preservar respostas a intervenções ao comprimir uma VLA

20/10/2026. Hipótese em avaliação; novidade e vantagem não estabelecidas.

## Pergunta e uso legítimo do dataset

Uma política comprimida pode aproximar trajetórias comuns e perder a resposta
que sua professora teria à remoção de um objeto. Queremos aprender uma interface
compacta que preserve essas diferenças, sob orçamento de memória fixo.
Counter-nuScenes fornece os pares visuais necessários. A professora é o próprio
Alpamayo congelado: a meta é fidelidade a essa política, não aprender a dirigir
corretamente a partir das suas próprias previsões.

Isso evita exigir trajetórias contrafactuais normativas que o arquivo local não
contém. Também limita a conclusão: copiar um erro da professora continua sendo
um erro. Sucesso e segurança exigem referências e avaliação adicionais.

## Por que a formulação antiga perdeu prioridade

[Cartender](https://arxiv.org/abs/2610.26390), submetido em 19/10/2026, já combina
alternativas contrafactuais para direção, compatibilidade cena/instrução,
instruções inviáveis e avaliação comportamental com a cena fixa. A variante
simples de supervisão incerta que testamos já havia perdido para filtros; esse
trabalho torna ainda menos defensável apresentar a combinação como novidade.

## Trabalhos próximos que precisamos superar

- [Goodfire: Causal analysis improves VLA efficiency](https://docs.goodfire.com/assets/reports/vla-interface.html)
  já usa intervenções na interface VLM/expert e testa arquiteturas mais econômicas.
  Localizar camadas em Alpamayo não seria contribuição suficiente.
- [RIFT](https://arxiv.org/abs/2608.11521) já intervém em caches e aprende uma
  construção eficiente do cache futuro. Não propomos chamar qualquer cache
  aprendido de novo mecanismo causal.
- [Not All Features Are Created Equal](https://arxiv.org/abs/2603.19233) já reúne
  intervenções em várias VLAs, subespaços e execução física. Um estudo só de
  probing tem uma comparação muito exigente.
- [STAR-KV](https://github.com/PriyanshBhatnagar/STAR-KV) aprende compressão
  espectral adaptativa; SVD e escalas por cabeça são controles, não novidade.
- [Sobolev Training](https://papers.neurips.cc/paper_files/paper/2017/hash/758a06618c69880a6cee5314ee42d52f-Abstract.html)
  já transfere derivadas, inclusive para políticas. Acrescentar diferenças finitas
  à distillation é uma baseline conceitual forte, não uma descoberta suficiente.
- [Anatomy of a Closed-Loop Collapse](https://arxiv.org/abs/2609.23048) documenta
  uma política comprimida que passa avaliações offline e falha em execução.
  Fidelidade offline não substitui validação comportamental.

## Experimentos reais desta rodada

1. Intervenções no cache de Alpamayo-1: original, editado completo, prefixo,
   posições de imagem, segmento gerado, última posição, deltas de rank 4/16,
   projeção aleatória rank 16 e híbridos que preservam o segmento gerado editado.
2. Três cenas iniciais e doze adicionais em uma sequência previamente sorteada,
   sem selecionar pela resposta. Quatro seeds do decoder; seed de linguagem 123.
3. Seis outras cenas para um primeiro treino: três de treino e três de teste,
   separadas por cena antes do treino. Referências do decoder fixadas separadamente
   da geração de linguagem. O replay original/editado deve coincidir com a
   implementação externa em erro médio de trajetória inferior a 1e-4 m.

O segundo experimento de aprendizagem comprime o cache **absoluto** de ambas
as condições por aproximações centradas de rank 16 por cabeça/camada e aprende
escalas espectrais globais. Não comprime apenas o delta em torno de um original
mantido exato: nesse caso, preservar a diferença e ajustar o endpoint editado
seriam o mesmo objetivo, uma distinção de nomes sem distinção matemática.

Comparamos, com os mesmos dados e orçamento:

- SVD sem calibração;
- distillation dos dois endpoints;
- endpoints mais preservação da diferença dos campos de velocidade;
- endpoints reponderados pela energia da diferença, sem termo cruzado.

São 18 pontos de calibração (3 cenas × 2 ruídos × 3 tempos), 36 updates Adam,
taxa 0,02 e escalas limitadas a [0,2]. O termo adicional tem peso 0,25 e piso
de energia 0,05. As oito seeds de avaliação 9201–9208 não entram no treino.
São decisões de um smoke test, não hiperparâmetros escolhidos pelo teste.

O erro principal compara **vetores** de resposta ao longo da trajetória,
`(student_edit − student_original) − (teacher_edit − teacher_original)`.
Reportamos também os dois erros de endpoint. Não basta igualar a magnitude da
mudança, pois mudar para o lado errado poderia obter a mesma magnitude.

## Critério para continuar, definido antes de ler a avaliação do treino

Exigir redução de pelo menos 20% no erro de resposta contra os dois controles
treinados, sem aumentar em mais de 10% o erro factual contra endpoint, e verificar
se o ganho aparece nas três cenas. Esse limiar é uma decisão de engenharia para
escalar o piloto, não um teste estatístico nem evidência suficiente para ICML.
Se passar, testar mais passos, diferentes inicializações/ordens, ranks e
cenas novas; comparar com calibração de trajetórias, Sobolev/Jacobian e métodos
de compressão atuais sob o mesmo custo.

Uma candidatura de paper ainda exige um método além desta loss simples,
ganhos em cenas novas e outro backbone, controles para o editor e entradas
nativas, além de memória/latência efetivamente medidas e desempenho em execução.
O código atual reconstrói os caches densos; **não mede aceleração de inferência**.
Todas as entradas desta rodada ainda replicam câmera frontal estática e padding
do arquivo local, sem validar vídeo multiview nativo.

Três cenas de teste não permitem afirmar generalização. Seeds do decoder
replicam amostragem condicionada a uma única geração de linguagem por imagem,
não a incerteza completa da VLA. Preservar diferenças sob esse acoplamento de
ruído também não equivale a preservar toda a distribuição de ações.
