# CAVLA — pilotos para uma contribuição de nível ICML

Pesquisa e execução iniciadas em 20/10/2026. O projeto está em fase de
falsificação de hipóteses; não há resultado de segurança nem método com novidade
estabelecida. CARLA não é necessário para os pilotos atuais.

**Rodada de 22/10 (noite):** com [200 clipes reais do PhysicalAI-AV](docs/RESULTADOS_DECISAO_E_INFORMACAO_20261022.md),
solvers baratos mudaram decisões do Alpamayo1.5 em 6–13% dos casos (o cache estilo
FlashDrive teve 12,8%), sem que minADE mostrasse isso. Essas inversões só ocorreram
em clipes nos quais a própria política é incerta entre seeds (0% quando as 6 seeds
concordam). A [nova candidata](docs/CANDIDATO_INFORMACAO_DA_DECISAO.md) aloca os passos
pela informação da decisão, em vez da saída inteira; a [teoria](docs/TEORIA_INVERSOES_DE_DECISAO.md)
mostra que simetria impede inversões e que o tempo entrópico piora com a dimensão. No
holdout, o ganho teve direção consistente e foi modesto; a confirmação com 436 clipes
inéditos está em andamento. Solvers grosseiros ficaram mais perto do motorista humano,
então fidelidade ao fluxo não equivale a acerto.

Na [rodada de controle e solvers de22/10](docs/RESULTADOS_ACOPLAMENTO_E_SOLVERS_20261022.md),
localizamos a instalação CARLA completa, coletamos inputs com quatro câmeras e
histórico temporal e executamos96 rollouts numéricos do Alpamayo1.5 congelado.
Uma mudança de solver chegou a trocar aceleração por frenagem no diagnóstico
offline do PID; isso não identifica o comando correto. Solvers clássicos já
explicam parte importante do ganho. A [nova hipótese de método](docs/CANDIDATO_INFERENCIA_COM_ESTADO_DO_CONTROLADOR.md)
é aprender controle do erro na decisão e na memória do executor durante
inferência parcial. Originalidade, estimador aprendido e ganho closed-loop
permanecem sem validação.117 testes de código passam.

**Atualização de22/10/2026:** os [testes ampliados de transições](docs/RESULTADOS_TRANSICOES_E_OBSERVABILIDADE_20261021.md)
incluíram4.614 veículos, ego-history, mapas, vizinhos, árvores calibradas,
CLIP/R1/Qwen e controles com quatro imagens e vídeo nativo. O ganho visual
inicial não resistiu de forma convincente à seleção de famílias; história real
não superou repetição do quadro atual. A linha não foi promovida a paper.

A [candidata teórica em avaliação](docs/CANDIDATO_RESTRICOES_LATENTES_E_ORDEM_INTERVENCOES.md)
estuda quanta informação precisa ser conservada para responder a remoções
combinadas. Há exemplos exatos de não identificação, uma proposição de
dimensão em uma classe restrita e limites de codificação verificados por testes.
Feedback clássico também alcança a mesma ordem de custo. Originalidade e
transferência para VLA permanecem abertas; a política arquivada não pode ser
assumida como um sistema de menor tempo viável. **108 testes de código passam**
no ambiente Torch local; isso não certifica os argumentos matemáticos ou novidade.

**Estado da composição:** a [composição parcial e o teste em logs novos](docs/RESULTADOS_COMPOSICAO_PARCIAL_E_HOLDOUT.md)
rejeitaram a versão atual como contribuição central: regra geométrica sem rede
teve 7,53% de erro balanceado contra 17,13% do modelo aprendido no holdout
composto. O controle foi adicionado posteriormente e calibra só na validação
antiga. A pipeline fica como baseline; não há ainda candidato ICML estabelecido.

Na [busca temporal subsequente](docs/RESULTADOS_BUSCA_TEMPORAL_20261021.md),
avaliamos risco por intervalo e informação visual além da velocidade constante.
O movimento importa, mas os 18 readouts selecionados por validação preferiram
correção zero. Não houve ganho confirmado das features VLA atuais; 68 testes passam.

A [rodada semântica](docs/RESULTADOS_SEMANTICA_E_ABLACOES_20261021.md) encontrou
decodificação exploratória de parked/stopped (AUC 0,796 contra 0,685 do CLIP crop
e 0,637 do controle numérico). Ablações confirmam dependência da imagem nesses
ROIs, mas a vantagem sobre contexto da cena é incerta e o teste simples de
transferência não mostrou ganho convincente. Isso ainda não constitui um paper.
Os 71 testes de código atuais passam.

A [replicação multimodelo](docs/RESULTADOS_MULTIMODELO_E_CONTEXTO_20261021.md)
executou Alpamayo R1/1.5/2 Super e Qwen3-VL8B na GPU local. Em18 cenas mistas
de14 logs, o R1 teve AUC dentro da cena0,899 contra0,622 do controle numérico.
A avaliação separada ainda tem só uma cena mista; não há ganho de decisão ou
método ICML estabelecido. Os72 testes atuais passam.

A [busca em física, informação e controle](docs/RESULTADOS_FISICA_INFORMACAO_20261021.md)
executou novos testes de sensibilidade cinemática e dispersão dos erros. Uma
mistura sem visão superou a primeira melhora probabilística visual; com família
comparável, o ganho VLA sobre numérico ficou pequeno e incerto. Não escalar este
protocolo como candidato. A investigação de identificação de decisões permanece
aberta, sem método original validado. **79 testes passam.**

Atualização de 21/10: o [protótipo de restrições por objeto](docs/RESULTADOS_GEOMETRIA_E_EDICAO_20261021.md)
foi treinado em 584 ROIs de 150 cenas locais, com features espaciais novas do
Alpamayo congelado. Na separação por gravação, o erro balanceado foi 13,93%
contra 18,79% do readout direto com as mesmas features, com seleção e calibração
apenas na validação interna. Há um ganho exploratório;
não é ainda um método causal validado nem uma contribuição original estabelecida.
O relatório inclui sobreajuste, controles de edição, limitações e fontes primárias.

Desenvolvimento do candidato: [45 treinos adicionais e resultado dos pares](docs/RESULTADOS_PARES_SOBREVIVENTES_V1.md).
Augmentação e consistência simples não trouxeram ganho claro sobre original/ruído.
O controle de pixels sem pré-treino ficou em 16,34%, contra 13,93% com features
da VLA. [Plano](docs/PLANO_CANDIDATO_RESTRICOES.md) e
[formulação de composição com informação parcial](docs/FORMULACAO_RESTRICOES_COMPOSTAS.md).

A [auditoria de duplicatas](docs/ADENDO_DUPLICATAS_COMPOSICAO.md) encontrou que
esses 584 ROIs representam 578 instâncias distintas por cena. Os novos testes
de composição usam uma representante por instância e retreinam todos os métodos.

- [Pesquisa, ideias e critérios de decisão](docs/PESQUISA_ICML_20261020.md)
- [Estimandos e falhas da formulação](docs/ESTIMANDOS_E_FALHAS.md)
- [Resultados executados](docs/RESULTADOS_INICIAIS.md)
- [Protocolo de aquisição contrafactual](docs/PROTOCOLO_AQUISICAO_V1.md)
- [Resultados de aquisição e controles de custo](docs/RESULTADOS_AQUISICAO_V1.md)

## Executar

O pacote requer Python >=3.11, NumPy, SciPy, Pillow e Matplotlib. Treinamento e
inferência requerem PyTorch. Nenhum ambiente compartilhado foi modificado.
Nesta máquina, os experimentos PyTorch usam o interpretador existente:

```bash
export CAVLA_PYTHON=/mnt/datasets/Explainability/alpamayo/ar1_venv/bin/python
PYTHONPATH=src python -m unittest discover -s tests -v
PYTHONPATH=src CUBLAS_WORKSPACE_CONFIG=:4096:8 "$CAVLA_PYTHON" -m cavla.pilot \
  --device cuda --out artifacts/controlled_convergence_20261020 \
  --steps 4000 --train-scenes 768 --test-scenes 512 --seeds 17 29 43
```

O piloto compara seis objetivos com os mesmos quatro rótulos por cena. Imagens
sintéticas e dois IDs de comandos não constituem uma VLA pré-treinada. Os
arquivos de manifesto registram hashes dos dados, código, versões e configuração.
Use uma pasta de saída nova para preservar execuções anteriores.

## Teste de aquisição ativa

```bash
PYTHONPATH=src CUBLAS_WORKSPACE_CONFIG=:4096:8 "$CAVLA_PYTHON" -m cavla.active_pilot \
  --device cuda --out artifacts/acquisition_v1_reproduction
PYTHONPATH=src CUBLAS_WORKSPACE_CONFIG=:4096:8 "$CAVLA_PYTHON" -m cavla.acquisition_controls \
  --device cuda --study artifacts/acquisition_v1_reproduction \
  --out artifacts/acquisition_controls_v1_reproduction
PYTHONPATH=src python -m cavla.acquisition_report \
  --study artifacts/acquisition_v1_reproduction \
  --controls artifacts/acquisition_controls_v1_reproduction \
  --report docs/RESULTADOS_AQUISICAO_REPRODUCTION.md
```

São cinco replicações independentes de dados/inicialização, três regimes de
ruído sintético e dois orçamentos. As consultas dos seletores são cobradas;
aleatória pode adquirir mais referências porque não paga por sondas. A política
é adaptada com BC comum em todos os métodos, a partir de rótulos adquiridos
de uma referência controlada. Os controles adicionais igualam exatamente o
número de repetições e rótulos da regra adaptativa e foram adicionados depois
da primeira replicação, conforme o adendo. Isto ainda não é fine-tuning de uma
VLA real, e os créditos não foram calibrados para tempo humano/GPU.

```bash
PYTHONPATH=src python -m cavla.audit_existing \
  --results-dir /mnt/datasets/Explainability/alpamayo/gemini_imgen/comp_results_fin_comb \
  --images-dir /mnt/datasets/Explainability/alpamayo/gemini_imgen/nuscenes_per_scene \
  --out artifacts/real_audit_20261020
PYTHONPATH=src python -m cavla.report --root .
```

O módulo `cavla.alpamayo_repeatability` executa o checkpoint real, offline. A
invocação completa, paths, seeds e hashes estão nos manifestos de
`artifacts/alpamayo_repeatability_20261020/` e
`artifacts/alpamayo_repeated16_20261020/`. O código externo é importado em modo
de leitura. O protocolo replica a câmera frontal única com padding do arquivo
analisado; isso não demonstra validade em vídeo multiview nativo.

## Estado científico

A rodada seguinte usou o Alpamayo real em 21 cenas locais: intervenções no cache,
seis calibrações treinadas e um controle fatorial de pixels. A loss simples de
preservação da resposta perdeu para controles nos ranks 16 e 64. O controle de
edição encontrou mudanças externas relevantes em seis cenas, motivando investigar
identificação sob edição imperfeita; ainda não há método validado para ICML.
Veja [resultados e candidato de pesquisa](docs/RESULTADOS_CANDIDATOS_REAIS.md).

A rodada de [supervisão incerta](docs/RESULTADOS_SUPERVISAO_INCERTA.md) executou
180 adaptações novas e verificou a proveniência dos 210 frames reais. A mistura
conjunta perdeu para filtros simples, inclusive no controle sem atalho de sinais;
foi rejeitada como candidata. A formulação seguinte exige referências
independentes para as restrições das tarefas nos pares reais.

A pesquisa específica sobre o laboratório e Counter-nuScenes está em
[Mi³, AV/VLA e ideias para ICML](docs/MI3_AV_VLA_IDEIAS_ICML.md).
O cruzamento local encontrou 146 cenas de CVAA com instruções do doScenes;
o alinhamento temporal e os referentes ainda precisam ser validados.

A nova rodada está em [relações parciais e utilidade de dados](docs/RESULTADOS_RELACOES_E_UTILIDADE.md):
100 adaptações em dez seeds e 240 micro-adaptações. Relações perfeitas reduziram
erro neste ambiente sintético, com confirmação independente de seeds; uma
baseline simples ancorada explica quase todo o ganho. Seleção por erro de efeito
não superou erro de ação. Isso não estabelece uma nova contribuição para ICML.

O piloto curto favorecia uma loss de interação; após dez vezes mais passos,
ela não venceu um contraste genérico na extrapolação de severidade. Não escalar
essa variante com base no resultado curto. A auditoria dos arquivos e as
repetições do modelo real dão suporte a desenhar novos testes, não a afirmar
melhora de driving ou novidade frente à literatura.
