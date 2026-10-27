# Busca de hipótese após rejeição do candidato geométrico

**Estado: ainda não há candidato com novidade e evidência suficientes para defender como paper ICML.** Esta rodada executou dois testes novos. O primeiro confirma que resolver o tempo importa; o segundo não confirmou vantagem incremental das features atuais da VLA. Não transformar esse resultado em mais uma proposta de loss.

## 1. Risco por intervalo, com controles cinemáticos

Código: `src/cavla/temporal_necessity_audit.py`. Resultado completo: `artifacts/temporal_necessity_v2_20261021/`.

Partimos de 578 instâncias distintas no desenvolvimento e 164 nos logs de avaliação anterior. Duas anotações passadas recentes estão disponíveis para 532 e 152 agentes, respectivamente. Todas as comparações abaixo usam esses mesmos subconjuntos. A extrapolação de velocidade constante utiliza somente anotações com timestamp menor ou igual ao timestamp real da câmera; intervalos e idades de anotações devem ser no máximo 0,75 s. O cache registra a procedência temporal de cada velocidade.

São 32 combinações distintas de velocidade/curvatura dos grids anteriores, agora avaliadas em janelas até 6,4 s, com agregação OR dos agentes presentes na cena original. **Não é a métrica anterior de remoções compostas e queries OOD:** mudam a população de cenas, a agregação e os horizontes. Não comparar diretamente estes percentuais aos 7,53%/17,13% anteriores.

Erro balanceado na avaliação já examinada:

| Janela | Pinhole estacionário | 3D passado estacionário | 3D passado + velocidade constante |
|---|---:|---:|---:|
| 0–1 s | 9,34% | 5,34% | 2,01% |
| 1–2 s | 18,78% | 14,39% | 5,73% |
| 2–4 s | 22,93% | 19,20% | 11,68% |
| 4–6,4 s | 24,70% | 19,92% | 18,18% |

O pinhole usa a caixa 2D/classe/calibração; os outros dois recebem **geometria 3D privilegiada**. A comparação entre os dois últimos isola a inclusão da velocidade com os mesmos dados passados. Portanto, há evidência de necessidade de movimento neste alvo temporal, mas não evidência de necessidade de uma VLA. A dificuldade cresce no longo prazo; isso tampouco prova que o erro restante seja visualmente previsível.

O controle adicional que congela a primeira caixa futura em t+0,1 s é um oráculo diagnóstico, explicitamente separado no JSON. Nenhum desses rótulos mede segurança física: ainda são distâncias de 2 m entre centro da trajetória sintética e caixas em futuros registrados não reativos, sem footprint do ego ou regras de trânsito.

## 2. Há informação visual incremental sobre mudanças de movimento?

[Protocolo escrito antes da execução](PROTOCOLO_INOVACAO_TEMPORAL_V1.md). Código: `src/cavla/temporal_innovation_probe.py`. Artefatos: `artifacts/temporal_innovation_v2_20261021/`.

Alvo: 64 centros futuros menos a extrapolação de velocidade constante. Todos os regressores recebem a mesma entrada numérica: caixa 2D, classe, centro 3D passado, velocidade passada, orientação e tamanho. Comparamos ridge linear e kernel RBF usando apenas esses dados, acrescidos de PCA16 de features VLA, ou acrescidos de 16 variáveis gaussianas como controle de capacidade. O backbone continua congelado e as features VLA são de **uma imagem frontal original**, não de vídeo temporal. A supervisão 3D é mais forte que a dos estudos anteriores.

Três splits por log reaproveitados; PCA e normalização ajustadas apenas no treino. A escolha entre hiperparâmetros e correção zero usa exclusivamente ADE médio por log na validação interna. São seis famílias de readout em três folds, com 189 ajustes de hiperparâmetros no total: 3 entradas × (6 ajustes lineares + 15 ajustes RBF) × 3 folds. Uma seleção é determinística; não há seeds independentes de modelos.

**As 18 seleções escolheram correção zero.** Os ajustes realmente foram calculados: na validação do fold 0, a base tem ADE de 1,552 m; o melhor ridge numérico tem 1,580 m e o melhor ridge com VLA tem 1,594 m. Os valores completos de todos os ajustes estão em `records.json`.

Consequentemente, todos os procedimentos selecionados coincidem com a base: ADE médio por log de **2,007 m em CV** e **2,753 m nos 152 agentes/16 logs da avaliação anterior**. Intervalos de diferença exatamente zero refletem a identidade dos preditores selecionados; não são prova estatística de equivalência entre classes de modelos. Não houve melhora incremental validada.

O resultado rejeita **esta sondagem** como justificativa para investir num método temporal com as features atuais. Não prova ausência de informação visual em vídeo, em outra camada, num modelo maior ou em mais dados. Também não determina se o problema foi amostra pequena, modalidade inadequada ou imprevisibilidade do futuro. Não mede decisões da política.

## 3. Novidade: novas fontes primárias examinadas

A leitura abaixo foi da página/resumo dos autores ou registro primário; não equivale a revisão completa dos métodos, código e resultados.

| Trabalho | Sobreposição que precisa ser evitada |
|---|---|
| [Counterfactual VLA, CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/html/Peng_Counterfactual_VLA_Self-Reflective_Vision-Language-Action_Model_with_Adaptive_Reasoning_CVPR_2026_paper.html) | Raciocínio contrafactual e mineração de cenas difíceis para corrigir ações já são uma proposta existente. |
| [CRAFT](https://arxiv.org/abs/2605.04470) | Combina proxy contrafactual denso com correção residual de interação em malha fechada; uma loss proxy + correção não basta como novidade. |
| [CounterScene](https://arxiv.org/abs/2603.21104) | Já modela dependências entre agentes e intervenções em agentes críticos para geração de cenários. |
| [SafeDriveVLA](https://arxiv.org/abs/2610.20609) | Conflito entre instrução e cena, modos de cumprimento e world model condicionado à instrução já estão cobertos. |
| [From Proxy Learning to Driving Decisions](https://arxiv.org/abs/2609.02688) | Já investiga quando ganhos em proxies deixam de transferir para seleção de trajetórias; apenas uma auditoria dessa falha não estabelece novidade aqui. |
| [Interventional Causal Representation Learning, ICML 2023](https://proceedings.mlr.press/v202/ahuja23a.html) | Intervenções imperfeitas já têm resultados sob hipóteses específicas. Artefatos de edição de pixels não satisfazem automaticamente essas hipóteses. |

O [Mi3 Lab](https://mi3-lab.github.io/) mantém significado de objetos/ações e decisões seguras alinhadas a objetivos como motivação. Isso favorece perguntas semânticas, mas alinhamento ao laboratório não substitui a demonstração de novidade.

## Decisão de pesquisa

Preservar a infraestrutura e o dataset local; não aprovar esta variante como candidato principal. Antes de novo treinamento de método, exigir uma tarefa em que a semântica possa mudar a decisão depois de controlar a cinemática e uma fonte verificável dessa supervisão. Vídeo passado e instruções com referente/tempo verificados são possibilidades de aquisição de evidência, **não candidatos ICML já estabelecidos**. Os pares de remoção podem auxiliar, mas não fornecem por si só ações corretas ou futuros reativos.

Próxima barreira concreta: verificar viabilidade e novidade de aprender representações úteis à decisão sob intervenções visuais imperfeitas, com controles que separem mudança semântica de efeito do editor. Não alegar identificação causal sem hipóteses testáveis, nem extrapolar um resultado de previsão para ganho de política. A literatura e os experimentos atuais ainda não resolveram essa barreira.

Validação do código: 68 testes passaram, incluindo quatro novos que verificam exclusão de anotações futuras, rejeição de histórico insuficiente/antigo, recuperação de sinal linear conhecido e preservação da opção cinemática quando a inovação é zero. Log: `artifacts/tests_temporal_20261021.log`. Nenhum ambiente ou código compartilhado do Alpamayo foi modificado.
