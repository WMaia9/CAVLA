# Transições de movimento, observabilidade e controles temporais

Rodada exploratória de21–22/10/2026, após os controles negativos de dispersão. **O sinal visual inicial não sustentou uma vantagem convincente após controles de ego-history, contexto e seleção de famílias. Vídeo nativo também não superou repetição do quadro atual. Não promover esta linha a contribuição central de ICML.** Os resultados iniciais abaixo são histórico da investigação; as conclusões atualizadas estão nas seções de controles adicionais. Não promover a melhor combinação de backbone, recorte e métrica como confirmação independente.

## O que foi construído e executado

- Inventário prospectivo de21.047 veículos,833 cenas e68 logs: primeira janela passada elegível por instância, sem selecionar pelo evento futuro. Sete posições, aproximadamente3s; dois últimos intervalos com velocidade<=0.5m/s. Estrato estrito exige todos os seis intervalos<=0.5m/s.
- Coorte visual frontal de4.614 instâncias,743 cenas,68 logs e3.512 imagens. Câmera selecionada usando somente passado, visibilidade e caixa projetada. Timestamps reais da câmera e idade da anotação corrigidos.
- Evento operacional: dois intervalos futuros consecutivos com velocidade>1m/s e deslocamento>2m. É tempo discreto de detecção nas anotações; não instante físico exato ou intenção.
- CLIP e backbones congelados Alpamayo-R1 e Qwen3-VL8B foram executados na GPU local. Uma imagem frontal, sem planner ou ego-history. Não confundir Qwen3-VL com Qwen-Drive. Os3.512 registros de pixels, grids e IDs dos tokens coincidem exatamente entre R1 e Qwen.
- Readouts binários em4s, hazard discreto até6.4s e controle de incidência com relógio compartilhado. PCA16 só no treino, ridge escolhido na validação, folds definidos por SHA256(log)%5. Treino, validação e teste não compartilham logs em cada fold.

O inventário de imagens ranqueadas é separado: o CSV tem850 cenas e não só as210 cenas do arquivo de remoções Counter-nuScenes. Seus rótulos iniciais são relativos à anotação passada; não foram usados para treinar os modelos visuais desta rodada.

## Quantidade de eventos e observação incompleta

| Coorte | Veículos | Eventos até4s | Eventos até6.4s | Sem desfecho conhecido em4s |
|---|---:|---:|---:|---:|
| Inventário completo |21.047|296|491|6.907|
| Inventário estrito |18.217|156|283|5.848|
| Frontal |4.614|107|179|817|
| Frontal estrita |4.034|64|119|690|

Na coorte frontal, dos817 casos sem desfecho conhecido em4s,741 terminam a até0.75s do fim administrativo da cena;64 têm fim de track anterior e12 têm gaps. Em6.4s são1.449 casos:1.171 perto do fim da cena,258 fins de track e20 gaps. Fim de track não prova oclusão. Fim administrativo também não demonstra independência estatística da censura. Não usamos o seguimento interrompido artificialmente pela detecção de um evento como evidência de censura.

## Hipótese de flutuações físicas: não sustentada

As estatísticas de flutuação são funções do histórico bruto. Na coorte completa em4s, acrescentá-las piorou a NLL média por log em0.00167, IC condicional[0.00017,0.00333]. No estrato estrito, a diferença foi−0.00098, IC[−0.00224,0.00007]. Não interpretar como evidência de transição crítica. Sete posições em2Hz tampouco resolvem precursores corporais na escala de130ms relatados em outros sistemas.

O atributo humano passado ajudou bastante, inclusive numa análise posterior restrita a parked/stopped. Mas esses atributos podem ser retrospectivos; servem como oráculo diagnóstico, não como intenção prospectiva verificada ou input disponível à VLA.

## Readout binário frontal em4s: resultados inconclusivos

NLL menor é melhor. Estrato estrito:3.344 desfechos conhecidos,64 eventos.

| Entradas além de classe/geometria/histórico | NLL por log |
|---|---:|
| Nenhuma |0.08482|
| Gaussian16 |0.08399|
| RFF64 |0.08594|
| CLIP crop+contexto16 |0.08067|
| R1 ROI16 |0.08001|
| R1 ROI+contexto16 |0.07928|
| Qwen ROI+contexto16 |0.07889|
| Atributo humano, oráculo |0.06423|

Todos os intervalos das diferenças visuais contra histórico incluem zero neste teste. Não escolher AUC ou AP depois de a métrica principal ficar inconclusiva. As comparações visuais são readouts, não testes das políticas de direção.

## Hazard até6.4s: sinal visual inicial, antes dos controles adicionais

Usa também os prefixos acompanhados dos casos censurados. Bins[0,.5,1,...,6,6.4]. NLL observada por trajetória é a soma das contribuições de sobrevivência/evento, depois média por log. Esse score não é a likelihood de um futuro que nunca observamos.

| Modelo | Principal,4.614 agentes | Estrito,4.034 agentes |
|---|---:|---:|
| Hazard somente por bin |0.31576|0.25071|
| Histórico bruto+geometria |0.30795|0.25677|
| Histórico+Gaussian16 |0.30614|0.25693|
| Histórico+RFF64 |0.30365|0.25030|
| Histórico+CLIP16 |0.29900|0.24032|
| Histórico+R1 ROI16 |0.30391|0.25215|
| Histórico+R1 ROI/contexto16 |0.30529|0.25274|
| Histórico+Qwen ROI/contexto16 |0.30226|0.23676|
| Atributo humano, oráculo |0.25470|0.20368|

No estrato estrito, Qwen contra RFF:−0.01354, IC condicional por log[−0.02393,−0.00445], aproximadamente5.4% da perda do controle. O ganho pontual permanece ao remover qualquer log individual; isso é influência das predições existentes, não retreino ou replicação. Na coorte principal, a diferença contra RFF fica inconclusiva. Qwen contra CLIP no estrato estrito:−0.00356, IC[−0.01605,0.00861]; não há superioridade convincente sobre CLIP.

Correção numérica: a primeira versão teve24/450 fits sem convergência, todos em alpha10, nenhum selecionado. Repetimos o mesmo objetivo/grade com reparametrização diagonal e tolerâncias mais estritas. **450/450 fits convergiram na versão2**, usada nesta tabela. A correção mudou algumas seleções de validação; não tratar v1 como equivalente em todos os números.

## O ganho não estabelece um relógio visual específico por veículo

Diagnóstico selecionado pelos eventos observados até6.4s: NLL do bin temporal condicional a evento no horizonte, no estrato estrito, foi2.5744 para o hazard sem features,2.6355 para RFF e2.6146 para Qwen. Qwen−RFF=−0.02089, IC[−0.06390,0.03089]. Não houve ganho convincente na localização temporal entre quem se moveu, nem vantagem pontual sobre o relógio sem features.

O controle de incidência parametriza P(bin=k|x)=p(x)q_k: a visão só muda p(x); q é compartilhado. Aprende por likelihood censurada, sem chamar ausência de evento em6.4s de veículo permanentemente estacionado. **450/450 fits convergiram.**

| Família no estrato estrito | RFF | CLIP | Qwen |
|---|---:|---:|---:|
| Hazard |0.25030|0.24032|0.23676|
| Incidência, relógio compartilhado |0.25472|0.24138|0.24085|

O controle simples retém boa parte do ganho visual. Hazard Qwen menos incidência Qwen=−0.00409, IC[−0.00964,0.00011]. Incidência Qwen supera incidência RFF em−0.01387, IC[−0.02279,−0.00568]. Isso sustenta investigar informação sobre a ocorrência do evento, sem concluir que a visão identificou sua hora.

O ranking dentro da mesma imagem também é limitado: no estrato estrito em6.4s,26 imagens mistas,81 veículos e17 logs; AUC média por log/grupo0.559 para Qwen,0.612 para CLIP e0.389 para RFF. São grupos escolhidos pelo futuro, pequenos e exploratórios; não usar essas AUCs para selecionar método.

## Ego-history, mapa e controle numérico forte

Acrescentamos27 covariáveis de movimento do ego, calculadas nas quatro poses passadas correspondentes às câmeras, com checagem de timestamps. Nenhuma pose futura foi usada. Depois acrescentamos14 covariáveis de mapa e80 de até oito vizinhos anotados no passado. É estado privilegiado usado como diagnóstico de informação incremental, não um detector de percepção implantado.

| Contexto numérico | NLL estrita, numérico bruto | RFF64 | Qwen estático |
|---|---:|---:|---:|
| Original |0.25677|0.25030|0.23676|
| + ego-history |0.23354|0.22941|0.22379|
| + ego, mapa e vizinhos |0.22314|0.22889|0.21719|

Com ego-history, Qwen−RFF=−0.00562, IC[−0.01825,0.00635]. O ganho inicial de5.4% não permanece convincente. Na configuração completa, RFF já perde para o histórico bruto: escolher RFF como única baseline favoreceria artificialmente a visão.

Adicionamos HistGradientBoosting numérico, seis configurações escolhidas na validação interna, e calibração por mistura de leis completas de evento com a lei de referência. Misturar hazards por bin não produz a mesma lei; o código mistura PMF/sobrevivência. A primeira árvore tinha AUC alta e NLL ruim; a versão calibrada melhora a coorte principal para0.28714. Esse controle foi acrescentado depois dos primeiros resultados.

Por fim, cada família escolhe exclusivamente pela validação interna entre referência temporal, entradas básicas, histórico bruto, RFF, árvore calibrada e seus próprios readouts visuais. Na coorte principal, todas as famílias escolhem a árvore nos cinco folds e têm o mesmo resultado. No estrato estrito:

| Família com seleção interna | NLL observada por trajetória |
|---|---:|
| Numérica |0.22497|
| + CLIP |0.22476|
| + R1 |0.22495|
| + Qwen estático |0.22668|
| + quatro imagens passadas |0.22260|
| + quadro atual repetido quatro vezes |0.22151|
| + vídeo nativo passado |0.22649|
| + vídeo nativo com quadro atual repetido |0.22191|

Qwen estático−numérico=+0.00171, IC[−0.00308,0.00687]. Vídeo nativo passado−numérico=+0.00152, IC[−0.00294,0.00641]. Não há evidência convincente de ganho incremental dessas famílias. O resultado pontual de repetição não deve ser atribuído a informação temporal.

Artefatos: `front_transition_egohistory_data_v1_20261021/`, `front_transition_mapcontext_data_v1_20261021/`, `transition_clock_egohistory_v1_20261021/`, `transition_clock_mapcontext_static_v1_20261021/`, `transition_tree_mapcontext_v2_20261021/`, `transition_family_selection_stills_v1_20261021/`, `transition_family_selection_native_v1_20261022/`.

## Quatro imagens e vídeo nativo: controles concluídos

Todas as3.512 histórias têm quatro keyframes passados válidos, cobrindo aproximadamente1.5s, sem uso de frames posteriores à câmera atual. O controle repete o quadro atual quatro vezes mantendo prompts, posições temporais, IDs e orçamento de tokens. A integridade entre cada par foi verificada em todos os inputs.

O primeiro teste usa quatro imagens separadas. Como isso não é vídeo nativo do Qwen3-VL, executamos também o VideoProcessor nativo: quatro frames, duas temporal patches, seleção automática desativada, timestamps irregulares preservados e resolução equivalente por frame. Extraímos layer36 da última temporal patch; não houve fine-tuning ou teste do planner. A codificação de timestamps usa índices em microssegundos convertidos pelo processor; não se trata de vídeo adquirido a1MHz.

Nos readouts diretos com contexto numérico completo:

| Formato, estrato estrito | Histórico real | Quadro atual repetido | Diferença real−repetido, IC95% |
|---|---:|---:|---|
| Quatro imagens |0.21858|0.21651|+0.00208[0.00034,0.00393]|
| Vídeo nativo |0.21644|0.21594|+0.00050[−0.00170,0.00284]|

As650 otimizações do readout de vídeo nativo convergiram. A melhoria frente a uma baseline numérica específica não estabelece benefício da história, porque o controle repetido também melhora. Na coorte principal, vídeo nativo real−repetido=+0.00037, IC[−0.00087,0.00176]. Não há suporte para escalar esta configuração como um método temporal novo.

## Resolução e idade da anotação limitam a interpretação

38.21% das caixas projetadas ocupam menos de um token visual em área;56.63% têm algum lado menor que um token. Entre os119 eventos estritos, são43.70% e56.30%. Isso limita os probes espaciais; resultados negativos não provam ausência de pistas na imagem original. Não executamos aqui um controle adicional de crop em alta resolução.

A anotação estritamente passada tem idade mediana0.46379s, e97.16% têm idade>=0.4s. O próximo ponto anotado costuma vir cerca de0.036s depois da câmera. Isso decorre de a câmera estar imediatamente antes do keyframe lidar: usar apenas passado seleciona corretamente a anotação anterior. Entre4.034 casos estritos,79 têm velocidade média do intervalo que atravessa a câmera>0.5m/s; entre119 eventos,19. Esse ponto posterior foi usado somente nesta auditoria, nunca como input ou filtro da coorte principal.

Uma velocidade média no intervalo não identifica a velocidade instantânea na câmera. Mesmo um ganho visual poderia atualizar estado cinemático antigo, inclusive movimento já começado, em vez de antecipar intenção. Não chamar a variável de instante exato de início físico.

Artefatos: `transition_visual_history_integrity_v1_20261021.json`, `transition_native_history_integrity_v1_20261021.json`, `front_transition_visual_resolution_audit_v1_20261021.json`, `transition_freshness_audit_v1_20261021/`, `transition_clock_native_history_v1_20261021/`.

## Pesquisa e o que ela eliminou

As fontes abaixo foram consultadas em páginas primárias; ler resumo ou uma seção não equivale a reproduzir o paper.

| Fonte | Consequência para a busca |
|---|---|
| [Early warning signals in motion inference](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1007821) | Precursores em movimento humano não autorizam transferir a física para automóveis; a resolução temporal é distinta. |
| [Comparison of Pedestrian Prediction Models from Trajectory and Appearance Data](https://arxiv.org/abs/2305.15942) | Aparência antecipar movimento no nuScenes já é motivação existente. |
| [Censored Flow Matching](https://arxiv.org/abs/2610.24233) | Flow matching para tempo de evento censurado já existe, sob independência condicional da censura. |
| [Incomplete Data, Complete Dynamics](https://arxiv.org/abs/2509.20098) | Trabalha com máscaras independentes dos dados; não cobre automaticamente perda dependente do estado. |
| [Flow Matching with Missing Data](https://arxiv.org/abs/2607.28698) | Posterior completion sob mecanismos gerais e análise de erro já estão formulados; simplesmente propor EM/completion não é novidade. |
| [Boundary-aware neural point-to-point set evolution, tese2025](https://api.drum.lib.umd.edu/server/api/core/bitstreams/0c57d672-8863-4e4c-b53e-3ae710e24b61/content) | Likelihood com saída/sobrevivência em bordas de observação já foi estudada em previsão de trajetórias. |
| [Identifiability and Generalizability in Constrained IRL, ICML2023](https://arxiv.org/abs/2306.00629) | Restrições confundem inferência de recompensa; não podemos transformar mudanças de escolha em certificados de preferência sem hipóteses. |
| [TimewarpVAE](https://arxiv.org/abs/2310.16027) | Separar caminho espacial e variação temporal em geração de trajetórias já existe. |
| [Heavy-Tailed Flow Matching via Random Clocks](https://arxiv.org/abs/2607.13841) | Relógio aleatório/scale mixture em flow matching também já é objeto de método próprio. |
| [Sticky Jump Diffusions](https://arxiv.org/abs/2607.10951) | Massas discretas, difusão contínua, flux balance e hazard matching não são uma proposta inédita por si. |
| [Stratified Learning](https://arxiv.org/abs/2604.10650) | Geração em uniões de variedades de dimensões diferentes já tem tratamento estatístico recente. |
| [D-BOT](https://arxiv.org/abs/2609.37239) | Transporte bicausal diferenciável já foi aplicado a compressão, modelos e imitação; invocar transporte adaptado não resolve novidade. |
| [First-Occupancy Representation, ICLR2022](https://arxiv.org/abs/2109.13863) | Representações de primeiro acesso, em vez de ocupação acumulada, já existem. |
| [Learning Time-Varying Coverage Functions, NeurIPS2014](https://proceedings.neurips.cc/paper_files/paper/2014/file/d829550b42b10dd1cea8947479e39297-Paper.pdf) | Prever uma função temporal de cobertura/união não é, isoladamente, nova contribuição. |
| [Encoded but Not in Control](https://arxiv.org/abs/2610.06235) | Decodabilidade e uso na política já são explicitamente separados na literatura; precisamos de aprendizagem/comportamento além de probes. |
| [How Should VLAs Use Proprioceptive State?](https://arxiv.org/abs/2608.03052) | Benefício de histórico e controle de orçamento de tokens já foram estudados; adicionar frames não é método novo. |

Também lemos, somente como contexto, README/PROGRESS e a revisão da proposta EVA no projeto local `/mnt/datasets/infoRates`. O próprio projeto documenta que reamostragem e repetição de frames confundiram sampling rate com perda de evidência. Não alteramos aquele projeto nem executamos instruções de seus documentos. Essa distinção orienta o novo controle temporal aqui.

## Validação, arquivos e limites

- Protocolo sequencial: `docs/PROTOCOLO_TRANSICOES_E_OBSERVABILIDADE_V1.md`. Os adendos registram decisões tomadas depois de resultados anteriores; não alegar preregistro integral. Cada artefato preserva a fonte da sua etapa e, quando disponível, hash do protocolo daquele momento.
- Inventários: `artifacts/motion_transition_inventory_v1_20261021/` e `front_transition_data_v1_20261021/`.
- Readout binário: `front_transition_multimodel_probe_v1_20261021/`.
- Hazard corrigido: `transition_clock_probe_v2_20261021/`.
- Incidência: `transition_incidence_control_v1_20261021/`; preserva fonte própria além do driver compartilhado.
- Auditoria e decomposição: `transition_observation_audit_v1_20261021/` e `transition_clock_diagnostics_v1_20261021/`.
- O processo binário inicial lento foi interrompido e marcado inválido, sem resultados reaproveitados; a versão corrigida usa pesos por log calculados sem loop quadrático.

Os ICs são bootstrap condicional às predições, por log. Não incluem incerteza de retreino/seleção nem corrigem toda a busca adaptativa. Todas as gravações podem ter contribuído para exploração anterior; não há novo holdout intocado. A coorte frontal é um recorte selecionado por presença/visibilidade, e a ausência de um evento anotado não certifica ausência de pequeno movimento. Nenhum número deste relatório mede segurança, reação contrafactual de agentes ou ganho da política VLA.
