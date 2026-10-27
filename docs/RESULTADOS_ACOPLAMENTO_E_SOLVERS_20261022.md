# Rodada de22/10: continuidade estocástica e erro numérico que chega ao controle

**Status: nova candidata de método em formulação, sem originalidade ou ganho fechado de direção demonstrados.** A descoberta operacional desta rodada foi localizar e usar a instalação CARLA completa em `/home/perceptron-lab/CARLA_0.9.16`, além da integração Carlamayo. A busca anterior apenas em `/mnt/datasets/carla` não justificava concluir que o simulador completo estava indisponível.

## Continuidade com distribuição condicional preservada

Executamos controle lateral linear-gaussiano com política congelada a=Kx+σz, seis controles iniciais, quatro intensidades de ruído de processo, covariâncias populacionais exatas e cinco simulações de50.000 trajetórias por condição. Uma extensão acrescentou a ação anterior à informação condicionante. O protocolo e os adendos estão em `PROTOCOLO_ACOPLAMENTO_POLITICA_V1.md`.

No ruído de processo0.02, usar o resíduo da ação anterior dado apenas posição/velocidade reduziu em87.3% a variância da diferença entre ações, preservando covariâncias instantâneas de estado/ação até precisão numérica. Ruído AR também suavizou, mas alterou a lei condicional e a dinâmica: a probabilidade simulada de primeira saída foi0.40434, contra0.005736 com ruído iid. Isso é um sistema artificial, não evidência de risco de uma VLA.

O controle mais forte foi decisivo: quando a ação anterior é observada e incluída no condicionamento, o ganho gratuito de suavização desaparece exatamente. Preservar a lei da ação atual dada essa informação fixa também o custo esperado entre ações. Ignorar informação disponível para produzir ganho seria inválido. Inovações gaussianas e transporte condicional são técnicas clássicas; esta construção não foi promovida como contribuição original.

Referências diretamente relevantes: [Autoregressive Policies,2019](https://arxiv.org/abs/1903.11524), [RNR-DP,2025](https://arxiv.org/abs/2502.12724), [Temporal Policy,2026](https://arxiv.org/abs/2607.29482), [Reactive Real-Time Flow Policies,2026](https://arxiv.org/abs/2609.36540). A última já estuda diferenças de distribuição na execução assíncrona. Uma reformulação genérica desse problema seria sobreposição forte.

Artefatos: `conditional_policy_coupling_v1_20261022/` e `conditional_policy_coupling_v2_20261022/`. A igualdade de marginais em cada instante não garante igualdade de primeira saída ou da lei condicionada ao histórico completo.

## CARLA e inputs completos

Iniciamos somente uma instância nossa, nas portas2100–2102. Não usamos um servidor de outro processo. Coletamos duas janelas da mesma rota Town10HD_Opt, com controlador simples de seguimento de waypoint a5m/s e nenhum NPC nesta coleta. Não é closed-loop com a VLA.

Cada janela tem quatro câmeras com FOV/ordem[0,1,2,6], quatro frames reais por câmera a10Hz e16 poses passadas. Resolução1920×1080 antes do processor. Todas as câmeras foram alinhadas ao frame exato da simulação, com timestamps conferidos. A conversão CARLA canhota→coordenadas destras foi implementada no CAVLA e testada para headings rotacionados; não copiamos uma conversão sem checar convenções. As duas janelas têm velocidades4.251 e4.058m/s. Nenhuma câmera foi preenchida com zeros.

As tentativas de coleta v1/v2 ficaram inválidas e não forneceram inputs aproveitados. A v2 revelou ausência de um frame com sensor_tick nominal0.1; usar sensor_tick0, emissão a cada tick síncrono, resolveu na v3. A instância isolada foi encerrada após a coleta; os modelos compartilhados e ambientes externos foram importados em leitura.

Artefato: `carla_native_contexts_v3_20261022/`. São duas janelas correlacionadas, não duas cenas independentes nem uma avaliação de generalização.

## Alpamayo1.5 congelado:96 rollouts de solver

Executamos o checkpoint local completo, sem fine-tuning ou quantização adicional. Cada janela tem a mesma amostra de linguagem(seed123) e ruído inicial do expert fixado entre solvers. Os96 rollouts reutilizam esses contextos; não são96 observações independentes. O replay da configuração oficial Euler10 coincidiu exatamente com o decoder separado na v2 válida. A primeira tentativa, que usava dt em Python em vez das diferenças exatas da grade Torch, parou por desvio de0.000642m e foi marcada inválida.

Primeiro: Euler4/8/10/20/40/80 e Heun4/10/20/40, duas seeds de ação. Depois, na seed9202 escolhida por diagnóstico, estendemos Euler até320 e Heun até160, emBF16 e com expert/projeções/cache emFP32. O VLM permaneceu congelado emBF16. A promoção muda o campo implementado, não apenas o solver.

Checagem de convergência, desvio médio XY em metros:

| Checagem | Janela0 | Janela1 |
|---|---:|---:|
| Heun80 vs160, BF16 |0.000629|0.04224|
| Heun80 vs160, FP32 |0.000623|0.01514|
| BF16 vsFP32, Heun160 |0.001833|0.006396|

Heun160 é uma referência numérica aproximada, com erro remanescente; não é ação correta ou prova de segurança. O caso difícil não desaparece emFP32. Não usar Euler80 como se fosse solução convergida: no caso1, Euler160→320 ainda muda0.432m.

## Controles clássicos explicam parte importante do fenômeno

Desvio médio da trajetória XY em relação aHeun160 BF16, na seed diagnóstica9202:

| Método | Avaliações do campo | Janela0 | Janela1 |
|---|---:|---:|---:|
| Euler4 |4|1.1468|8.6652|
| Euler8 |8|0.5862|6.1149|
| Euler10 |10|0.4850|4.4988|
| Heun5 |10|0.3225|6.5242|
| Euler, grade inicial densa,4 |4|1.0469|2.6449|
| Euler, grade inicial densa,10 |10|0.4420|2.6719|
| RK4,2 passos |8|0.1318|6.4984|
| RK4,5 passos |20|0.0039|1.0766|
| RK4,10 passos |40|0.0063|0.2082|

O melhor solver não é universal nesse recorte. Tampouco os dados refutam ProbeFlow: o cosseno inicial é0.9924 na janela0 e0.7843 na1, já sinalizando a janela difícil. Não afirmar que cosseno falha com base nestes dois casos. Grande parte do desvio está no fim da trajetória: no Euler10 da janela1, a média até1s é0.0929m, contra4.4988m no horizonte completo. Isso não limita jerk ou erro do controlador por si só.

Figura: `artifacts/native_solver_report_v1_20261022/classical_controls.png` ePDF. Tempos dos jobs de convergência concorrentes não são benchmark de latência; os próprios loops ainda incluem cópias diagnósticas para CPU. Nenhum speedup end-to-end foi estabelecido.

## A diferença alcança o controlador, mas a referência precisa ser declarada

Executamos em leitura o seguidor PID da integração, sem servidor, com a mesma velocidade e estado de PID explicitado para cada trajetória. O teste usa buffers de erro inicialmente vazios e três valores hipotéticos de steering anterior. Isso é diagnóstico posterior, não rollout de veículo. Na tabela, steering anterior0:

| Janela | Solver | Steering | Acelerador | Freio |
|---|---|---:|---:|---:|
|0|Euler4|−0.0938|0|0.1898|
|0|Euler10|−0.0884|0|0.5362|
|0|Heun160|−0.0833|0|0.7768|
|1|Euler4|0.1|0.6|0|
|1|Euler10|0.1|0.6|0|
|1|Euler inicial denso10|0.1|0|0.6034|
|1|Heun160|0.1|0|1|

Assim, trocar a integração pode mudar acelerar para frear nesse input. **Não sabemos por isso qual comando é correto.** Se o objetivo for preservar a política publicada, Euler10 é a referência comportamental. Se for estudar o fluxo contínuo, é preciso uma referência convergida e outra pergunta. Não alternar entre essas referências para favorecer um método.

Há ainda um problema para um certificado: Euler4 eEuler10 da janela1 têm o mesmo comando saturado, mas velocidades desejadas5.603 e4.611m/s, respectivamente. Os erros inseridos no buffer do PID diferem. Igualdade da ação de um frame não garante igualdade do controlador no futuro. O seletor de amostras com memória do plano anterior também pode reintroduzir dependência do horizonte completo. Essa é a dificuldade específica que a nova candidata precisa enfrentar.

Artefato: `native_controller_gate_v1_20261022/`. Nenhum comando foi aplicado ao veículo neste diagnóstico.

## Literatura e decisão da rodada

- [ProbeFlow](https://arxiv.org/abs/2603.17850): roteia chunks para schedules por consistência do campo; a versão atual explicita custo de avaliações e trade-offs. É baseline obrigatória, não ideia nossa.
- [FlowCast](https://arxiv.org/abs/2602.01329): previsão/verificação de velocidades com controle de desvio já existe.
- [FlashDrive](https://arxiv.org/abs/2608.12932): já acelera o próprio Alpamayo1.5, inclusive usando caching de passos. Trocar apenas schedule ou modelo não é suficiente.
- [Realtime-VLA FLASH](https://arxiv.org/abs/2605.13778): drafting e verificação de ações contínuas com fallback já existem.
- [Score and Distribution Matching Policy](https://arxiv.org/abs/2412.09265): destilar distribuições de ação em poucos passos também não é contribuição inédita.
- [Q-learning with Adjoint Matching](https://arxiv.org/abs/2601.14234): adjuntos em políticas flow já são usados para otimização por valor; não confundir com análise de erro numérico.
- [Goal-oriented model-error estimation](https://ir.cwi.nl/pub/12312/): ponderação de resíduos por objetivos tem uma história anterior em engenharia numérica.

**Não treinar uma rede nova apenas para ganhar de Euler uniforme.** A aposta de método é aprender limites do erro na decisão e na memória do controlador a partir de integrações parciais, tratando explicitamente as mudanças de ramo do programa de controle. Está em `CANDIDATO_INFERENCIA_COM_ESTADO_DO_CONTROLADOR.md`. Ela exige revisão dirigida de novidade, um estimador que custe menos que resolver melhor aODE, testes em novas cenas e ganho closed-loop com latência cobrada. Nada disso foi validado ainda.

Validação atual:117 testes de código passam. Isso verifica implementações e identidades testadas, não originalidade nem aceitação emICML.
