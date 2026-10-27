# Auditoria dos experimentos e do potencial ICML — 27/10/2026

Auditor experimental independente, solicitado pelo usuário. Revisei a versão congelada em `paper/reviews/codex_20261027/snapshot/main.tex`, protocolos, fontes e artefatos; consultei processos, logs e configurações de execução sem interromper nada. Executei apenas análises de CPU em `artifacts/audit_agent_20261027`. Este documento não é uma avaliação oficial do ICML.

## Avaliação principal

**Vale continuar; o trabalho passou de uma hipótese plausível para uma candidata de análise com uma primeira replicação held-out positiva. Ainda não há sustentação para equivalência distribucional geral, melhora de segurança ou explicação de todo o ganho do sistema FlashDrive.** O resultado held-out de Alpamayo1.5 já está disponível e é mais informativo que ampliar o inventário de modelos: o Euler10 passa as duas margens pré-especificadas nos 120 clipes da nova amostra. Isso precisa ser confirmado nos modelos restantes e distinguido da definição operacional de temperatura por uma única medida de dispersão.

Não seria adequado exigir uma nova loss apenas para transformar o trabalho em um paper de método. Um paper de análise pode ter potencial ICML se demonstrar um mecanismo não trivial, predições que generalizam e uma consequência relevante para como os métodos são avaliados. A presente evidência ainda não completa esses requisitos.

## Estado real das frentes

| Frente | Estado verificado na auditoria | Leitura científica |
|---|---|---|
| Alpamayo1.5, amostra original | concluído | núcleo exploratório; 57/60 clipes da tabela principal pertencem ao treino do modelo |
| Alpamayo-R1 e Alpamayo2, original | concluídos | replicação dentro da mesma família, mesmo conjunto de entradas |
| A1.5, nova amostra test alinhada | **120/120 concluídos**, manifest presente | primeira replicação no split explicitamente excluído do treino segundo o release |
| R1, nova amostra | rodando, cerca de 43/120 no primeiro retrato | não inferir resultado final com a amostra parcial |
| A2, nova amostra | previsto, sem execução concluída encontrada | necessário para cumprir a previsão de pelo menos 2/3 modelos |
| Qwen-Drive RL/SFT v2 | concluídos, análise presente | v1 inválido; v2 dá limites importantes da explicação por temperatura |
| SmolVLA/LIBERO | grade concluída | contração replica; efeito de sucesso da temperatura é fraco/inconclusivo na faixa central |
| FlashDrive NVIDIA | 60 clipes concluídos e analisados | efeito pontual consistente; equivalência do cache não estabelecida |
| FlashDrive z-lab otimizado | **60 clipes concluídos às 11:19**; manifest presente | fiz análise CPU independente abaixo; não era necessário esperar outra GPU para conhecê-la |
| AlpaSim | E10, E4 e H32/tau1 com 58 cenas cada; H32/tau0,84 iniciou durante auditoria | piloto incompleto; faltam os dois braços de comparação de temperatura |
| NAVSIM | download registra `ALL_DONE` | dados disponíveis; nenhuma avaliação ReCogDrive/DiffusionDrive concluída encontrada |

O estado muda durante a auditoria. Logs indicam progresso; não trate nomes de pastas como evidência de conclusão. O único número instantâneo de utilização da GPU observado foi 1%, com cerca de 66 GiB ocupados; isso não estima produtividade média, pois houve transição de configuração e outros jobs ativos. Nenhum job foi iniciado, pausado ou encerrado pelo auditor.

## 1. O maior problema de validade externa foi identificado e está sendo corrigido

O artefato `artifacts/split_overlap_v1_20261027/summary.json` confirma que **57 dos 60 clipes principais** são `train` em `alpamayo_aligned.parquet`; apenas três são `test`. O split `default` do dataset não é o split de treinamento do checkpoint. A versão congelada reconhece isso em `main.tex:159`, o que é correto. A comparação interna solver × escala do ruído não se torna automaticamente inválida: entradas e modelo são idênticos entre braços. Porém, generalização para situações não vistas pelo modelo exige a nova avaliação e o resultado principal deveria migrar para ela.

O protocolo adendos 13/13b define 120 clipes primários de uma seleção de 400, todos `test` do split alinhado, com margens de ADE1 ±0,05 m e flips ±2 pontos. A avaliação de 120 primários é o compromisso verificável; 400 arquivos de contexto não significam 400 clipes avaliados por modelo.

Verifiquei também novidade para o projeto por `clip_id`, evitando confundir ausência no treino com dados nunca examinados:

| Nova seleção | Interseção de clip_ids com v1(200) + v2(436) | Interseção de chunks |
|---|---:|---:|
| Primária, 120 | **0** | 120 |
| Seleção total, 400 | 2 | 400 |

Os dois clipes exatos reutilizados estão nos novos `context_133` e `context_183`, fora da amostra primária. Assim, **a confirmação primária usa clip_ids novos para o projeto**. O SDK `physical_ai_av/dataset.py` emprega chunks para localizar arquivos packed `.parquet`/`.zip`; não encontrei evidência de que um chunk corresponda a uma viagem ou unidade temporal independente. Portanto, não classifico compartilhar chunk como contaminação ou dependência temporal. Independência entre viagens/capturas continua não verificada. O texto “um por chunk” não basta para demonstrar independência estatística entre cenas.

Contagens reproduzíveis em `artifacts/audit_agent_20261027/overlap.json`.

## 2. Resultado novo: A1.5 passa os critérios primários no held-out para Euler10

Usei os planos completos de `temperature_probe_heldout_v3_20261027`, os 120 contextos correspondentes e as funções de métricas do próprio projeto. Fiz 1000 bootstraps por clipe, recalculando tau por dispersão em cada reamostragem. A análise é uma auditoria dos dados já concluídos, não um pré-registro novo; não alterei as tabelas do paper.

| Sampler | tau efetivo [IC90%] | Resíduo ADE1 [IC90%], m | Resíduo flips [IC90%], pontos | Equivalência nas duas métricas |
|---|---|---|---|---|
| Euler10 | 0,849 [0,836; 0,860] | +0,0075 [-0,0022; +0,0176] | -0,38 [-1,45; +0,75] | **sim** |
| Euler4 | 0,655 [0,636; 0,681] | +0,0286 [+0,0043; +0,0531] | +0,91 [-0,70; +2,68] | não estabelecida |

ADE1 do Euler10 é 1,791 m contra 1,966 m da referência tau1; na temperatura pareada, a referência dá 1,783 m. O padrão antigo replica em cenas não usadas na descoberta e no split excluído do treino segundo os metadados. **Este é um avanço substantivo para o candidato.**

Há dois limites:

- A aprovação do protocolo “em pelo menos 2/3 modelos” continua pendente de R1 e A2; não se deve contar gerações parciais.
- Heun3/5 no novo conjunto têm dispersão fora do extremo superior da grade de temperatura em algumas avaliações. A função atual usa `np.interp`, que trunca tau em 1. Uma leitura do resíduo como comparação com temperatura pareada é inválida fora da grade; é necessário marcar como não casado ou ampliar a grade. Meu artefato registra `in_temperature_grid` e a fração de bootstraps dentro da grade. Isso não afeta os resultados E4/E10 acima.

Fontes e resultados: `artifacts/audit_agent_20261027/heldout_a15_audit.py`, `heldout_a15_summary.json` e `heldout_a15_audit.log`.

## 3. Resultado novo: FlashDrive otimizado é consistente com o efeito, mas não prova equivalência

Executei `cavla.flashdrive_analysis` sobre a sonda z-lab já concluída, mantendo a unidade clipe e a média das três janelas efetivamente geradas pelo stack de streaming. Os quatro frames/janelas de entrada não viram quatro observações independentes.

- Cache: tau efetivo **0,650**, IC90% [0,627; 0,668].
- ADE1: **2,115 m**, contra 2,382 m do Heun32/tau1.
- Na temperatura pareada, ADE1 = 2,108 m; resíduo **+0,0076 m**, IC90% **[-0,042; +0,062]**. **Não estabelece equivalência** na margem ±0,05 m.
- Cache versus Euler8: **-0,110 m**, IC90% [-0,137; -0,082].
- Cache versus Euler4: **+0,0022 m**, IC90% [-0,042; +0,047].

O comportamento é coerente com o cache atuar como uma contração mais forte que Euler8 e semelhante a Euler4, mas a equivalência permanece imprecisa. O checkpoint NVIDIA já dava a mesma leitura: tau≈0,640, resíduo +0,0077 m, IC90% [-0,034; +0,051]. É uma reprodução do padrão em outro checkpoint/stack, **não uma replicação em cenas novas**: os 60 clipes são os mesmos.

O código do projeto implementa uma versão do cache dentro do stack oficial. Isso permite isolar o componente, mas não atribuir à temperatura todos os ganhos do sistema publicado, que também inclui checkpoint, streaming, quantização e raciocínio. O rascunho atual é mais cuidadoso sobre isso (`main.tex:191`).

Artefatos próprios: `artifacts/audit_agent_20261027/flashdrive_zlab/summary.json` e snapshot do analisador. Não calculei flips do FlashDrive, que seguem pendentes.

## 4. AlpaSim: piloto genuíno de ego fechado, ainda sem resultado da hipótese central

Os três primeiros braços têm 58 cenas com métricas. Os códigos de saída são 1 porque duas cenas de rota inválida se repetem nos três braços: waypoint inicial a 27,36 m da rota original e dobra de 180 graus. A repetição dos mesmos defeitos, com os mesmos valores, em três configurações apoia tratá-los como problema da rota/dados e usar as 58 cenas comuns, conforme adendo 2. O retorno não significa que os 58 rollouts foram perdidos; também não significa uma grade de 60 cenas concluída com sucesso.

As configurações reais continuam com `endpoints.trafficsim.skip=true`, `physics_update_mode=EGO_ONLY` e `force_determinism=false`. O código `services/traffic_service.py` confirma que skip reproduz agentes gravados. Existe feedback de observação e controle no ego; **não existe reação dos demais agentes**. Isso deve acompanhar qualquer conclusão de segurança. A ausência de seeds reprodutíveis por inferência não impede uma comparação pareada por cena, mas aumenta o ruído e enfraquece a atribuição de diferenças ao solver em um único rollout.

Extraí apenas as métricas brutas publicadas nos logs dos 58 rollouts concluídos:

| Métrica bruta | Euler10 | Euler4 | Heun32/tau1 |
|---|---:|---:|---:|
| Alguma colisão | 8/58 | 8/58 | 6/58 |
| Algum offroad | 18/58 | 16/58 | 18/58 |
| Colisão OU offroad | 23/58 | 22/58 | 22/58 |

**Estas não são as taxas primárias `collision_at_fault`/`offroad_or_collision_at_fault` após agregação oficial.** São um diagnóstico intermediário e não mostram uma melhora bruta evidente de Euler sobre Heun/tau1. Faltam os dois braços de temperatura para testar C1–C3; não é possível concluir agora que o fenômeno explica segurança. Também não se deve parar ou ampliar a grade com base nesses números intermediários.

Artefato: `artifacts/audit_agent_20261027/alpasim_interim_raw_metrics.json`. Não agreguei as cenas falhas como sucesso nem as excluí além da regra já declarada.

## 5. Qwen e robótica: limites, não confirmações irrestritas

O Qwen v1 consultava um intervalo de fluxo não treinado e foi corretamente invalidado. Não repetir a história antiga de que “o erro numérico acrescenta diversidade num fluxo determinístico”. Na v2, referência em [0; 0,9] com endpoint final, checagem Heun64, expert fp32 e modo reasoning, a dispersão de E10 fica próxima à referência e a de E4 encolhe.

O E10 mapeia a tau≈0,98. O E4 mapeia a tau≈0,42 (RL) e ≈0,37 (SFT), porém o ADE praticamente não melhora com reduzir tau. O E4 paga erro residual; a equivalência falha no ADE1 do RL e nos flips do SFT. Isso sustenta uma tese mais estreita: o controle de escala pode ser útil para a avaliação, mas a redução de passos não equivale universalmente a ele e não reproduz automaticamente qualidade/comportamento.

As seeds foram corrigidas para 42–47 e a diferença Heun32/64 cai para milímetros. Continua o comando de navegação derivado do heading futuro, comum aos braços; adequado para comparação interna condicionada, limitado para alegações de implantação. A exposição desses clipes ao treino do Qwen não foi verificada.

SmolVLA confirma contração fora de direção, mas a curva de sucesso entre tau0,5–1 é aproximadamente plana. Isso não confirma a explicação para ganhos de sucesso de Coda e não testa o corretor aprendido desse método. O nulo deve permanecer como limite e não ser removido do relatório.

## Onde gastar a próxima semana de GPU

1. **Concluir R1/A2 nos 120 held-out primários e os dois braços restantes do piloto AlpaSim.** São os testes já comprometidos que podem mudar a interpretação central. Não acrescentar novos modelos antes de colher e analisar esse conjunto.
2. **Verificar convergência da referência em uma subamostra pequena e pré-definida**, com diversidade de velocidades, incerteza e ruídos tau, em vez de chamar Heun32 de fluxo exato. O Qwen tem essa checagem; os novos Alpamayos precisam de evidência própria.
3. **Congelar uma comparação além do spread e taxas agregadas.** Separar calibração de tau da avaliação de modos/decisões condicionais; comparar temperatura e shrinkage simples com uma estatística de distribuição que tenha poder mensurável. Uma sonda com mais ruídos em um número menor de cenas pode ser mais informativa que mais gerações do mesmo modelo com seis ruídos.
4. **Só depois decidir entre closed-loop maior e uma família NAVSIM.** Se o piloto fechado tiver informação suficiente e sinal interpretável, planejar tamanho por discordâncias e efeito mínimo relevante, seeds por cena e tratamento de falhas. Se for inconclusivo, não presumir que aumentar indiscriminadamente N resolverá. Se a fragilidade principal continuar sendo generalidade, uma família fora de Alpamayo tem prioridade sobre repetir as três gerações.

Em paralelo e sem GPU: analisar o FlashDrive z-lab já concluído; corrigir referências teóricas, completar apêndice e métricas; documentar versões inválidas como inválidas; atualizar o cronograma antigo e o relatório histórico, que ainda misturam a tese anterior e os resultados corrigidos.

## Gates para uma decisão ICML

- **Gate empírico imediato:** pelo menos 2/3 modelos passam as margens pré-especificadas no held-out, com referência numericamente justificada. Hoje A1.5 passou; o resto está pendente.
- **Gate de identificação:** provar que o resultado útil excede “escolhemos o parâmetro para igualar dispersão” e revelar onde uma aproximação de escala falha em modos e decisões por contexto. Igualdade de duas médias não é igualdade de políticas.
- **Gate de impacto:** demonstrar um viés de avaliação relevante em uma aceleração efetiva ou uma consequência executada reproduzível, sem depender de eventos raros, exposição ao treino ou comparação de referências distintas. Não é obrigatório achar melhora de segurança; é obrigatório que a conclusão afirme apenas o que os dados identificam.
- **Gate de novidade:** mecanismo e protocolo devem ser diferenciados de teoria gaussiana clássica e controle de temperatura existente. Os revisores do paper fazem essa auditoria bibliográfica; o volume de GPU não substitui essa demonstração.

**Recomendação atual:** continuar com escopo concentrado. A confirmação held-out A1.5 e a conclusão do probe z-lab são avanços reais de hoje, mas o paper congelado ainda é exploratório e incompleto. O candidato pode amadurecer para ICML como estudo de mecanismo e avaliação; a evidência presente ainda não sustenta uma equivalência geral ou um ganho de segurança.
