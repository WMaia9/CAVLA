# Revisão com dois revisores e um auditor — 27/10/2026

Solicitada pelo usuário. Dois agentes emitiram pareceres separados sobre o mesmo snapshot; um terceiro auditou experimentos e fez análises adicionais exclusivamente em CPU. O coordenador conferiu os três documentos, os artefatos novos e as fontes primárias centrais. São revisões internas simuladas, não avaliações oficiais nem estimativas de probabilidade de aceite.

## Versão e relatórios

- Snapshot: `snapshot/main.tex` e `snapshot/main.pdf`, copiados antes da revisão. SHA256 do TeX: `47c930ee22a667cdeec6e60054fa0bfb12babf5aa621be3367f8d4366eaf7f97`. Manifest em `snapshot_manifest.json`.
- Revisor 1, rigor matemático/estatístico: `reviewer_1_rigor.md`.
- Revisor 2, novidade/significado: `reviewer_2_novidade.md`.
- Auditor, experimentos e investimento: `auditoria_experimentos.md`.

Os novos resultados abaixo são posteriores ao snapshot e não foram tratados como resultados já presentes no PDF. Nenhum job, código de inferência, protocolo ou texto do paper foi alterado nesta revisão. Os únicos arquivos novos são o snapshot, os pareceres e as análises CPU próprias.

## Conclusão do coordenador

**Continuar é justificado. A versão congelada ainda não alcança a régua de aceite do ICML.** O resultado novo de Alpamayo1.5 fora da partição de treino fortalece materialmente a hipótese para Euler10. O impedimento deixou de ser apenas falta de experimento: agora é necessário delimitar uma contribuição que exceda contração de diversidade e efeitos médios conhecidos, corrigir a teoria e demonstrar onde a explicação funciona ou falha.

Há potencial como paper de mecanismo e avaliação. Não é necessário inventar outra loss. Uma investigação que revela um viés relevante de avaliação pode ser uma contribuição forte, desde que seja nova, identificável, reproduzível e tenha consequência concreta. Nem um teste positivo adicional garante esse pacote, nem a ausência de melhoria de segurança por si só elimina um paper de análise.

## Onde os pareceres concordam

| Tema | Leitura conjunta |
|---|---|
| Relevância | Separar distribuição aprendida da distribuição efetivamente produzida pelo sampler é uma pergunta útil. |
| Evidência | Há regularidade de contração e equivalência de alguns funcionais, especialmente Euler10; generalidade e distribuição inteira não estão demonstradas. |
| Identificação | Igualar dispersão e obter taxas médias parecidas não identifica temperatura de maneira única. O controle de contração linear também explica parte dos resultados. |
| Teoria | A base gaussiana é parcialmente clássica; a ponte para rede aprendida e decoder não linear precisa de limites explícitos. A proposição universal de RK está incorreta. |
| Novidade | Existe espaço específico em discretização contínua de políticas, mas a narrativa genérica de temperatura escondida e avaliação enganosa já tem antecedente importante. |
| Próximo investimento | Concluir os testes comprometidos e os controles decisivos antes de ampliar indiscriminadamente famílias e cenários. |

Os focos são complementares: o revisor 1 prioriza validade e sustentação; o revisor 2 considera novidade/identificação o maior obstáculo; o auditor dá mais peso à confirmação nova que ainda não entrou no PDF. As recomendações sobre a versão congelada não devem ser confundidas com uma previsão de rejeição de uma futura versão corrigida.

## Avanços verificados durante a revisão

### A1.5 em 120 clipes fora do treino

Os 120 clipes primários são `test` no split alinhado do Alpamayo e não repetem clip_ids das amostras anteriores. O antigo conjunto principal contém 57/60 clipes `train`; o paper declara isso e a confirmação corrige essa limitação. Compartilhar shards de armazenamento não demonstra, sozinho, dependência temporal.

Análise do auditor com 1000 bootstraps por clipe, tau recalculado em cada um:

| Sampler | Tau efetivo | Resíduo ADE1, IC90%, m | Resíduo de inversões, IC90%, pontos | Resultado nas margens fixadas |
|---|---:|---|---|---|
| Euler10 | 0,849 | +0,0075 [-0,0022; +0,0176] | -0,38 [-1,45; +0,75] | **Passa ambos** |
| Euler4 | 0,655 | +0,0286 [+0,0043; +0,0531] | +0,91 [-0,70; +2,68] | Não estabelece equivalência |

Isso é uma confirmação importante e delimitada, não equivalência geral de políticas. A previsão de passar em pelo menos dois dos três modelos ainda depende de R1/A2. No novo conjunto, o sinal do efeito de minADE6 do Euler10 é negativo, diferentemente da piora exploratória usada na narrativa; a troca ADE1 versus best-of-six não deve ser tratada como lei universal.

Artefato: `artifacts/audit_agent_20261027/heldout_a15_summary.json`.

### FlashDrive z-lab concluído

O auditor analisou os planos já completos, sem nova inferência: tau do cache 0,650; resíduo ADE1 +0,0076 m, IC90% [-0,042; +0,062]. O padrão central é compatível com a hipótese, mas a margem de equivalência ±0,05 m não é satisfeita. O ganho sobre Euler8 é aproximadamente 0,110 m; a diferença contra Euler4 é aproximadamente 0,002 m.

Esse resultado ajuda a decompor o efeito do componente de cache no stack, mas não explica todo o sistema publicado nem apaga seu mérito de latência. Decisões e controles de distribuição continuam pendentes.

Artefato: `artifacts/audit_agent_20261027/flashdrive_zlab/summary.json`.

### AlpaSim e Qwen

AlpaSim tem 58 cenas com métricas nos três primeiros braços; faltam os dois braços de temperatura para testar a hipótese. Os agentes de tráfego seguem gravações, embora o ego receba feedback e replaneje. Não há conclusão de segurança. As duas rotas inválidas repetidas entre configurações precisam continuar registradas.

Qwen v1 foi invalidado e não deve sustentar a narrativa antiga. Qwen v2 corrigido fornece limites da explicação: reduzir dispersão quase não melhora ADE e o Euler4 mantém resíduos relevantes. Os resultados novos e os históricos precisam permanecer claramente separados.

## Quatro prioridades que mudam a avaliação

1. **Fechar a contribuição e a bibliografia, agora, sem GPU.** Confrontar Zheng et al., ICLR2025, que já mostra erro numérico reduzindo temperatura/diversidade e enviesando métricas em difusão discreta; e TSR, ICML2026, sobre controle de temperatura em fluxos. Nosso mecanismo pode ser distinto, mas a diferença precisa ser demonstrada, não apenas declarada por ser AV.
2. **Corrigir teoria e análise, agora.** Restringir ou provar adequadamente a proposição de RK; separar fluxo exato, rede e trajetória decodificada; não converter matching de taxas em igualdade de distribuição. Marcar comparações fora da grade de tau — Heun3/5 no novo conjunto são truncados por `np.interp` e não estão verdadeiramente pareados. Reajustar controles calibrados no bootstrap ou calibrá-los em dados separados.
3. **Concluir os compromissos de GPU existentes.** R1/A2 nos 120 clipes, os braços restantes do piloto AlpaSim e checagens pequenas de convergência da referência nos contextos/taus relevantes. Não mudar endpoints para selecionar somente o Euler10 positivo.
4. **Escolher um teste que separe explicações concorrentes.** Congelar uma medida não usada no ajuste da dispersão: probabilidades condicionais de decisões, estrutura de modos, uma distância entre distribuições com poder adequado ou um resíduo quantitativamente previsto pela anisotropia. Se contração linear reproduz tudo que medimos, a conclusão deve ser sobre controle de contração na avaliação, não sobre identificação exclusiva de temperatura.

Uma GPU-semana útil resolve uma dessas dúvidas; muitas tabelas adicionais sem critério de decisão não substituem esse trabalho. Depois dos testes acima, decidir entre ampliar circuito fechado e avaliar uma família NAVSIM conforme a lacuna que permanecer.

## Fontes primárias centrais

- Zheng et al., ICLR2025, [Masked Diffusion Models are Secretly Time-Agnostic Masked Models and Exploit Inaccurate Categorical Sampling](https://arxiv.org/abs/2409.02908): antecedente conceitual próximo, mecanismo discreto diferente.
- Xu et al., ICML2026, [Temporal Score Rescaling for Temperature Sampling in Diffusion and Flow Models](https://arxiv.org/abs/2510.01184): controle de temperatura e distinção de diversidade, inclusive robótica.
- Claustre et al., [Gaussian Flow-Matching Schedules](https://arxiv.org/abs/2609.25839): base gaussiana que deve ser diferenciada da contribuição nova.

Os pareceres individuais contêm evidências, referências de código e perguntas completas. A recomendação conjunta é **prosseguir com escopo concentrado e critérios explícitos**, incorporando a confirmação positiva sem promover prematuramente equivalência geral ou segurança.

## Notas explícitas solicitadas pelo usuário

Após a primeira consolidação, cada revisor atribuiu a própria nota, usando a mesma escala interna de 1 a 10: 4 = rejeição leve, 5 = limítrofe, 6 = aceitação leve. Essa escala não é apresentada como formulário oficial do ICML. As notas avaliam a versão congelada, e não estimam probabilidade de aceite de uma futura versão.

| Revisor | Nota geral | Recomendação | Confiança declarada |
|---|---:|---|---:|
| 1 — rigor | 4/10 | weak reject | 4/5 |
| 2 — novidade | 4/10 | weak reject | 4/5 |

Revisor 1: rigor 4, evidência 4, contribuição 4, clareza 6. Revisor 2: novidade 4, significado potencial 6, evidência 4, clareza 6. O revisor 2 elevaria a evidência para 5 ao incorporar o held-out A1.5, mantendo a recomendação global; o revisor 1 considera essa atualização promissora sem alterar a nota do snapshot.

Ambos pedem, para reconsiderar na direção de 6/10, corrigir a teoria, confirmar os efeitos com controles que separem explicações concorrentes e demonstrar uma contribuição distinta da literatura com consequência externa convincente. Concluir mais execuções, por si só, não produz automaticamente essa mudança de nota.
