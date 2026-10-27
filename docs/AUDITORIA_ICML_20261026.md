# Auditoria independente do candidato ICML — 26/10/2026

Revisão do rascunho `paper/main.tex`, tabelas e artefatos, protocolos, código dos probes, configurações efetivamente geradas no AlpaSim, processos e logs. Os jobs continuam intactos. As contagens abaixo são um retrato durante a execução, não resultados finais. Os pareceres em `paper/reviews` são material interno, não avaliações da conferência.

## Veredito

Há um candidato de pesquisa identificável, mais consistente que em 22/10: investigar o efeito da discretização sobre a distribuição de ações e separar ganhos atribuíveis à contração do ruído dos demais efeitos do sampler. Ainda não há uma contribuição ICML demonstrada. O núcleo atual é evidência empírica de uma família, com teoria parcialmente clássica e afirmações distribucionais mais fortes que os testes disponíveis. Uma contribuição de análise pode ser adequada ao ICML sem propor outra loss, desde que estabeleça um mecanismo novo, seus limites e consequências de forma convincente.

Recomendação: manter a investigação, concluir o piloto já iniciado e corrigir o enquadramento antes de expandir indiscriminadamente os experimentos. Número de modelos e tamanho do dataset, sozinhos, não resolvem a contribuição científica.

## O que avançou e está sustentado

- Três gerações do Alpamayo, mesmo conjunto de 60 clipes, seis ruídos e referência Heun32. No Euler10, tau efetivo por dispersão: 0,833 / 0,837 / 0,833. O padrão é consistente dentro da família.
- Contração em espaço de trajetória ocorre em cerca de 90–93% das amostras Euler10, não 100%. O rascunho atual já usa o número mais cuidadoso.
- Controle negativo útil: Heun3/5 não reproduz a curva de temperatura nas inversões de decisão, mesmo quando sua dispersão é semelhante à referência.
- Qwen-Drive SFT e RL dão o comportamento oposto: o solver grosseiro acrescenta dispersão. Isso restringe a hipótese e deve permanecer no paper.
- SmolVLA/LIBERO acrescenta execução em circuito fechado, mas a curva de sucesso é pouco sensível na faixa intermediária de ruído. Não confirma que temperatura explica ganhos de sucesso.
- A linha de schedules por informação foi abandonada como método quando o controle de dispersão removeu sua vantagem. Essa decisão é cientificamente correta.
- O rascunho passou a registrar incerteza estatística, amostra unificada e limitações. Ainda há TODOs, figuras ausentes e inconsistências entre o cronograma, os relatórios históricos e o texto atual.

## Execuções atuais

GPU observada: 100% de utilização, aproximadamente 65 GiB ocupados. AlpaSim e FlashDrive compartilham a placa. Isso não impede a análise das ações; tempos de execução dessas execuções concorrentes não são um benchmark limpo de latência.

### AlpaSim

Cinco configurações previstas, 60 cenas e um rollout por cena. No retrato salvo pela auditoria:

| Configuração | Cenas com arquivo de métricas | Agregação final |
|---|---:|---|
| Euler10, tau=1 | 58 | ausente: execução terminou com erro |
| Euler4, tau=1 | 8 | execução em andamento |
| Heun32, tau=1 | 0 | sem resultados novos desta rodada |
| Heun32, tau=0,84 | 0 | sem resultados novos desta rodada |
| Heun32, tau=0,65 | 0 | sem resultados novos desta rodada |

As pastas Heun já existentes contêm tentativas anteriores, sem métricas finais desta grade. A mera existência da pasta não demonstra uma avaliação concluída.

O Euler10 terminou com dois jobs falhos após retries: rota a 27,36 m da trajetória original, acima do limite de 10 m, e rota dobrando 180 graus. O runtime pulou a agregação porque `eval.allow_aggregation_with_failed_rollouts=false`. Há 58 cenas aproveitáveis, mas não se deve chamar isso de uma grade concluída nem excluir silenciosamente as duas restantes. É necessário determinar se as falhas decorrem do cenário, da geração de rota ou do comportamento do ego; falhas provocadas pela política podem ser parte do desfecho. Retentar com novo ruído e reter apenas a tentativa bem-sucedida também pode enviesar a comparação.

**Duas limitações de desenho verificadas na configuração real:**

1. `endpoints.trafficsim.skip=true` e `physics_update_mode=EGO_ONLY`. Em `services/traffic_service.py`, o modo skip interpola as trajetórias gravadas dos agentes. O ego recebe novas imagens e replaneja: existe circuito fechado para o ego. Os outros agentes, contudo, não reagem ao ego. O protocolo não deve apresentar isso como tráfego reativo.
2. `model.force_determinism=false`. O código só fixa as seeds por inferência quando essa opção é verdadeira. Parear por cena continua sendo possível, mas não se demonstrou pareamento dos ruídos/raciocínios entre configurações. Com um único rollout, isso aumenta a incerteza sobre a atribuição ao solver. A próxima rodada deve registrar seeds por cena/rollout e usar um esquema de aleatoriedade reprodutível, sem alterar a definição do piloto já iniciado.

### FlashDrive

Código oficial instalado, smoke em ambos os checkpoints e probe NVIDIA em andamento; checkpoint z-lab está na fila do script. Portanto, o cronograma que diz “não iniciado” está desatualizado. O probe faz quatro janelas consecutivas por clipe e varia o sampler mantendo o contexto de cada janela. As janelas do mesmo clipe não são observações independentes. Ainda não existe resultado consolidado que permita atribuir o ganho publicado ao cache ou à temperatura.

### Outras famílias

Download NAVSIM em andamento; ReCogDrive/DiffusionDrive ainda são planos. A generalidade positiva em direção permanece concentrada no Alpamayo. A ampliação tem valor se testar uma previsão congelada e puder refutá-la, não apenas adicionar modelos ao inventário.

## Estatística: sem diferença detectável não significa equivalência

Releitura de `artifacts/paper_numbers_v1_20261026/summary.json`, sem usar as notas dos revisores: das seis células principais (três modelos × Euler4/10), apenas **Alpamayo-R1/Euler10** passa simultaneamente os testes de equivalência declarados para ADE e inversões. As outras cinco não passam; isso é inconclusivo, não prova de diferença nem de equivalência.

Por exemplo, no Alpamayo1.5/Euler10, o resíduo em inversões é +0,46 ponto percentual, IC95% [-1,18; +2,31]; em minADE6 é +0,106 m, IC95% [-0,033; +0,334]. A semelhança dos valores centrais é interessante, mas não estabelece intercambialidade das distribuições.

O cálculo de tau faz matching da dispersão por construção. Evidências adicionais precisam vir de quantidades não usadas nesse ajuste: frequências condicionais das decisões, estrutura de modos, distâncias entre distribuições e caudas. Seis ruídos por cena limitam estimativas de modos raros. Há também diferença entre igualdade de taxas agregadas de inversão contra uma terceira referência e proximidade entre as duas distribuições comparadas.

A fração `1-resíduo/efeito` pode superar 100% ou ficar instável perto de efeito zero. Deve ser apresentada como decomposição operacional de uma métrica, com incerteza; não como porcentagem causalmente explicada de cada mudança de decisão.

Os 436 clipes já foram usados na confirmação dos schedules. Para novos testes de temperatura eles podem oferecer uma avaliação com desenho congelado, mas não devem ser descritos como dados jamais examinados pelo projeto. É preciso explicitar o uso prévio e reservar confirmação realmente nova quando a alegação depender dela. Enriquecer cenas ambíguas altera a população avaliada: separar resultados por estrato ou ponderar adequadamente ao extrapolar para a população geral.

## Teoria: correções necessárias

### A identidade da média é do campo ótimo, não automaticamente da rede

Para interpolação linear e acoplamento independente, o minimizador ideal satisfaz `v*(z,0)=E[x1|c]-z`. A rede treinada aproxima esse campo. O rascunho diz que um passo “não tem diversidade alguma” sem qualificar a idealização.

Recalculei nos próprios planos salvos a dispersão média entre as seis seeds de Euler1, nos mesmos 60 clipes: **0,228 m / 0,320 m / 0,206 m** para Alpamayo1.5/R1/2. Todos os 60 clipes de cada modelo têm dispersão acima de 1 mm. Portanto, os próprios experimentos refutam a versão literal dessa afirmação para os modelos reais. Chamar cada endpoint Euler1 de “média condicional” também exige qualificação, sobretudo após um decoder não linear de ações para trajetórias.

### A proposição para todo Runge–Kutta explícito é falsa como escrita

O texto usa `h Lip(v)<1` para afirmar que qualquer RK explícito é um homeomorfismo que preserva o lado de um hiperplano de simetria. Essa condição é suficiente para o argumento de Euler, mas não foi justificada para coeficientes gerais de RK.

Contraexemplo reproduzido em CPU: fluxo retificado exato de uma Gaussiana padrão,

`v(x,t) = (2t-1) / ((1-t)^2+t^2) * x`.

Seu Lipschitz global no intervalo [0,1] é 1. Tome `t=0,4`, `h=0,1`, `x=1` e um RK explícito consistente de duas etapas com `a21=1`, `b=(100,-99)`. Então `hL=0,1<1`, mas o resultado numérico é **-2,846**, enquanto o fluxo exato termina em **+0,981**. O método é deliberadamente extremo; ele refuta a quantificação universal, não os métodos Euler/Heun/RK4 concretos testados. Restringir a proposição a Euler ou demonstrar uma condição que dependa dos coeficientes do método.

Outro erro textual verificável: o fator gaussiano de Euler10 não fica entre 0,82 e 0,88 para todo `s>=0,3`. Em `s=10` ele vale 0,617; em `s=100`, 0,090. Limitar o intervalo de s da afirmação.

### Escala do ruído não é temperatura probabilística em geral

O parâmetro tau implementado multiplica o ruído inicial. Isso define uma família operacional legítima, mas não implica que a densidade final seja proporcional a uma potência da densidade original.

Se F é o mapa invertível do fluxo e z=F^{-1}(x), então, em dimensão d,

`q_tau(x)/q_1(x) = tau^(-d) exp[-(tau^(-2)-1)||z||^2/2]`.

Esse fator depende da posição latente, não apenas de `q_1(x)`. Em geral ele pode mudar pesos entre modos. “Temperatura efetiva” deve ser definida como escala de ruído pareada por dispersão; equivalência física ou probabilística requer hipóteses adicionais.

Também há duas temperaturas diferentes no código do Alpamayo: o `temperature=0.6` passado pelo driver configura a geração de linguagem; o tau deste estudo escala o ruído do expert de ações. O cronograma não deve propor o parâmetro oficial de linguagem como se fosse o mesmo controle de ações.

## Literatura que falta enfrentar

**Temporal Score Rescaling for Temperature Sampling in Diffusion and Flow Models**, Xu et al., ICML 2026: https://arxiv.org/abs/2510.01184 ; texto: https://arxiv.org/html/2510.01184v2 . Não encontrei essa referência em `paper`, `docs` ou `src/cavla` na auditoria.

O trabalho controla a temperatura local de modelos de fluxo/difusão e inclui manipulação com pi0. Também distingue redução de ruído/diversidade de temperatura probabilisticamente interpretável. Não demonstra, nas partes consultadas, a tese específica sobre discretização dos Alpamayos; não elimina automaticamente o candidato. Contudo, é confronto obrigatório para a nomenclatura, a novidade e os controles. Um método novo que apenas ajustasse temperatura ou diversidade enfrentaria sobreposição importante. O CNS discutido nesse paper escala ruído de SDE ao longo do processo; não é idêntico à nossa escala inicial, portanto não se deve tratá-lo como experimento já feito.

Outros confrontos primários verificados: Gaussian Flow-Matching Schedules (https://arxiv.org/abs/2609.25839), FlashDrive (https://arxiv.org/abs/2608.12932) e Coda (https://arxiv.org/abs/2609.21216). Coda inclui um corretor aprendido: explicar um baseline de poucos passos não basta para explicar a melhora desse corretor. FlashDrive combina checkpoint, streaming, quantização, raciocínio e cache: o efeito isolado do cache não é a explicação de todo o sistema.

## Prioridades para decidir o investimento

1. **Antes de ampliar:** corrigir a teoria, a definição de temperatura e a descrição efetiva do AlpaSim; registrar tratamento de falhas e seeds. Preservar o piloto e seus resultados nulos.
2. **Concluir o teste oficial do FlashDrive:** separar checkpoint e cache; avaliar medidas além de dispersão e médias agregadas. Não concluir sobre o sistema completo a partir de um componente.
3. **Verificar referência e poder:** mostrar convergência de Heun32 em cenas difíceis e diferentes escalas de ruído; calcular amostra necessária para margens substantivas. Sessenta cenas/um rollout são piloto.
4. **Congelar confirmação:** ajustar tau numa amostra e avaliar distribuição/decisões em cenas novas, com margens pré-definidas e negativos Heun/Qwen. Não escolher a métrica apenas depois de ver onde ela coincide.
5. **Fechar a contribuição:** ou uma lei/mecanismo preditivo com domínio de validade e consequências robustas, ou evidência convincente de um viés de avaliação em acelerações publicadas. “A dispersão cai com menos passos” isoladamente é insuficiente. A simples aplicação de temperatura em AV também.

O prazo de janeiro de 2027 no cronograma permanece uma hipótese de planejamento: o site oficial consultado ainda apresentou a edição 2026 e não confirmei o CFP2027. Resultados de busca para uma conferência de línguas minoritárias também chamada ICML não são o prazo da conferência de machine learning.

## Artefatos desta auditoria

- `artifacts/auditoria_icml_20261026/audit_snapshot.py`: leitura CPU dos artefatos, inventário de métricas e checagens matemáticas.
- `artifacts/auditoria_icml_20261026/summary.json`: resultados, horário e hashes dos insumos principais.
- Nenhum job, modelo, configuração de execução, rascunho ou protocolo existente foi alterado por esta auditoria.
