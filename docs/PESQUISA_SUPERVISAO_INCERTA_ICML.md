# Pesquisa de método: intervenções, validade e supervisão parcial

20/10/2026. Atualiza a proposta baseada em Counter-nuScenes + doScenes.
Este documento formula hipóteses e limites; resultados executados estão em
`RESULTADOS_SUPERVISAO_INCERTA.md`. Nenhum método novo de nível ICML está
estabelecido por estas leituras ou pelos pilotos.

## O que a pesquisa adicional mudou

| Fonte primária | O que já existe | Implicação para nós |
|---|---|---|
| [CLIC — From Action Labels to Sets](https://arxiv.org/abs/2502.07645) | Construção/agregação de conjuntos de ações desejadas e ajuste da distribuição da política; feedback parcial, relativo e ruidoso | Uma loss de restrições ou conjuntos aceitáveis não é novidade |
| [IVA — Do What?](https://aclanthology.org/2025.findings-emnlp.635/) | Premissas falsas, esclarecimento e alternativas fundamentadas na observação | Detectar objeto ausente e perguntar também não basta |
| [Learning from weak labelers as constraints](https://proceedings.iclr.cc/paper_files/paper/2025/hash/e91fb65c6324a984ea9ef39a5b84af04-Abstract-Conference.html) | Aprendizagem sob rótulos fracos com informação sobre limites do erro dos anotadores | Confiabilidade latente e restrições de erro já possuem métodos próprios |
| [Learning to Reweight Examples, ICML 2018](https://proceedings.mlr.press/v80/ren18a.html) | Reponderação por gradientes com pequeno conjunto limpo | Usar auditorias para pesar exemplos é baseline, não contribuição automática |
| [CFNBC](https://arxiv.org/abs/2607.27261) | Seleção de pares de nuisances por mudança da ação e cobertura de respostas; reparo da política | Selecionar remoções por sensibilidade e fazer fine-tuning já tem concorrente direto |
| [MAGMA-GEN](https://arxiv.org/abs/2609.20056) | Geração de supervisão de recuperação validada por reexecução contrafactual | Filtrar reparos pela melhora posterior também já é uma linha existente |

CLIC é especialmente relevante: a formulação de conjuntos desejados pede
interseção com o conjunto ótimo e permite ações subótimas adicionais. A
construção usa pertinência suave, agregação de correções e alinhamento de
distribuição. Portanto, trocar um alvo exato por uma região ou usar uma mistura
de confiabilidade não é uma distinção suficiente. Consultamos suas seções de
formulação e política, não apenas o título.

IVA também foi consultado no texto completo. Trabalha com requisições corretas
e de premissa falsa, treina instruções/respostas com observações e contempla
correções linguísticas. Não podemos apresentar o caso “o referente desapareceu”
como um problema nunca estudado. A questão candidata precisa envolver como
transferir supervisão entre mundos com diferentes condições de tarefa.

CFNBC pressupõe que a transformação preserve a ação do especialista. Isso é
útil como controle de augmentação. Remover um objeto que restringe a direção ou
que é o referente da instrução não satisfaz automaticamente essa premissa.

O artigo ICLR de anotadores fracos considera limites de erro fornecidos como
informação adicional, evitando depender apenas de um modelo generativo de
rótulos. Isso reforça a necessidade de explicitar de onde vem a confiabilidade
no nosso caso. Não executamos uma reprodução de CLIC, IVA, Ren ou desse artigo;
os pilotos implementam controles e uma mistura simples, com escopo próprio.

## Três variáveis que não podemos fundir

1. **A edição é fiel?** Removeu o elemento pretendido sem introduzir alterações
   indevidas? Um score de pixels fora da caixa é só diagnóstico parcial.
2. **A instrução ainda define uma tarefa?** O referente existe, o pedido tem
   sentido neste instante e há alguma realização aceitável?
3. **A restrição de ação está correta?** Mesmo com boa edição e tarefa válida,
   copiar uma ação ou impor uma direção pode ser errado.

Um filtro único pode aprender uma correlação conveniente sem distinguir essas
situações. Um classificador de validade perfeito tampouco fornece automaticamente
a trajetória correta. O piloto de relações erradas estressa apenas o terceiro
eixo, não valida os três.

## Contraexemplo de informação insuficiente

Considere um factual conhecido e uma observação editada. Há duas ações
disponíveis na edição, esquerda e direita. Em dois problemas possíveis, todas
as imagens, máscaras, instruções, ações factuais e bits “tarefa válida” são
idênticos. No primeiro problema apenas esquerda satisfaz a restrição normativa
editada; no segundo, apenas direita. Essa restrição não foi observada nem é
determinada por uma hipótese estrutural conhecida.

Os dados disponíveis são iguais nos dois problemas, logo qualquer aprendiz
baseado somente neles produz a mesma previsão/distribuição. Não pode garantir
a ação correta em ambos. Sob uma mistura uniforme dos dois problemas, a melhor
probabilidade de acerto é no máximo 1/2. Isso é uma observação elementar de
não identificação, não um novo teorema para ICML.

Ela impede uma promessa injustificada: pares de imagens e um bit de validade
não bastam, em geral, para gerar rótulos corretos de direção. Precisamos de
restrições independentes, um planejador validado ou hipóteses estruturais
defensáveis. Não usar a resposta congelada da VLA como se resolvesse esse déficit.

## Hipótese de método que merece investigação posterior

Aprender **quais restrições de ação podem ser transferidas entre os dois mundos**,
condicionado à instrução e à intervenção. A saída inclui restrições possíveis,
incerteza e um estado de tarefa indefinida; não simplesmente um peso para a
mesma loss sobre todos os pares. A anotação precisa permitir várias ações
aceitáveis e abstenção quando o par não estabelece uma resposta.

Notação de trabalho: um operador sobre restrições
`K(C_editada | imagem_original, imagem_editada, instrução, C_factual)`.
Isso não autoriza mudar o objetivo do passageiro: remover o carro branco não
permite inventar que o passageiro passou a querer seguir o carro preto.

Diferenças **a demonstrar**, ainda não estabelecidas:

- Contra CLIC: aprender a transformação das restrições entre mundos, em vez de
  assumir que conjuntos desejados já foram fornecidos para o estado relevante.
- Contra IVA: melhorar aprendizagem da política em tarefas válidas que mudaram,
  além de reconhecer/comentar premissas falsas.
- Contra ECT: aproveitar supervisão incompleta com custo medido, sem esconder
  ações contrafactuais completas no treinamento.
- Contra filtragem: aproveitar partes válidas da supervisão quando a intervenção
  altera somente alguns elementos da tarefa; não resolver tudo descartando pares.

Um pipeline de grounding + regras fixas é um controle obrigatório. Se ele
explicar o ganho, a narrativa de um novo método de ML não está sustentada.

## Critérios de continuação para uma proposta de ICML

- Pelo menos uma diferença algorítmica/formal específica que sobreviva aos
  concorrentes acima. “Aplicado a AV” e combinação de datasets não são suficientes.
- Melhora da política sobre os melhores controles simples, não só acurácia de
  classificação de pares ou correlação com mudanças da VLA.
- Comparação justa de anotação: separar bit de validade, restrição parcial,
  ação exata e verificações da edição. Medir tempo e concordância; não inventar
  créditos equivalentes entre esses tipos.
- Generalização por cena, mecanismo e instrução; pelo menos outro backbone
  para alegação ampla. Frames e edições da mesma cena ficam na mesma partição.
- Avaliação dinâmica para alegações de direção/segurança em malha fechada.
  Um toy estático não demonstra isso.
- Fonte independente para alvos normativos. A VLA que será corrigida não pode
  ser seu próprio árbitro de correção contrafactual.

Não vamos escolher arquitetura ou aumentar fine-tuning só porque uma fórmula
soa sofisticada. Primeiro, a mistura conjunta de confiabilidade precisa superar
seus controles; mesmo um resultado positivo seria apenas uma triagem.

## Fontes e profundidade

Quatro PDFs e extrações preservados em
`artifacts/uncertain_supervision_research_20261020/`, com hashes. CLIC, IVA e
CFNBC foram consultados em formulação/método; Ren em formulação/resumo e no
texto disponível. O artigo ICLR foi consultado pelo resumo e acesso ao PDF;
MAGMA-GEN foi triado pelo resumo, sem revisão integral. Limitações e mecanismos
dos papers não devem ser reduzidos a seus abstracts numa revisão definitiva.
