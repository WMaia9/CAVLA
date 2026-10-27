# Estimandos e falhas que os pilotos precisam distinguir

Estas observações são diagnósticos elementares, não teoremas novos de ICML.

## 1. Um head auxiliar perfeito pode coexistir com uma política errada

Considere `z=(z_action,z_effect)`, `pi(z)=z_action` e
`h(zI-z)=z_effectI-z_effect`. Escolha `z_action` constante e faça a segunda
componente codificar o efeito correto. O erro auxiliar é zero e a política não
reage. Logo, decodabilidade do efeito não implica uso pela política. Este
contraexemplo está em `tests/test_core.py`.

## 2. A loss mista é uma combinação de erros supervisionados

Se os quatro rótulos de ação são conhecidos e `e_j=ahat_j-a*_j`, o erro de
interação é `e3-e2-e1+e0`. Seu quadrado introduz termos cruzados entre erros.
Ele não acrescenta informação de rótulo aos quatro exemplos. Pode alterar
otimização/generalização, mas isso precisa vencer uma baseline genérica de
contrastes e orçamento equivalente. Se todos `e_j=0`, a loss de interação já é zero.

## 3. Alteração de ADE não é deslocamento da política

Com referência `g=0`, previsões `a=+1` e `b=-1` têm a mesma distância até `g`,
mas estão a distância 2 entre si. Portanto `|ADE(b,g)-ADE(a,g)|` pode ser zero
mesmo com forte resposta. Isso também não diz qual previsão seria correta
depois de editar uma cena: o futuro logado corresponde à cena original.

## 4. Efeito por seed depende do acoplamento do ruído

Se `U~N(0,1)`, os geradores `F0(U)=U` e `F1(U)=-U` têm a mesma distribuição
de saída. Ainda assim, `E[(F1(U)-F0(U))²]=4`. Um estimando baseado nesse
acoplamento responde a uma pergunta diferente da igualdade das distribuições.
Ele pode ser legítimo se a identidade das variáveis exógenas fizer parte do SCM
do algoritmo; não deve ser apresentado como distância intrínseca das marginais.

## 5. Distância amostral inclui variância

Para pares iid de respostas vetoriais `D_i`, com média `mu`,

`E[||D_i||²] = ||mu||² + tr(Cov(D_i))`.

A média de deslocamentos quadráticos por seed mistura mudança média e
variabilidade. A U-estatística

`U = sum_{i != j} <D_i,D_j> / (n(n-1))`

tem esperança `||mu||²`. O código divide também pela dimensão para reportar
média por coordenada. Valores negativos em amostras finitas são possíveis;
clipping muda o estimando. Isto é uma identidade estatística padrão.

## 6. Mudança de média não detecta toda mudança de distribuição

Duas políticas podem ter a mesma trajetória média e diferentes probabilidades
de modos perigosos. MMD com kernel característico compara distribuições, mas
não identifica risco sem uma função de custo externa. Em amostras pareadas,
o termo cruzado com mesmo índice pode ter dependência: a implementação usa
apenas índices distintos em todos os termos. A referência estatística é
[Gretton et al.](https://jmlr.org/papers/v13/gretton12a.html).

## 7. O que não concluímos

Esses fatos não identificam o SCM físico, não provam generalização OOD e não
transformam um efeito da política em uma ação correta. Uma futura contribuição
formal teria que especificar suporte de intervenções, observabilidade, política
de referência, distribuição de tarefas e condições da garantia.
