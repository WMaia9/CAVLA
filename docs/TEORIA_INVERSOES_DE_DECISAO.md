# Teoria: quando a integração numérica muda a decisão de uma política de flow matching

Notas de trabalho, com esboços de prova a revisar. Notação: x_t=(1−t)z+t·x₁, z~N(0,I), campo marginal v(x,t)=E[x₁−z | x_t=x], fluxo exato Φ: z↦x₁, decisão D(x₁)∈{classes} com fronteira 𝔅. Um integrador explícito de ordem p e grade 0=t₀<…<t_k=1 define Φ_k. Inversão: D(Φ_k(z))≠D(Φ(z)).

## Proposição 1 (simetria ⇒ nenhuma inversão)

Seja g uma reflexão ortogonal com hiperplano fixo H={x: ⟨μ,x⟩=0}. Suponha p₁∘g=p₁ e decisão D(x)=sign⟨μ,x⟩. Então, para qualquer Runge–Kutta explícito com passos h_i tais que h_i·Lip(v(·,t))<1, vale D(Φ_k(z))=D(Φ(z)) para quase todo z.

*Esboço.* A base gaussiana é invariante a g, logo a lei de x_t também é, e v é g-equivariante: v(gx,t)=g·v(x,t). Para x∈H, gv(x)=v(x), então v(x)∈H. Cada estágio RK é uma composição de mapas equivariantes; o passo Ψ_h(x)=x+h·Σb_iv(…) é equivariante e leva H em H. Com h·L<1, Ψ_h é homeomorfismo de ℝᵈ e de H (Banach), logo leva ℝᵈ∖H em si. Cada semiespaço é conexo e Ψ_h depende continuamente de h com Ψ₀=id, então os lados se preservam. O mesmo vale para o fluxo exato. ∎

*Leitura.* Erros numéricos podem deformar arbitrariamente a trajetória (ADE grande) sem nunca mudar uma decisão protegida por simetria. Inversões são um fenômeno de **quebra de simetria**: precisam que a fronteira de decisão, puxada para o espaço de ruído, não seja invariante ao integrador. Verificado no modelo solúvel: 0 inversões em todos os k com w=0,5; 2–20% com w≠0,5.

## Proposição 2 (taxa de inversão e alocação ótima de passos)

Suponha v suave por partes, fronteira 𝔅 suave e densidade de z limitada perto de Φ⁻¹(𝔅). Para uma grade com densidade de passos λ(t) (h(t)≈1/(kλ(t))) e método de ordem p:

P(inversão) = k^{−p} ∫_{Φ⁻¹(𝔅)} ρ(z) | n(z)·∫₀¹ λ(t)^{−p} J_{t→1}(z) ℓ(z,t) dt | dS(z) + o(k^{−p}),

em que ℓ é o coeficiente do erro local, J_{t→1} a jacobiana do fluxo de t até 1 e n a normal da fronteira. Isso segue da expansão clássica do erro global (soma de erros locais transportados) combinada à fórmula de coárea: só a componente normal à fronteira, avaliada em pontos próximos dela, muda decisões.

*Consequências.*
1. **Escala:** k^{−p} (verificado: Euler 1/k; Heun mais rápido).
2. **Alocação ótima (equidistribuição orientada ao objetivo):** minimizar a integral sob ∫λ=1 dá λ*(t) ∝ w(t)^{1/(p+1)}, com w(t)=E[ρ|n·J_{t→1}ℓ| na fronteira]. Isso é o princípio do dual-weighted residual (engenharia numérica), com o funcional de interesse sendo a decisão.
3. **Conexão com informação:** w(t) só é grande quando trajetórias próximas da fronteira ainda têm componente normal sendo determinada, isto é, quando a posterior da decisão dado x_t está mudando. Isso motiva o proxy dI(D;x_t)/dt. A ligação é empírica (correlação 0,82–0,96 no modelo solúvel) e falta prova; o proxy observável no Alpamayo é a concordância da classe do endpoint previsto com a final.
4. **Contra o tempo entrópico:** H(x₁|x_t) inclui d−1 direções tangentes à fronteira, que não entram em w(t). Quando d cresce, o tempo entrópico migra os passos para onde se resolvem detalhes tangenciais; a perda de fidelidade da decisão cresce com d (verificado: em d=64, 2–4× pior que uniforme).

## Pendências

- Prova rigorosa da Proposição 2 com fronteiras não suaves (classes por limiares de velocidade/heading são poliedrais).
- Relação quantitativa entre w(t) e dI/dt (talvez via identidade I-MMSE restrita ao funcional de decisão).
- Versão estocástica (ruído inicial e linguagem amostrada) e efeito do seletor de amostras com memória.
