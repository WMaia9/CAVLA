# Controle exploratório de suporte dos rótulos

Escrito após observar resultados parciais da execução corrigida v2 e antes de
executar este controle. A corrupção +1 registrada no protocolo principal introduz
um atalho: +1 nunca ocorre nas relações corretas deste ambiente. Isso permite
detectar pares ruins apenas lendo o rótulo, sem entender a imagem. O resultado
principal deve reportar explicitamente esse limite, mesmo que algum método vença.

1. Medir um classificador logístico **apenas das relações observadas**, treinado
   nas mesmas 32 auditorias. Avaliar nas outras 64; não usar seus rótulos no treino.
2. Novo regime `structured_matched`: escolher as mesmas 24/32 aproximações do
   protocolo, mas trocar a relação -1 por **0**. Zero já aparece em pares corretos.
   Sem mudar arquitetura, pesos, margens, passos, auditorias ou testes. É um
   controle de suporte, não uma validação de tarefas semanticamente inválidas.
3. Reutilizar os cinco checkpoints factuais congelados de v2; adaptar os mesmos
   nove métodos durante 1.200 passos. O treino supervisionado de confiabilidade
   continua usando somente os 32 bits auditados. Gate original aplicado ao novo
   regime: >=10% sobre soft_filter, anchor_soft e robust_fixed, >=4/5 vitórias
   sobre cada um e retenção ID contra soft_filter.

Se a mistura conjunta não superar filtros neste regime mais exigente, não
promovê-la como método. Não ajustar tau ou pesos para obter resultado positivo.
Esta rodada é exploratória, pois sua motivação surgiu após inspecionar v2.
