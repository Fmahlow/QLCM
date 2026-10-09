# Resultados do piloto v2 e diagnóstico de orçamento dos estudantes — 2026-10-09

Execução no servidor GPU (RTX 3080 Ti, torch 2.14.0+cu126; ambiente em
`reports/pilot-mnist/environment/`). Código do piloto no commit 66bb33b, sem alterações.
Checkpoints e amostras ficam em `results/` (não versionado); os relatórios JSON
compactos estão copiados em `v2/reports/`.

Métricas: FID-like sobre features de um classificador pequeno (acurácia held-out
0,9589), 512 amostras por configuração. Não é FID Inception nem comparável a
valores publicados. Tempos: mediana end-to-end para 512 amostras, simulação exata
de statevector, sem shots nem ruído. Nada aqui é evidência de vantagem quântica.

## Verificações antes da execução

- `unittest v2.test_core`: 4/4.
- Dry-runs do orquestrador e do preparador de classificador.
- Smoke sintético completo em CUDA (8 etapas, 20 linhas), antes não testado.
- Classificador MNIST treinado do zero e validado no split de teste completo:
  acurácia 0,9589, confiança média real 0,9416 (gate: acurácia >= 0,8).

## Piloto (protocolo do HANDOFF: 50 épocas de estudante, EMA 0,999)

Três seeds, 24 etapas, 0 falhas, 134,5 min de computação. Os três teachers
passaram no gate. FID-like médio das três seeds:

| Modelo | 1 passo | 4 passos | 8 passos | 32 passos | Tempo 32 passos |
|---|---|---|---|---|---|
| Teacher DDIM | 483,1 | 52,3 | 31,1 | 24,7 | 328 ms |
| Estudante híbrido | 347,5 | 137,1 | 114,6 | 96,1 | 329 ms |
| Estudante clássico | 426,9 | 106,9 | 82,4 | 62,7 | 27 ms |
| Sem entrelaçamento | 347,1 | 136,8 | 114,0 | 94,8 | 326 ms |
| Surrogate | 347,1 | 136,8 | 114,0 | 94,8 | 54 ms |

O critério de avanço proposto não foi atendido em nenhuma seed: o melhor estudante
por seed foi o clássico em 32 passos (64,2 / 71,2 / 52,7) contra o teacher completo
(22,8 / 24,8 / 26,4). O diagnóstico abaixo mostra que esse resultado negativo vem
do orçamento de treino dos estudantes, não da destilação em si.

## Diagnóstico de orçamento dos estudantes (fora do protocolo, autorizado em 2026-10-09)

`v2/diagnostics/student_budget.py`: reaproveita o AE e o teacher de cada seed do
piloto e muda apenas `student_epochs` e `ema_decay`. Estudantes hybrid, classical e
surrogate (no_entanglement omitido: o surrogate é seu equivalente algébrico exato e
teve métricas idênticas no piloto). 27 treinos, 9 avaliações, 0 falhas. Os treinos
rodaram em paralelo; as avaliações, em sequência.

FID-like médio das três seeds, estudante híbrido:

| Épocas / EMA | 1 passo | 2 passos | 4 passos | 8 passos | 32 passos |
|---|---|---|---|---|---|
| 50 / 0,999 (piloto) | 347,5 | 342,8 | 137,1 | 114,6 | 96,1 |
| 50 / 0,95 | 29,3 | 29,0 | 24,9 | 29,9 | 44,2 |
| 200 / 0,999 | 32,3 | 32,3 | 25,4 | 30,3 | 38,3 |
| 200 / 0,95 | 28,0 | 27,7 | 24,3 | 27,8 | 43,1 |

Estudante clássico:

| Épocas / EMA | 1 passo | 2 passos | 4 passos | 8 passos | 32 passos |
|---|---|---|---|---|---|
| 50 / 0,999 (piloto) | 426,9 | 419,4 | 106,9 | 82,4 | 62,7 |
| 50 / 0,95 | 28,8 | 28,5 | 24,4 | 29,2 | 42,3 |
| 200 / 0,999 | 32,1 | 32,1 | 25,5 | 30,5 | 40,9 |
| 200 / 0,95 | 28,0 | 27,7 | 24,1 | 27,6 | 41,4 |

O surrogate fica a menos de 1 ponto do híbrido em 1–8 passos em todas as variantes.

Achados:

- **O piloto estava subtreinado.** Reduzir a EMA para 0,95 ou quadruplicar as épocas
  leva o FID-like de 1 passo de ~347–427 para ~28–32, com entropia de rótulos 2,27
  (teacher completo: 2,28). Com 50 épocas e EMA 0,999 o alvo EMA fica atrasado
  demais para propagar a fronteira até t=T.
- **A destilação compensa frente ao DDIM reduzido.** Com 200 épocas / EMA 0,95, o
  estudante em 1 passo (28,0; ~11 ms no híbrido) supera o teacher em 8 passos (31,1;
  83 ms), e em 4 passos (24,3) empata com o teacher completo (24,7; 328 ms).
- **Critério de avanço.** Com 200 épocas / EMA 0,95, os estudantes em 4 passos
  atendem o critério nas três seeds (FID-like até 10% pior que o teacher completo,
  perda de entropia <= 0,1 ln10, >= 2x mais rápido), para hybrid, classical e
  surrogate. Com 50 / 0,95 só o clássico passa nas três; com 200 / 0,999 a seed 1
  não passa.
- **Nenhum benefício do circuito no estudante.** Híbrido, clássico e surrogate são
  indistinguíveis em qualidade (diferenças menores que a variação entre seeds), e o
  clássico é ~8–12x mais rápido por passo. Pela regra do HANDOFF ("se o estudante
  clássico dominar, não ampliar o híbrido por padrão"), o estudante híbrido não se
  justifica neste regime.
- **Mais passos pioram os estudantes.** A qualidade é melhor em 4 passos e degrada
  em 16–32 (FID-like ~38–44, com alta variação entre seeds). Não investigado.

Limitações: três seeds, 512 amostras, sem teste de equivalência; diferenças de
poucos pontos de FID-like (por exemplo, estudante em 4 passos abaixo do teacher
completo na seed 0) estão dentro do ruído. Apenas dois valores de épocas e de EMA,
learning rate fixo. Shots finitos, ruído e leitura integral de OQIDDM continuam
pendentes.
