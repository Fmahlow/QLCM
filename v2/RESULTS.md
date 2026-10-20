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

## Extensão de datasets e controle do teacher — 2026-10-13

`v2/diagnostics/dataset_teacher_campaign.py`. Pergunta dupla: a destilação
generaliza para outros dados, e o circuito no teacher faz diferença? Para cada
dataset e seed, um AE e três teachers sobre esse mesmo AE: hybrid, classical e
no_entanglement. O teacher híbrido recebe estudantes hybrid/classical/surrogate; os
teachers de controle recebem um estudante classical. Estudantes com 200 épocas e
EMA 0,95; demais hiperparâmetros iguais aos do piloto. No MNIST, AE, teacher
híbrido e seus estudantes são os do piloto e do diagnóstico (copiados, não
retreinados). 4 datasets x 3 seeds: 36 teachers, todos aprovados no gate, 0 falhas.

Mudanças de código: `teacher_kind` configurável em `v2/run.py` (default `hybrid`;
checkpoints do piloto reavaliados dão métricas idênticas), KMNIST e PneumoniaMNIST
em `v1/src/datasets.py` (aditivo). Essas mudanças alteram o fingerprint do
orquestrador: o piloto original não pode mais ser retomado com `--resume`.

Classificadores (validados no split de teste completo; orçamento fixado antes de
olhar o teste): Fashion-MNIST e KMNIST com 60.000 imagens / 5 épocas, acurácia
0,8889 e 0,9314; PneumoniaMNIST com 4.708 imagens / 10 épocas, acurácia 0,8269.
O MNIST mantém 10.000 / 3 épocas (0,9589). **FID-like não é comparável entre
datasets.**

PneumoniaMNIST (MedMNIST v2, CC BY 4.0) tem 2 classes desbalanceadas (1.214 /
3.494). O gate de diversidade usa ali 0,8 x a entropia real dos rótulos de treino
(0,571) em vez de 0,8 ln10; com 2 classes a entropia mede só o equilíbrio entre
classes, e o FID-like vem de um classificador binário.

FID-like, média ± desvio populacional de três seeds. "T" é o tipo de teacher.

| Dataset | T | Teacher 1 passo | Teacher 4 | Teacher 32 | Estudante clássico 1 passo | Estudante clássico 4 | Estudante clássico 32 |
|---|---|---|---|---|---|---|---|
| MNIST | hybrid | 483,1 ± 6,5 | 52,3 ± 4,7 | 24,7 ± 1,5 | 28,0 ± 2,7 | 24,1 ± 3,0 | 41,4 ± 11,2 |
| MNIST | classical | 489,9 ± 5,2 | 53,4 ± 4,2 | 25,3 ± 1,5 | 28,6 ± 3,0 | 24,5 ± 3,6 | 39,1 ± 9,9 |
| MNIST | no_entanglement | 484,2 ± 7,1 | 52,7 ± 4,1 | 24,9 ± 1,4 | 28,2 ± 2,8 | 23,8 ± 3,3 | 41,3 ± 10,5 |
| Fashion-MNIST | hybrid | 479,9 ± 14,0 | 104,3 ± 1,2 | 45,4 ± 1,0 | 45,7 ± 1,9 | 41,7 ± 3,6 | 41,5 ± 15,2 |
| Fashion-MNIST | classical | 493,4 ± 8,7 | 105,6 ± 1,2 | 45,1 ± 0,7 | 44,7 ± 1,1 | 40,4 ± 3,4 | 43,0 ± 17,3 |
| Fashion-MNIST | no_entanglement | 481,6 ± 14,0 | 104,8 ± 1,7 | 45,2 ± 1,0 | 46,1 ± 1,4 | 42,1 ± 3,7 | 39,7 ± 14,0 |
| KMNIST | hybrid | 1183,6 ± 16,4 | 371,9 ± 28,5 | 211,7 ± 12,2 | 216,8 ± 11,4 | 212,0 ± 13,2 | 182,9 ± 18,5 |
| KMNIST | classical | 1186,3 ± 16,2 | 372,8 ± 24,7 | 210,9 ± 9,9 | 218,1 ± 9,2 | 212,0 ± 11,9 | 186,5 ± 7,9 |
| KMNIST | no_entanglement | 1182,8 ± 17,6 | 372,5 ± 28,9 | 212,1 ± 12,3 | 219,0 ± 11,2 | 213,5 ± 13,4 | 188,3 ± 18,8 |
| PneumoniaMNIST | hybrid | 116,2 ± 6,0 | 31,6 ± 5,3 | 7,1 ± 2,9 | 6,1 ± 2,6 | 5,2 ± 1,5 | 105,4 ± 88,5 |
| PneumoniaMNIST | classical | 118,0 ± 6,7 | 32,1 ± 5,5 | 7,1 ± 2,9 | 6,0 ± 2,5 | 5,2 ± 1,4 | 96,2 ± 71,2 |
| PneumoniaMNIST | no_entanglement | 116,1 ± 6,0 | 31,5 ± 5,3 | 7,1 ± 2,9 | 6,0 ± 2,5 | 5,3 ± 1,5 | 107,2 ± 90,9 |

Estudantes do teacher híbrido, FID-like em 1 passo (hybrid / classical / surrogate):
MNIST 28,0 / 28,0 / 28,0; Fashion-MNIST 45,1 / 45,7 / 45,2; KMNIST 216,3 / 216,8 /
216,3; PneumoniaMNIST 5,8 / 6,1 / 5,8. Tempo de 1 passo para 512 amostras: ~11 ms
com circuito simulado (teacher ou estudante hybrid), ~1,3 ms clássico, ~2,2 ms
surrogate, nos quatro datasets.

Achados:

- **A destilação generaliza.** Nos quatro datasets, o estudante em 1 passo fica no
  nível do teacher completo de 32 passos e muito à frente do teacher em 1–4 passos.
  O critério de avanço proposto é atendido nas três seeds em todos os datasets por
  estudantes hybrid, classical e surrogate do teacher híbrido (em 4 passos no MNIST;
  em 1–8 no Fashion-MNIST e KMNIST; em 1–2 no PneumoniaMNIST).
- **O circuito no teacher não faz diferença.** Teachers hybrid, classical e
  no_entanglement têm curvas DDIM indistinguíveis em todos os datasets (diferenças
  muito menores que o desvio entre seeds), e os estudantes clássicos destilados de
  cada um também. O teacher clássico é ~7x mais rápido por passo.
- **O circuito no estudante não faz diferença.** Confirmado nos três datasets novos.
- **Consequência:** nesta arquitetura (ramo de 4 qubits somado a um backbone MLP
  clássico), o pipeline totalmente clássico iguala o híbrido em qualidade em todo
  ponto medido, com custo menor. Não há contribuição mensurável do circuito, nem com
  entrelaçamento. Isso é um resultado sobre esta arquitetura e escala, não sobre
  circuitos em geral.
- **Mais passos no estudante:** degradação forte no PneumoniaMNIST (FID-like ~100 em
  32 passos, desvio ~80) e moderada no MNIST; no KMNIST os estudantes melhoram com
  mais passos. Não investigado.
- **KMNIST:** o teacher completo passa no gate mas tem FID-like ~212, alto em termos
  absolutos; a qualidade generativa ali é fraca para todos os modelos, provavelmente
  limitada pelo AE de latente 16 ou pela capacidade do denoiser (não testado).

Limitações adicionais: os tempos do PneumoniaMNIST foram reavaliados com a GPU
livre depois do treino; os treinos rodaram em paralelo e seus tempos de parede não
são benchmark. Controle do teacher sem estudantes hybrid/surrogate (só classical).
