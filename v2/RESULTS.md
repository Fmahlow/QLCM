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

## Onde o circuito importa? Extensão pré-registrada — 2026-10-20

Pré-registro em `PROTOCOL.md` (commit 440e2ae, antes de qualquer execução). Scripts em
`v2/diagnostics/` (`branch_contribution.py`, `capacity_sweep.py`, `noise_shots.py`,
`step_degradation.py`, `kmnist_latent.py`, `summarize_extension.py`); relatórios
compactos em `v2/reports/extension-circuit/`. Treinos na CPU, avaliações na GPU.
Antes de rodar: o circuito passou a fundir as portas que não dependem dos dados numa
matriz (testes contra a simulação porta a porta; checkpoints antigos reavaliados dão
FID-like idêntico). Todas as execuções terminaram sem falhas.

Pergunta: nesta família de modelos (ramo de circuito somado a um backbone MLP, latente
de AE), existe algum regime em que o circuito, ou o entrelaçamento, contribua de forma
mensurável? E a destilação reduz o custo quântico mesmo com shots e ruído?

### E0: o que o ramo faz nos modelos existentes

Campanha de 2026-10-13, 4 datasets x 3 seeds. FID-like médio de três variantes:
inalterado; `features_mean` (as features do circuito trocadas pela média sobre latentes
ruidosos de treino, o que tira só a computação que depende da entrada); `readout_zeroed`
(tira o ramo inteiro, inclusive a constante aprendida). Razões RMS em relação ao
backbone, mediana sobre t.

| Dataset | Modelo | Passos | Inalterado | features_mean | readout_zeroed | Ramo / backbone | Parte variável / backbone |
|---|---|---|---|---|---|---|---|
| MNIST | teacher hybrid | 32 | 24,7 | 25,4 | 49,5 | 0,36 | 0,10 |
| MNIST | estudante hybrid | 1 | 28,0 | 28,0 | 37,7 | 0,24 | 0,07 |
| Fashion-MNIST | teacher hybrid | 32 | 45,4 | 48,3 | 54,2 | 0,28 | 0,11 |
| Fashion-MNIST | estudante hybrid | 1 | 45,1 | 46,0 | 47,2 | 0,19 | 0,05 |
| KMNIST | teacher hybrid | 32 | 211,7 | 214,4 | 251,8 | 0,38 | 0,09 |
| KMNIST | estudante hybrid | 1 | 216,3 | 219,2 | 216,6 | 0,23 | 0,05 |
| PneumoniaMNIST | teacher hybrid | 32 | 7,1 | 7,4 | 12,2 | 0,40 | 0,08 |
| PneumoniaMNIST | estudante hybrid | 1 | 5,8 | 5,8 | 22,2 | 0,27 | 0,06 |

Teachers no_entanglement e estudantes surrogate dão o mesmo quadro (relatório). O ramo
não é ignorado pelo otimizador: tirá-lo inteiro piora bastante. Mas quase todo o efeito
vem de uma constante aprendida (o bias do readout). A parte que depende da entrada vale
5–11% do backbone, e trocá-la pela média custa 0–3 pontos de FID-like.

### E1: varredura de capacidade com controles de mesmo tamanho

MNIST e Fashion-MNIST, 3 seeds. Teacher e estudante do mesmo tipo e tamanho; hidden=0
remove o backbone. 144 pipelines (12 reaproveitados da campanha). FID-like do estudante
em 1 passo (métrica primária), média ± desvio populacional:

| hidden | Dataset | hybrid | surrogate | classical_branch | classical (MLP puro) |
|---|---|---|---|---|---|
| 0 | MNIST | 144,4 ± 2,8 | 131,4 ± 7,8 | 138,8 ± 1,7 | — |
| 4 | MNIST | 53,8 ± 4,3 | 61,5 ± 10,8 | 54,5 ± 7,7 | 160,7 ± 9,7 |
| 8 | MNIST | 51,3 ± 6,3 | 48,1 ± 5,7 | 50,3 ± 6,8 | 51,5 ± 1,1 |
| 16 | MNIST | 36,5 ± 5,6 | 37,3 ± 6,6 | 40,9 ± 5,1 | 38,5 ± 5,3 |
| 32 | MNIST | 33,6 ± 3,5 | 33,9 ± 3,6 | 35,0 ± 4,1 | 34,3 ± 4,5 |
| 128 | MNIST | 28,0 ± 2,7 | 29,4 ± 6,3 | 30,3 ± 2,4 | 28,6 ± 3,0 |
| 0 | Fashion-MNIST | 152,1 ± 3,8 | 157,2 ± 7,9 | 98,8 ± 11,1 | — |
| 4 | Fashion-MNIST | 95,4 ± 5,7 | 96,0 ± 14,6 | 72,9 ± 7,1 | 124,2 ± 8,2 |
| 8 | Fashion-MNIST | 70,7 ± 12,2 | 72,2 ± 1,1 | 67,9 ± 6,0 | 86,0 ± 10,3 |
| 16 | Fashion-MNIST | 58,7 ± 3,1 | 57,7 ± 2,1 | 56,1 ± 2,2 | 59,0 ± 2,3 |
| 32 | Fashion-MNIST | 47,2 ± 2,7 | 46,5 ± 3,2 | 51,5 ± 0,1 | 47,0 ± 2,4 |
| 128 | Fashion-MNIST | 45,1 ± 1,5 | 43,3 ± 3,2 | 43,5 ± 0,6 | 44,7 ± 1,1 |

Coluna classical com hidden 2 (não mostrada): 278,1 no MNIST e 179,4 no Fashion-MNIST.
Tabelas completas, com estudante em 4 passos e teacher em 32, saem de
`summarize_extension.py`.

Critérios pré-registrados na métrica primária (hybrid melhor nas 3 seeds e ≥10%):
- *circuito contribui* (contra classical_branch): 1 de 12 células, MNIST hidden=16
  (+11%). Ali o surrogate sem entrelaçamento é idêntico ao hybrid por seed
  (35,2/30,4/46,2 contra 35,2/30,5/44,0) e o MLP puro empata, então a vantagem é da
  função separável, simulável classicamente, e não se repete no Fashion-MNIST (−5%).
- *entrelaçamento contribui* (contra surrogate): 0 de 12.
- Nas métricas secundárias (estudante em 4 passos, teacher em 32) há 5 passes em 48
  testes (4 de "circuito", 1 de "entrelaçamento"), em células isoladas, sem padrão em
  hidden ou dataset e sem correção para comparações múltiplas. No sentido contrário, o
  classical_branch é 31–54% melhor que o circuito no Fashion-MNIST com hidden 0 e 4.
- Fronteira por parâmetros: hybrid e MLP puro coincidem a partir de hidden 8. Em
  hidden 0–4 o hybrid supera o MLP de tamanho parecido, mas o classical_branch do mesmo
  tamanho também; a vantagem é da estrutura compressor → ramo → readout, não do circuito.
- Com hidden 0 a destilação falha (estudante de 1 passo ~140–150 contra teacher de 32
  passos ~52–56): sem backbone, o estudante não tem capacidade para o mapa de 1 passo.

### E3: shots finitos e ruído, só na inferência (MNIST)

Modelos treinados sem ruído. Simulação de matriz densidade com ruído despolarizante
local p por qubit após a codificação e após cada camada treinável; todos os qubits
medidos na base computacional, com S shots por avaliação do circuito. Média de 3 seeds.

| Pipeline | p | Shots | Teacher 4 | Teacher 8 | Teacher 32 | Estudante 1 | Estudante 2 | Estudante 4 |
|---|---|---|---|---|---|---|---|---|
| hidden 128 | 0 | exato | 52,3 | 31,1 | 24,7 | 28,0 | 27,7 | 24,3 |
| hidden 128 | 0 | 64 | 52,2 | 31,1 | 24,7 | 27,9 | 27,7 | 24,3 |
| hidden 128 | 0,03 | 64 | 52,3 | 31,1 | 24,6 | 28,0 | 27,7 | 24,2 |
| hidden 0 | 0 | exato | 86,0 | 60,0 | 52,2 | 144,4 | 141,1 | 110,7 |
| hidden 0 | 0 | 64 | 85,6 | 59,8 | 52,0 | 138,7 | 133,1 | 108,0 |
| hidden 0 | 0,03 | exato | 86,8 | 60,2 | 52,2 | 158,4 | 153,3 | 99,7 |
| hidden 0 | 0,03 | 64 | 86,2 | 60,1 | 52,1 | 149,0 | 141,5 | 95,9 |

Grade completa (p em {0; 0,001; 0,01; 0,03}, shots em {exato, 1024, 256, 64}) no relatório.

- No pipeline padrão (hidden 128), 64 shots e p=0,03 não mudam o FID-like, o que é
  coerente com o E0: a parte do circuito que depende da entrada quase não pesa. Com o
  mesmo total de shots por amostra, o estudante ganha: 1 passo x 64 shots = 64 shots dá
  28,0, contra 52,2 do teacher com 4 passos x 64 shots = 256 shots; o teacher de 32
  passos precisa de 2048 shots para chegar a 24,7. A redução de custo quântico que a
  destilação traz é real, mas vale para um circuito que, nesse modelo, pode ser
  dispensado.
- Com hidden 0 (o ramo é o modelo inteiro), a destilação falha (E1) e o teacher ganha
  em qualquer orçamento. O ruído altera o estudante de forma pequena e não monotônica
  (às vezes melhora o FID-like, provavelmente por injetar variância). Isso não é
  evidência de robustez.
- Limitações: ruído só na inferência, modelo de ruído simples (despolarizante local,
  sem erro de leitura nem crosstalk), treino sem ruído nem shots.

### E4a: por que os estudantes pioram com mais passos

MNIST e PneumoniaMNIST, estudantes da campanha (200 épocas / EMA 0,95). O erro do
estudante f(z_t,t) contra o endpoint da trajetória DDIM completa do teacher é pequeno em
todo t (MSE no latente normalizado: 0,0003 em t=1, 0,028 em t=32 no MNIST, contra
energia do sinal ~1) e fica igual quando a entrada é a que o próprio estudante gera
depois do primeiro passo. Ou seja, não é erro por passo. A causa é divergência de uma
fração das amostras quando o estudante é aplicado muitas vezes:

| Dataset | Modelo | Fração com energia latente > 4 (1 / 4 / 8 / 16 / 32 passos) |
|---|---|---|
| MNIST | teacher DDIM | 0 / 0 / 0 / 0 / 0 |
| MNIST | estudante classical | 0 / 0,003 / 0,010 / 0,039 / 0,124 |
| PneumoniaMNIST | teacher DDIM | 0 / 0 / 0,001 / 0,004 / 0,007 |
| PneumoniaMNIST | estudante classical | 0,010 / 0,020 / 0,042 / 0,083 / 0,189 |

Hybrid e surrogate dão o mesmo (relatório). A mediana fica perto de 1, mas a cauda
explode (máximo ~10^3–10^5 em 32 passos), o que bate com a variância de pixels
crescendo com os passos na campanha. Correção possível, não testada: limitar ou
reescalar o latente entre passos, ou treinar o estudante também com entradas
renoisadas a partir das próprias predições.

### E4b: KMNIST é limitado pelo AE

FID-like de reconstrução (512 imagens de teste, codifica → decodifica) e de geração,
3 seeds. Piso da métrica (512 outras imagens reais): 13,7.

| Latente | Reconstrução do AE | Teacher hybrid 32 passos | Teacher 4 passos | Estudante classical 1 passo | Estudante 4 passos |
|---|---|---|---|---|---|
| 16 (campanha) | 204,1 ± 3,7 | 211,7 ± 12,2 | 371,9 ± 28,5 | 216,8 ± 11,4 | 212,0 ± 13,2 |
| 32 | 81,5 ± 2,6 | 136,5 ± 6,1 | 290,4 ± 7,7 | 142,1 ± 7,3 | 136,9 ± 1,5 |
| 64 | 24,9 ± 1,3 | 113,8 ± 17,7 | 239,0 ± 27,6 | 124,6 ± 17,0 | 116,8 ± 8,1 |

Com latente 16, a reconstrução sozinha já custa tanto quanto a geração: o AE é o
gargalo. Com latente 64, o gargalo passa para o denoiser, e a destilação continua
valendo (estudante de 1 passo perto do teacher de 32).

### E2: escala do circuito sem backbone (MNIST, hidden=0)

O ramo é o modelo inteiro. 81 pipelines, 3 seeds. FID-like do estudante em 1 passo
(primária) e do teacher em 32 passos; o classical_branch depende só do número de qubits.

| Qubits | Prof. | Reupload | Estudante hybrid | Estudante surrogate | Estudante classical_branch | Teacher hybrid | Teacher surrogate | Teacher classical_branch |
|---|---|---|---|---|---|---|---|---|
| 4 | 2 | não | 144,4 ± 2,8 | 131,4 ± 7,8 | 138,8 ± 1,7 | 52,2 ± 4,5 | 52,4 ± 6,2 | 55,1 ± 4,6 |
| 4 | 2 | sim | 197,9 ± 45,9 | 172,2 ± 11,7 | 138,8 ± 1,7 | 49,8 ± 9,6 | 49,9 ± 6,4 | 55,1 ± 4,6 |
| 4 | 4 | não | 135,6 ± 5,8 | 133,7 ± 9,1 | 138,8 ± 1,7 | 49,4 ± 5,1 | 49,4 ± 5,1 | 55,1 ± 4,6 |
| 4 | 4 | sim | 274,4 ± 37,3 | 214,9 ± 62,5 | 138,8 ± 1,7 | 49,1 ± 6,4 | 52,1 ± 7,8 | 55,1 ± 4,6 |
| 6 | 2 | não | 85,9 ± 12,0 | 86,8 ± 16,0 | 72,9 ± 3,9 | 50,6 ± 2,8 | 51,5 ± 3,2 | 50,7 ± 4,3 |
| 6 | 2 | sim | 152,0 ± 21,4 | 121,6 ± 22,3 | 72,9 ± 3,9 | 49,6 ± 5,5 | 49,8 ± 5,8 | 50,7 ± 4,3 |
| 6 | 4 | não | 78,8 ± 16,1 | 73,4 ± 5,5 | 72,9 ± 3,9 | 48,5 ± 5,1 | 46,5 ± 3,3 | 50,7 ± 4,3 |
| 6 | 4 | sim | 188,4 ± 20,4 | 172,8 ± 22,5 | 72,9 ± 3,9 | 48,0 ± 3,6 | 47,7 ± 3,8 | 50,7 ± 4,3 |
| 8 | 2 | não | 52,1 ± 1,5 | 54,1 ± 1,4 | 55,3 ± 3,9 | 46,3 ± 4,1 | 46,4 ± 3,5 | 48,3 ± 3,4 |
| 8 | 2 | sim | 126,4 ± 32,2 | 83,9 ± 10,5 | 55,3 ± 3,9 | 44,6 ± 4,6 | 48,0 ± 5,6 | 48,3 ± 3,4 |
| 8 | 4 | não | 56,9 ± 3,0 | 57,4 ± 4,0 | 55,3 ± 3,9 | 46,0 ± 4,1 | 46,1 ± 3,8 | 48,3 ± 3,4 |
| 8 | 4 | sim | 158,2 ± 44,8 | 178,4 ± 22,7 | 55,3 ± 3,9 | 46,0 ± 4,7 | 46,6 ± 4,3 | 48,3 ± 3,4 |

- Critérios na métrica primária: *circuito contribui* 0 de 12, *entrelaçamento
  contribui* 0 de 12.
- No teacher (secundária), *circuito contribui* passa em 4 qubits / profundidade 4,
  com e sem reupload (+10% e +11%), mas o surrogate sem entrelaçamento é igual ao hybrid
  ali (49,4 contra 49,4 sem reupload): de novo é a função separável, não o
  entrelaçamento. *Entrelaçamento contribui*: 0 de 12.
- Mais qubits melhoram o estudante (144 → 86 → 52), mas o classical_branch da mesma
  largura melhora igual (139 → 73 → 55): o ganho vem da largura do compressor e do
  readout (2q entradas, q saídas), não do circuito.
- Re-uploading piora muito os estudantes de 1 passo e aumenta a variância entre seeds,
  sem efeito claro no teacher.
- Com 8 qubits sem reupload, a destilação volta a funcionar sem backbone (estudante
  52,1 contra teacher 46,3), mas o classical_branch faz o mesmo (55,3).

### Conclusão da extensão

Nenhum dos regimes pré-registrados mostrou contribuição do circuito que não fosse
reproduzida por um controle clássico do mesmo tamanho, e o entrelaçamento não passou
o critério em nenhuma das 24 comparações primárias (E1 + E2). Os poucos passes nos
critérios de "circuito" vêm sempre acompanhados de um surrogate separável igual ao
hybrid, que é simulável classicamente com custo linear em qubits. Nos modelos com
backbone, o ramo funciona quase só como uma constante aprendida (E0), o que também
explica a insensibilidade a shots e ruído (E3).

O que se sustenta é o resultado de destilação: com backbone, o estudante de 1 passo
iguala o teacher de 32 passos em todos os datasets e reduz as avaliações de circuito
32x; com o mesmo total de shots, ele ganha do teacher com poucos passos. Mas esse ganho
de custo vale para um circuito que, nesta família de modelos, pode ser trocado por
uma camada clássica sem perda.

Limitações: 3 seeds, 512 amostras, FID-like de classificador pequeno, sem teste de
equivalência nem correção para comparações múltiplas; só uma arquitetura de ramo
aditivo; até 8 qubits; ruído apenas na inferência.
