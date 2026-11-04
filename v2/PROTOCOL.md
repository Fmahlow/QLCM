# Decisão, evidências e protocolo v2 — 2026-10-06

## Hipótese e escopo

A hipótese candidata é que consistency distillation pode reduzir o custo de
inferência de um denoiser híbrido quântico em latente aprendido mantendo qualidade
e diversidade; manter circuitos no estudante pode ou não valer a pena. O estudo
caracteriza regimes qualidade/custo, não afirma quantum advantage ou “primeiro”.
Piloto MNIST, seeds 0/1/2; Fashion-MNIST é extensão opcional posterior. Não executar grade
ampla ou shots/ruído antes de gate. Handoff contém comandos e critérios propostos.

## Literatura e novidade

Busca em 2026-10-06: quantum latent consistency; quantum consistency distillation;
quantum diffusion distillation teacher student; hybrid quantum latent diffusion;
OQIDDM e DOI. Não encontrar termo exato não prova ausência do método.

- Kölle 2401.07049v1, seção 3.4, texto integral:
  https://arxiv.org/html/2401.07049v1 . Composição unitária/single sampling em estados
  de imagens, não teacher-student latente aprendido. Uma aplicação de U não implica
  custo constante de preparação, gates, profundidade, tomography ou shots.
- OQIDDM, DOI https://doi.org/10.1016/j.future.2025.107875 : acesso parcial ao preview
  editorial. Combina OQINN, consistency e sampling O(1) com matriz unitária; datasets
  MNIST/Fashion/EMNIST/faces, hardware. Não foi possível verificar equações completas
  de teacher, objetivo, EMA e encoder. Leitura integral é pendência de novidade.
- De Falco et al. https://arxiv.org/html/2501.11174v1 : texto completo. AE clássico
  aprendido, latente 10D, denoiser com VQCs para imagem/tempo/combinação e residual;
  FID/KID/IS. Esses componentes não são novidade de QLCM. Difusão iterativa no método
  inspecionado, não identificado consistency teacher-student.
- Yeter-Aydeniz et al. https://arxiv.org/html/2508.09903v1 : texto completo. QVAE,
  quantum residual DDPM em imagens médicas, 1000 passos, qualidade/diversidade/ruído.
- Qian et al. https://arxiv.org/html/2607.07072v1 : texto completo. AE + mixed-state
  latent diffusion, previsão rho0 e atualização analítica; prever endpoint não é
  destilar a trajetória em poucos passos.
- https://arxiv.org/html/2602.03405v1 : observáveis não locais adaptativos/skip;
  readout mais expressivo isoladamente também tem antecedentes.
- https://iris.uniroma1.it/handle/11573/1755867 : resumo institucional DDPM/DDIM
  híbrido e overhead de conversão; texto integral não acessado.

Veredito: H/R/T quantum student com teacher clássico parece combinação incremental.
Nova hipótese teacher híbrido e custo de chamadas/medições é candidata substancial,
mas não confirmada como inédita. Comparar com OQIDDM e com DDIM é obrigatório.

## Auditoria histórica e dados

v1 foi preservado. README antigo relata 30 treinos/50 avaliações; paper README
estava desatualizado. Existem tabelas transcritas e sete grids locais. Há dois
smokes antigos de difusão em pixels, com timesteps diferentes (10/5), não QLCM.
Não há summary.csv, métricas JSON, checkpoint AE/teacher/student/classifier da
campanha local ou rastreado no Git inspecionado. Usuário confirmou dados no servidor
sem internet; não estão perdidos nem no Hugging Face. Não houve acesso remoto.

Recuperar: summary.csv; todos metrics JSON (histogramas); configs/version hash;
AE/teacher/students e metadados; classifier e real_features; logs de perdas;
ambiente/GPU/CUDA; seeds; arquivos das duas campanhas anteriores; grades completas;
reconstruções AE e amostras teacher para diagnóstico. Médias/DP populacional de três
seeds no artigo não permitem recuperar linhas individuais ou causalidade.

Achados v1: alpha_bar final ~0.60295 em T50 mas sampler inicia N(0,I); latentes sem
normalização; sem fronteira identity; chamada student(t0) não supervisionada pelo
ramo online; target online/no EMA; MSE clean peso0.5 pode induzir média condicional;
pares DDIM com saltos grandes e enviesados; teacher/AE checkpoints opcionais e
comandos antigos sem teacher; parâmetros de módulos não usados contados; RZ/CZ
finais invisíveis ao readout Z (depth1 equivalente separável); runtime sem CUDA sync;
SciPy faltava nos requisitos; classifier sem validação held-out registrada. Não
atribuir colapso a nenhum desses fatores sem intervenção controlada.

O artigo histórico chama baseline parameter-matched apesar de 90640 vs139640,
usa “estatisticamente indistinguível” sem teste de equivalência e sugere causalidade
subset/épocas sem ablação. FID-like usa features de classifier pequeno, não compara
numericamente com FID publicado. Confiança softmax não é acurácia. Correções do novo
método não validam nem corrigem resultados antigos. O manifesto registra fontes.

## Decisões metodológicas implementadas

v2 usa agenda cosine em 0..T com clean exato em t0 e alpha_bar terminal=1e-6.
Latentes são normalizados por média/std de treino AE (clamp std0.01 documentado).
Prior ainda aproxima marginal terminal; registrar estatísticas para verificar.
Teacher v-pred evita dividir por sqrt(alpha_bar) pequeno. Distilação usa pares
adjacentes uniformes e DDIM determinístico como solver discretizado, sem afirmar
trajetória ODE exata. Loss consistency apenas, sem MSE clean extra; fronteira
f=abar*z+sqrt(1-abar)*F impõe f(z,0)=z; EMA após optimizer. Coeficientes são escolha
simples para piloto, não reivindicação de parametrização ótima de LCM.

Circuito RY/RZ/CZ com rotação final RY(pi/4) torna entanglement/phase observáveis.
Controle no_entanglement remove só CZ; surrogate executa exatamente a mesma função
separável, com mesmos parâmetros e gradientes. O surrogate é controle de execução
clássica do bloco separável, não aproximação do circuito entrelaçado. Testes
confirmam equivalência e diferença com entrelaçamento em entradas aleatórias.
Não trocar o circuito v1 silenciosamente. Apenas módulos utilizados são instanciados;
contar parâmetros registrados e com gradiente. Igualar budget de épocas não iguala
capacidade nem custo; comparar fronteiras de Pareto e estudante clássico.

Teacher full DDIM, teacher DDIM reduzido e estudantes compartilham AE/prior; entradas
iniciais comuns, RNG fixo nas repetições. Student few-step renoisa em níveis positivos
antes da próxima avaliação, sem chamada extra endpoint t0. End-to-end timing inclui
decoder, sync e repetições; treinamento e hardware não incluídos. Calls per sample
não equivalem a shots, gates ou latency hardware. Shots/noise não implementados.

## Local versus GPU

Local: testes sintéticos, forward/backward, targets congelados/EMA, fronteira,
surrogate/entanglement, métricas/custos, orquestração e resume. GPU: validar classifier,
qualidade/diversidade teacher, piloto multiseed, benchmark real e comparação DDIM.
Só depois considerar orçamento de shots finitos e ruído, treinamento com feedback
estatístico e curvas Pareto. Não há melhoria de qualidade demonstrada nesta entrega.

Tectonic já existia; compilador nativo falhou ao input multifile (não suportado),
com warnings overfull no TikZ. Nenhum TeX instalado. Compilar cópia temporária do
projeto com ferramenta existente se recursos cached disponíveis; diagnosticar sem
reescrever artigo histórico ou retirar evidências para forçar compilação.

## Pré-registro da extensão "onde o circuito importa" — 2026-10-20

Escrito antes de rodar qualquer experimento abaixo. Fora do protocolo do piloto.
Comum a todos: seeds 0/1/2; AE de latente 16 de cada dataset/seed reaproveitado da
campanha `results/v2-campaign-datasets-teachers` (exceto E4b); estudantes com 200
épocas / EMA 0,95; demais hiperparâmetros de `v2/gpu_pilot.json`; 512 amostras;
FID-like do classificador validado de cada dataset. Sem gate de teacher: todos os
modelos são avaliados. Treinos na CPU (2 threads por processo, em paralelo);
avaliações finais em sequência na GPU. Mudança de implementação: o circuito agora
funde as portas independentes dos dados numa matriz (mesma função; testes comparam
com a simulação porta a porta; reavaliar checkpoints antigos dá métricas idênticas).

Controles novos: `classical_branch` troca o circuito por Linear(2q,q)+Tanh com a
mesma entrada, saída e faixa (36 parâmetros contra 16 do circuito de 4 qubits e
profundidade 2: o controle tem mais parâmetros, o que é conservador para o
circuito). `hidden=0` remove o backbone MLP: o modelo vira compressor -> ramo ->
readout. `surrogate` continua sendo o equivalente exato sem entrelaçamento.

**E0: contribuição do ramo nos modelos existentes.** Para teachers e estudantes
com ramo da campanha (4 datasets): razão RMS ramo/backbone ao longo de t e FID-like
com o readout zerado. Descritivo, sem critério.

**E1: varredura de capacidade (MNIST e Fashion-MNIST).** Pipeline casado: teacher
e estudante do mesmo tipo e tamanho. Tipos hybrid, surrogate e classical_branch
com hidden em {0,4,8,16,32,128}; classical com hidden em {2,4,8,16,32,128}. Métrica
primária: FID-like do estudante em 1 passo; secundária: teacher em 32 passos.
Critérios por dataset e hidden, pareados por seed:
- *circuito contribui*: hybrid melhor que classical_branch nas 3 seeds e média ao
  menos 10% menor;
- *entrelaçamento contribui*: hybrid melhor que surrogate nas 3 seeds e média ao
  menos 10% menor;
- fronteira: FID-like contra parâmetros registrados, hybrid comparado ao melhor
  classical com igual ou menos parâmetros (descritivo).

**E2: escala do circuito (MNIST, hidden=0).** qubits em {4,6,8}, profundidade em
{2,4}, re-uploading sim/não; hybrid e surrogate em todas as combinações,
classical_branch por número de qubits. Mesmos critérios do E1 por configuração.
Tendência com qubits é descritiva.

**E3: shots finitos e ruído, só na inferência.** Modelos treinados sem ruído:
pipeline hybrid hidden=128 da campanha e hidden=0 do E1 (MNIST). Ruído
despolarizante local com p em {0, 0,001, 0,01, 0,03} por qubit após cada camada
(codificação e cada camada treinável); shots por avaliação de circuito em
{exato, 1024, 256, 64}. Teacher em {4,8,32} passos, estudante em {1,2,4}. Comparação
principal: FID-like contra total de shots por amostra (passos x shots). Hipótese:
com o mesmo total de shots, o estudante de 1 passo é melhor que o teacher.
Descritivo; a robustez só é informativa onde o E0/E1 mostram que o ramo pesa.

**E4: pendências.** (a) Por que estudantes pioram com mais passos: erro do
estudante f(z_t,t) contra o endpoint da trajetória DDIM completa do teacher a partir
de z_t, por t, em MNIST e PneumoniaMNIST. (b) KMNIST: FID-like de reconstrução do AE
para latente 16; AEs de latente 32 e 64, com teacher hybrid hidden=128 e estudante
classical. Ambos descritivos.

## Pré-registro da revisão "controles e robustez" — 2026-11-04

Escrito antes de rodar qualquer experimento abaixo, em resposta à revisão simulada de
2026-11-03 (`paper/TODO.md`). Servidor novo, mesma GPU (RTX 3080 Ti), mesmo torch
(2.14.0+cu126). Os checkpoints da campanha de 2026-10-13 não foram recuperados: a campanha
(4 datasets x seeds 0/1/2, mesmo script e orçamento: 200 épocas / EMA 0,95) é retreinada
do zero em `results/v2-campaign-datasets-teachers`, sem `--pilot/--diag` (o MNIST deixa de
reaproveitar piloto e diagnóstico, mas a config efetiva é a mesma). Diferenças numéricas
em relação à campanha original são esperadas (não determinismo da GPU e circuito fundido);
serão reportadas lado a lado. Os relatórios E0-E4 continuam valendo para os checkpoints
originais e não são refeitos.

**E5: replicação MNIST fora da amostra (TODO 1).** Seeds novas 3,4,5,6,7, mesmo script da
campanha (`results/v2-mnist-replication`), orçamento fixado antes (200 épocas / EMA 0,95),
sem nenhum ajuste. Critérios, nas 5 seeds novas:
- *destilação*: estudante hybrid de 1 passo com FID-like até 10% pior que o teacher hybrid
  em 32 passos, por seed (contagem x/5) e na razão geométrica média (IC 90%);
- *equivalência*: TOST pareado com margem de ±10% (IC 90% da razão dentro de [-9,1%, +10%])
  para hybrid vs classical (estudante 1 passo; teacher 32 passos). Com 5 seeds o poder é
  baixo; também reportamos o agrupamento com as seeds 0-2 retreinadas (8 seeds).

**E6: controle positivo (TODO 2).** Só inferência, nos checkpoints retreinados da campanha
(4 datasets x 3 seeds), teacher hybrid (32 passos) e estudante hybrid (1 passo). Perturbações
conhecidas do ramo: (a) escala da parte que depende da entrada, ramo = média + a (ramo - média),
a em {0; 0,5; 1,5; 2; 3}, média sobre latentes ruidosos de treino (a=0 é o `features_mean` do E0);
(b) permutação das saídas do ramo entre amostras do lote (mantém a marginal, destrói a
dependência da entrada); (c) calibração independente do ramo: ruído gaussiano de desvio s em
{0,05; 0,1; 0,2; 0,4} somado ao latente normalizado gerado antes do decoder. Métrica: FID-like
com 5.000 amostras (E7). Critério de *detecção* por dataset e perturbação: IC 90% da razão
pareada (3 seeds) exclui 0 e a média é de ao menos 10%. Reportar a menor perturbação detectada
e, para (c), a curva FID-like x s, que dá a sensibilidade da métrica a uma degradação conhecida.

**E7: pisos e robustez da métrica (TODO 3).** Para os 4 datasets e seeds 0-2 retreinadas:
- 5.000 amostras geradas (mesmo ruído inicial para todos os modelos de uma seed); referência
  real = 5.000 primeiras imagens de teste (PneumoniaMNIST: as 624 de teste);
- piso real-vs-real: 5.000 outras imagens de teste (PneumoniaMNIST: as 524 de validação) contra
  a referência; piso do AE: reconstrução das mesmas imagens do piso real;
- precision/recall (Kynkäänniemi et al., 2019), k=3, nas features do classificador;
- métricas no latente: Fréchet e MMD (kernel RBF, banda pela mediana) entre latentes gerados
  normalizados e latentes de teste codificados pelo mesmo AE (sem decoder nem classificador);
- modelos: teachers hybrid/classical/no_entanglement em 32 passos e DDIM em 1/2/4/8; estudantes
  hybrid/classical/surrogate do teacher hybrid e classical dos teachers de controle, em 1 e 4 passos.
Refazer os testes de equivalência do artigo (margem ±10%) com 5.000 amostras e em cada métrica.
Critério de robustez: a conclusão de equivalência do artigo se mantém se o IC 90% continuar
dentro da margem com 5.000 amostras; se não, a conclusão é enfraquecida no texto.

**E8: baselines de poucos passos (TODO 4).** Teacher hybrid, seeds 0-2, 4 datasets:
DPM-Solver++(2M) com 1/2/4/8 avaliações e Heun (2ª ordem, espaço DDIM) com 3/5/7 avaliações,
nos mesmos níveis inteiros da agenda usados pelo DDIM. [Texto original do commit 538048d,
restaurado; ver a emenda E8-a abaixo.] Regressão direta do endpoint (Luhman &
Luhman, 2021): rede classical do mesmo tamanho do estudante, treinada por 200 épocas para mapear
10.000 ruídos fixos no endpoint do teacher em 32 passos DDIM, avaliada em 1 passo. Descritivo,
com uma pergunta: o estudante consistency de 1 passo é melhor que o melhor solver com até 4
avaliações? Resposta por dataset com a razão pareada (IC 90%).

**E9: readout com correladores (TODO 5).** Novo readout `zz`: o circuito devolve <Z_i> e
<Z_i Z_j> (i<j), q + q(q-1)/2 features. Controles com a mesma saída: surrogate separável
(<Z_i Z_j> = <Z_i><Z_j>, exato sem entrelaçamento) e classical_branch Linear(2q, mesma
dimensão)+Tanh. Grade: MNIST e Fashion-MNIST, seeds 0-2 retreinadas, (q=4, prof. 2) com
hidden em {0, 4, 16} e (q=6, prof. 4, hidden 0); readout `zz` e, como par casado no mesmo
run, readout `z` nas mesmas células. Pipeline casado como no E1. Critérios do E1 por célula:
*entrelaçamento contribui* (hybrid melhor que surrogate nas 3 seeds e média >=10% menor) e
*circuito contribui* (idem contra classical_branch); e *correladores ajudam*: hybrid zz melhor
que hybrid z nas 3 seeds e >=10%. Primária: estudante em 1 passo; secundária: teacher 32.

**E10: mais seeds no E1 (TODO 7).** Grade E1 completa para MNIST e Fashion-MNIST com seeds
novas 3,4,5 (AEs novos do Fashion-MNIST; MNIST usa os AEs do E5). As tabelas do E1 passam a
ter 6 seeds (0-2 dos checkpoints originais, 3-5 novas). Critério do E1 adaptado: hybrid melhor
em ao menos 5 de 6 seeds e média >=10% menor; também o IC 90% pareado por célula. As seeds 0-2
continuam sendo as do pré-registro original; nenhuma conclusão do E1 original é reescrita sem
mostrar as duas versões.

**Grade de amostras (TODO 6).** Figura qualitativa com amostras do teacher (32), estudante
hybrid (1) e estudante classical (1), seed 0, nos 4 datasets.

*Adendo post hoc ao E7 (2026-11-04, depois da primeira execução do `review_eval`).* No MNIST o piso
real-vs-real pré-registrado (5.000 imagens de teste contra as outras 5.000) deu FID-like 17,3, acima do
teacher (16,1): as duas metades do teste do MNIST vêm de populações de escritores diferentes e não são
intercambiáveis. Mantemos o piso pré-registrado e acrescentamos um segundo, marcado como post hoc:
imagens de treino nunca vistas pelo AE/teacher (índices 10.000-15.000) contra a mesma referência
(não existe no PneumoniaMNIST, cujo treino inteiro é usado).

*Adendo post hoc ao E6 (2026-11-04, depois de ver o E6).* As perturbações pré-registradas não produzem
uma degradação de tamanho conhecido: mudar o ramo altera o FID-like em 1-7% (o ramo pesa pouco, como no
E0) e o ruído no latente *melhora* o FID-like (as amostras são subdispersas; o ruído aumenta a dispersão).
Para mostrar sensibilidade a uma mudança de ~10% acrescentamos dois controles de efeito conhecido, marcados
como post hoc: (d) readout zerado (remove o ramo inteiro, como no E0) e (e) contaminação de uma fração
p em {2, 5, 10, 20}% das amostras por amostras do mesmo teacher em 2 passos DDIM (amostrador degenerado).

*Emenda E8-a (escrita em 2026-11-04 15:56 no commit 43bdeaa, antes de qualquer resultado do E8;
naquele commit o parágrafo do E8 acima foi reescrito no lugar, sem marcação; o texto original foi
restaurado acima em 2026-11-04 e a mudança é registrada aqui).* Teste de fumaça com modelos de 1 época:
Heun no primeiro intervalo explode porque sigma/alpha ~ 1e3 no nível terminal (alpha_bar_T = 1e-6);
o primeiro passo passa a ser Euler (= DDIM) e a correção de 2ª ordem vale a partir do segundo, o que
dá 2k-2 avaliações para k passos. Com 1 ou 2 avaliações o DPM-Solver++(2M) coincide com o DDIM
(primeiro e último passos de 1ª ordem). A grade executada é, por isso, DPM-Solver++(2M) com 3/4/6/8
avaliações e Heun com 4/6/8 (em vez de 1/2/4/8 e 3/5/7).

## Pré-registro da rodada de revisão 1 — 2026-11-04 (noite)

Escrito depois da revisão simulada `paper/reviews/review-2026-11-04-round1.md` e antes de rodar
qualquer análise abaixo. Só inferência sobre checkpoints existentes, exceto R5.

**R1: reavaliação das varreduras (resposta aos itens maiores 3 e 6).** E9 (seeds 0-2, todas as 16
pipelines por dataset/seed) e E10 (seeds 3-5, todas as 24 pipelines; MNIST e Fashion-MNIST) com o
protocolo do E7: 5.000 amostras (ruído inicial comum por dataset/seed), FID-like, precision/recall
(k=3), Fréchet e MMD latentes; estudante em 1 passo (primária) e teacher em 32 passos. Para cada
célula e para os agregados (células como unidade), as comparações hybrid/surrogate e
hybrid/classical_branch em cada métrica. Critério de robustez: a conclusão do texto (entrelaçamento
não contribui; o circuito não contribui além do surrogate separável) se mantém se os agregados
hybrid/surrogate continuarem equivalentes (IC 90% dentro de ±10%) em FID-like e Fréchet latente.

**R2: controle positivo no nível do ramo (item maior 1).** Nos checkpoints do E10 (seeds 3-5) com
h em {0, 4, 16}, MNIST e Fashion-MNIST, pipeline hybrid: as perturbações pré-registradas do E6
(escala a em {0; 0,5; 1,5; 2; 3} da parte que depende da entrada; permutação; readout zerado),
estudante em 1 passo e teacher em 32, 5.000 amostras, mesma regra de detecção do E6 (IC 90% exclui 0
e mudança >= 10%).

**R3: controle positivo no nível do modelo (item maior 1).** Pares de pipelines treinadas de forma
independente com diferença conhecida de capacidade, avaliados com a mesma máquina de razões
pareadas: MLP puro h=16 contra h=32 e h=8 contra h=16 (E10 seeds 3-5 com 5.000 amostras, e E1+E10
seis seeds com 512); hybrid h=32 contra h=128. Pergunta: o IC 90% exclui 0 para diferenças de
~10-20% entre modelos treinados independentemente? Descritivo.

**R4: pisos (item maior 4).** Regra única: o piso relevante é o de imagens reais da distribuição de
treino que nenhum modelo viu (treino 10.000-15.000; PneumoniaMNIST: validação), porque os geradores
são treinados no split de treino; o piso pré-registrado (teste contra teste) é reportado ao lado.
Reconstruções do AE das imagens do piso relevante e das próprias imagens de referência. P/R do
PneumoniaMNIST com 624 amostras geradas (média de 10 subamostras) para igualar os tamanhos.

**R5: regressão do endpoint com mais dados (item menor 10).** Mesma rede e orçamento de épocas,
100.000 pares ruído-endpoint em vez de 10.000, perda de validação em 2.000 pares separados.
Descritivo; substitui a frase "pior" do texto se a diferença sumir.

**R6: análises sem novos dados.** Intervalos agregados com bootstrap sobre seeds (células de uma seed
compartilham o AE), regra do E10 também com média geométrica das razões pareadas,
surrogate/classical_branch em h>=16, todos os intervalos que excluem 0 nos dois sentidos.
