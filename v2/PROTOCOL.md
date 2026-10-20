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
