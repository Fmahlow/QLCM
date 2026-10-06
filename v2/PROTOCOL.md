# Decisão, evidências e protocolo v2 — 2026-10-06

## Hipótese e escopo

A hipótese candidata é que consistency distillation pode reduzir o custo de
inferência de um denoiser híbrido quântico em latente aprendido mantendo qualidade
e diversidade; manter circuitos no estudante pode ou não valer a pena. O estudo
caracteriza regimes qualidade/custo, não afirma quantum advantage ou “primeiro”.
Piloto Fashion-MNIST, seeds 0/1/2; MNIST confirmação posterior. Não executar grade
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
