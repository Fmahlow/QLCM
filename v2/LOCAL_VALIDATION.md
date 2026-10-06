# Verificação local — 2026-10-06

Execução no Mac, CPU com 2 threads, venv isolado `/private/tmp/qlcm-pilot-venv`.
Dependências instaladas por uv; versões em requirements-local.lock.txt. Não houve
training GPU, download MNIST/Fashion ou acesso ao servidor. v1 permaneceu intacto.

Comandos efetivamente testados (da raiz):

```sh
PYTHONDONTWRITEBYTECODE=1 /private/tmp/qlcm-pilot-venv/bin/python -m unittest v2.test_core -v
PYTHONDONTWRITEBYTECODE=1 /private/tmp/qlcm-pilot-venv/bin/python -m v2.run --stage dry-run --output /private/tmp/not-written
PYTHONDONTWRITEBYTECODE=1 /private/tmp/qlcm-pilot-venv/bin/python -m v2.orchestrate --dry-run
PYTHONDONTWRITEBYTECODE=1 /private/tmp/qlcm-pilot-venv/bin/python -m v2.orchestrate --config v2/smoke.json --output /private/tmp/qlcm-orch-locked --seeds 7 --stage-hours .02 --total-hours .1
```

Quatro testes unittest passaram em 1.72 segundos na execução final: gradientes
finitos em todos parâmetros utilizados; teacher/target sem gradiente; atualização
EMA exata; fronteira identity; agenda/prior terminal; conversão v inversível;
amostragem 1/2/4/8; pares adjacentes; surrogate sem entanglement equivalente em
outputs/gradientes; entanglement altera outputs e RZ final tem gradiente; timing e
contagem de parâmetros utilizados. Dry-run não salvou checkpoint.

Orquestração sintética completou oito etapas, com AE, teacher híbrido, gate e quatro
estudantes do mesmo teacher. 20 linhas de avaliação (5 modelos x 4 budgets), todas
restritas a smoke, não evidência de qualidade do estudo científico.
Soma dos tempos de subprocessos: 31.94 s. AE 4.46 s, teacher 4.29 s, avaliação
teacher 1.89 s; estudantes hybrid/classical/no_entanglement/surrogate
4.15/4.03/4.50/5.73 s; avaliação final 2.88 s. Esses tempos incluem startup Python;
nenhuma extrapolação honesta para GPU foi possível. Para batch de 16 amostras,
tempo mediano end-to-end variou de 0.353 a 7.073 ms entre modelos/budgets. São
medições locais pequenas e ruidosas, não speedup científico. Config smoke tem
64 imagens, uma época por estágio, latent4/hidden16/3q/depth2/T8.

Retomada foi testada: todas as oito etapas foram puladas após validar checksums;
config modificada foi rejeitada por fingerprint; checkpoint temporariamente
modificado foi rejeitado e restaurado; pasta vazia sem AE rejeita distill; lock
advisory rejeita segunda orquestração simultânea. Retomada é por etapa completa,
não optimizer state. Arquivos brutos de teste estão em /tmp, não no commit.

FID-like/SciPy foi exercitado em features sintéticas: distância de array consigo
mesmo ~-1.18e-14 (erro numérico); deslocamento 0.5 em oito dimensões retornou
2.0000000000000107, valor analítico 2. Não valida classifier nem FID generativo.

Identidade histórica verificada separadamente por NumPy: depth1 do circuito v1
sem RZ/CZ finais versus original difere <6e-16; surrogate separável reproduz
readout. Novo circuito tem basis mixing final e teste de gradiente RZ.

Compilação histórica: compilador nativo chegou ao input da tabela e falhou porque
não suporta projeto multifile; warning overfull TikZ ~40.8pt. Tectonic 0.17.0 local
foi testado em cópia de v1 em /private/tmp/qlcm-tex-check com --only-cached;
interrompeu por size11.clo ausente nesse cache. Nenhum TeX foi instalado; compilação
completa permanece não verificada. Não modificar histórico para fingir sucesso.

Pendências: testar CUDA e builds do servidor; validar classifier held-out; obter
artefatos históricos; executar piloto e teacher gate; ler OQIDDM integralmente;
definir limites científicos finais antes de campanha ou extensão de shots/ruído.

Última execução de entrega (após adicionar o proxy de confiança do teacher gate):
`/private/tmp/qlcm-orch-delivery`, mesma config sintética, 20 linhas, completou e
resume pulou oito etapas. Quatro unit tests passaram em 2.065 s. Soma dos
subprocessos 44.67 s; AE8.14, teacher5.31, teacher-eval2.48, estudantes
5.70/5.95/7.27/7.18, avaliação final2.64 s. A diferença frente ao benchmark anterior
reforça que smoke inclui startup e contenção local; não usar como ETA GPU.
