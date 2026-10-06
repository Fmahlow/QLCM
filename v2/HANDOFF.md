# Handoff: quando receber “rode o piloto QLCM”

Decisão humana de 2026-10-06: investigar quando destilar um **teacher híbrido
quântico de difusão latente** para um estudante consistency híbrido compensa sob
orçamento de chamadas e medições. Comparar estudante clássico do **mesmo teacher**
e teacher com DDIM reduzido. Não afirmar prioridade: falta leitura integral de
OQIDDM. v1 é estudo preliminar com teacher clássico; não alterar seus resultados.

## Preparação no servidor GPU

Copiar este worktree completo (README, v1 e v2), entrar na raiz que contém `v2/`.
Não rodar de dentro de v1. Nenhuma conexão remota foi realizada nesta entrega.

```sh
python3 -m venv .venv-v2
.venv-v2/bin/python -m pip install -r v2/requirements.txt
.venv-v2/bin/python -m unittest v2.test_core -v
.venv-v2/bin/python -m v2.orchestrate --dry-run
```

`requirements-local.lock.txt` registra o ambiente Mac testado; não prescrever
esse lock como build CUDA. Escolher wheel torch/torchvision compatível com o
servidor e registrar `pip freeze`, GPU, driver e CUDA junto aos resultados.

## Artefatos necessários

O novo AE e teacher serão treinados do zero. Não dependem dos checkpoints v1.
Para avaliar Fashion-MNIST, recuperar do servidor o checkpoint
`results/fashion_mnist/fashion_mnist_classifier.pt` (SmallClassifier v1) e validar
no split de teste completo. Nenhum resultado de validação foi inventado.

```sh
.venv-v2/bin/python -m v2.validate_classifier --checkpoint results/fashion_mnist/fashion_mnist_classifier.pt --dataset fashion_mnist --output results/fashion_classifier_validation.json --device cuda
```

Se esse checkpoint não existir ou for incompatível, preparar e validar um
classificador independente antes de iniciar. Não treinar o classificador dentro
da avaliação gerativa. `v1/src/eval_metrics.py` contém a arquitetura e rotina
histórica de treinamento; salvá-lo numa pasta separada exige execução explícita.

## Comando exato de execução

```sh
.venv-v2/bin/python -m v2.orchestrate --config v2/gpu_pilot.json --output results/v2-pilot-fmnist --classifier results/fashion_mnist/fashion_mnist_classifier.pt --classifier-validation results/fashion_classifier_validation.json --seeds 0,1,2 --stage-hours 6 --total-hours 24
```

Para retomar, repetir **exatamente** esse comando acrescentando `--resume`.
Fingerprint inclui config completa, seeds, fontes, classifier/validation hashes,
gate e limites. Divergência exige nova pasta de saída; não sobrescrever resultados.
Retoma etapas completas com checksum, não optimizer no meio de uma época. Uma
trava advisory impede duas orquestrações na mesma pasta.
Uma etapa interrompida sem checkpoint pode ser reiniciada. Checkpoint órfão de
uma etapa que falhou após salvar exige inspeção e nova pasta; falha é explícita.
O limite total é por invocação, não uma cota cumulativa entre retomadas.

Ordem automática por seed:

1. AE clássico, 20 épocas; estimar média/desvio e normalizar latentes.
2. Teacher híbrido v-pred, 150 épocas, agenda 32 níveis.
3. Avaliar teacher DDIM em 1/2/4/8/16/32 chamadas; aplicar gate antes de estudantes.
4. Destilar estudantes hybrid/classical/no_entanglement/surrogate por 50 épocas;
   mesma representação, teacher, agenda e seed. EMA 0.999. Budget por épocas,
   registrar diferença de tempo/parâmetros em vez de chamar parameter-matched.
5. Avaliar estudantes e teacher/DDIM reduzido; escrever `report.json`.

**Defaults heurísticos, propostos e explícitos:** classificador deve ter acurácia
held-out >=0.8; teacher full-step deve ter FID-like finito, entropia >=0.8 ln10
e confiança média >=0.8 da confiança média em dados reais do mesmo classifier.
A razão normaliza esse proxy pela referência real, mas confiança não mede fidelidade.
São filtros para impedir colapso óbvio, não certificação de fidelidade nem limites
científicos aprovados. Ajustá-los deve mudar fingerprint e ser justificado.
Inspecionar grids/artefatos antes de campanha; amostras são tensores `samples_*.pt`.
Gate automático só permite piloto, nunca autoriza campanha ampla.

Critério proposto de avanço: em **cada uma das três seeds**, estudante com FID-like
até 10% pior que teacher full-step, perda de entropia <=0.1 ln10 e tempo mediano
end-to-end pelo menos 2x menor. Comparar também com DDIM no mesmo custo; se DDIM
ou estudante clássico dominar, não ampliar o estudante híbrido por padrão.
A diferença precisa superar a variabilidade da avaliação. Não é teste estatístico
de equivalência; mais amostras/seeds seriam necessárias para conclusão definitiva.
Parar em teacher gate falho, perdas não finitas, timeout ou incompatibilidade.
Investigar teacher antes de treinar estudantes novamente. Não aumentar qubits ou
épocas silenciosamente. Shots finitos/ruído são extensão futura, não implementada.

## Logs, resultados e tempos

Cada seed tem config efetiva, logs por etapa, checkpoints AE/teacher/online/EMA,
loss histories, teacher evaluation/gate, final evaluation e samples. Manifest na
raiz guarda comandos e hashes. Tempos incluem decoder, warmup=2/repetições=5,
sincronização CUDA/MPS; chamadas de circuito por amostra e lote são reportadas.
Não incluem treino nem custo de fila/hardware. Shots são `null`, nunca zero.

CPU foi testado em 64 Bars-and-Stripes, latent=4, 3 qubits, depth=2, batch=16,
uma época por estágio; resultados em `LOCAL_VALIDATION.md`. É apenas teste de
execução. Nenhuma qualidade Fashion-MNIST nova ou benchmark GPU foi obtido.
Os tempos GPU são desconhecidos: medir primeiro os logs AE/teacher e interromper
se extrapolarem o orçamento. Limite de 6h/etapa e 24h/invocação é proteção, não ETA.

## Próxima tarefa do agente

Ler este arquivo e PROTOCOL.md; verificar estado do manifest e dependências;
executar dry-run e testes; validar classifier; iniciar apenas piloto dentro dos
limites acima; monitorar logs. Ao completar, comparar qualidade/diversidade/custo
sem alegar vantagem quântica. Não executar remoto, publicar ou ampliar campanha
sem nova autorização. Recuperar artefatos históricos do servidor separadamente.
