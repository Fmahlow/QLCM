# Pendências antes da submissão

Origem: revisão simulada de 2026-11-03 (recomendação: major revision). Todos os
itens de texto e de análise local já foram aplicados em `main.tex` e
`analysis/make_assets.py`. O que segue precisa do servidor GPU (checkpoints e
amostras ficam em `results/`, que não está versionado) ou de ação humana.

## Precisa de GPU (em ordem de prioridade)

1. **Replicar MNIST com seeds novas (3 a 5)** usando o orçamento já fixado
   (200 épocas, EMA 0,95), para remover o viés de seleção: o orçamento foi
   escolhido pelo FID-like de teste nas mesmas seeds que a campanha MNIST reutiliza.
2. **Controle positivo.** Perturbar o ramo de forma conhecida (por exemplo, escalar
   ou embaralhar a saída do bloco, ou trocar parte da função por ruído) e mostrar
   que o pipeline + FID-like detecta uma mudança da ordem de 10%. Sem isso, a
   equivalência em h=128 tem pouca sensibilidade demonstrada.
3. **Pisos e robustez da métrica.**
   - Real-vs-real e reconstrução do AE para MNIST, Fashion-MNIST e PneumoniaMNIST
     (hoje só há para KMNIST).
   - Mais amostras (2.000 a 5.000) nas comparações principais.
   - Precision/recall (Kynkäänniemi et al., 2019).
   - Uma métrica no espaço latente (Fréchet ou MMD entre latentes gerados e
     latentes codificados do teste), que mede o denoiser sem passar pelo decoder
     e pelo classificador.
4. **Baselines de poucos passos:** DPM-Solver++ (ou Heun/UniPC) com 4 a 8 avaliações
   para o teacher; opcionalmente progressive distillation ou regressão direta do
   endpoint (Luhman & Luhman, 2021). Hoje a vantagem da destilação só é mostrada
   contra DDIM, e o texto já está restrito a isso.
5. **Readout com correladores** (⟨Z_i Z_j⟩) ou rank de readout fixo ao variar q,
   para testar se o entrelaçamento pode importar nesta arquitetura.
6. **Grade de amostras** (figura qualitativa) para teacher, estudante híbrido e
   estudante clássico em cada dataset: as amostras só existem no servidor.
7. Opcional: mais seeds nas células E1/E2 mais ruidosas, para estreitar os
   intervalos por célula (o poder do critério por célula é baixo; ver Tabela 14).

## Precisa de ação humana

- Ler o texto completo do OQIDDM (Zhang et al., FGCS 173, 107875, 2025) e
  confirmar os autores no `references.bib` (vieram de citação secundária).
  Ajustar a afirmação de novidade em Related Work conforme o que ele faz.
- Afiliação, agradecimentos e escolha da revista (formato).
- O Overleaf tem uma branch própria (`overleaf-2026-10-06-1755`) que aponta para o
  paper antigo em `v1/`; o paper novo está em `paper/`.
