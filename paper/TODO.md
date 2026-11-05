# Pendências antes da submissão

Origem: revisão simulada de 2026-11-03 (major revision). Os itens de GPU foram executados em
2026-11-04 (pré-registro em `v2/PROTOCOL.md`, resultados em `v2/RESULTS.md` e
`v2/reports/review-2026-11/`) e o texto foi revisado em duas rodadas de revisão simulada
(`paper/reviews/`).

## Itens de GPU (2026-11-03) — estado

1. Replicar MNIST com seeds novas: **feito** (E5, seeds 3-7). As equivalências replicam; o gap do
   estudante de 1 passo também (~9-13%); o resultado de 4 passos não.
2. Controle positivo: **feito** (E6 em h=128: não detectável, o ramo pesa pouco; R2 em h=0/4/16:
   44/72 perturbações detectadas; R3 entre modelos treinados independentemente).
3. Pisos e robustez da métrica: **feito** (E7/R4: pisos com regra única, 5.000 amostras, P/R,
   Fréchet e MMD latentes; R1 reavaliou as varreduras E9/E10 com o mesmo protocolo).
4. Baselines de poucos passos: **feito** (E8: DPM-Solver++(2M), Heun, regressão do endpoint com
   10k e 100k pares). Progressive distillation não foi feita (era opcional).
5. Readout com correladores: **feito** (E9). Rank de readout fixo ao variar q: não testado.
6. Grade de amostras: **feita** (`figures/fig_samples.pdf`).
7. Mais seeds: **feito para o E1** (E10, 6 seeds). O E2 continua com 3 seeds.

## Ainda pendente

- [humano] Ler o texto completo do OQIDDM (Zhang et al., FGCS 173, 107875, 2025), confirmar os
  autores no `references.bib` e ajustar a afirmação de novidade em Related Work.
- [humano] Afiliação, agradecimentos e escolha da revista (formato).
- [humano] O Overleaf tem uma branch própria (`overleaf-2026-10-06-1755`) que aponta para o paper
  antigo em `v1/`; o paper novo está em `paper/`.
- [ambiente] Compilar `main.tex` (não há LaTeX neste servidor; o texto passou por checagem
  estática de referências, citações e colunas das tabelas).
- Opcional: mais seeds no E2 (células com re-uploading são as mais ruidosas); solvers sob shots
  finitos; rank de readout fixo.
