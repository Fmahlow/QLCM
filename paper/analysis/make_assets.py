"""Build every table, figure and statistic used in paper/main.tex from v2/reports.

Run from the repository root (needs numpy, scipy, matplotlib):
    python paper/analysis/make_assets.py

Outputs: paper/generated/*.tex (tables and number macros), paper/generated/stats.json,
paper/figures/*.pdf. Nothing is trained or evaluated here; inputs are the compact JSON
reports produced by the v2 runs.
"""
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
REP = ROOT / 'v2/reports'
GEN = ROOT / 'paper/generated'
FIG = ROOT / 'paper/figures'
GEN.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

MARGIN = math.log(1.10)   # equivalence margin: same 10% as the preregistered contribution threshold
DATASETS = ['mnist', 'fashion_mnist', 'kmnist', 'pneumoniamnist']
DNAME = {'mnist': 'MNIST', 'fashion_mnist': 'Fashion-MNIST', 'kmnist': 'KMNIST', 'pneumoniamnist': 'PneumoniaMNIST'}
# Categorical slots in fixed order (reference palette, light mode); teacher is neutral ink.
COL = {'hybrid': '#2a78d6', 'surrogate': '#eb6834', 'classical_branch': '#1baf7a', 'classical': '#eda100', 'teacher': '#52514e'}
MARK = {'hybrid': 'o', 'surrogate': 's', 'classical_branch': '^', 'classical': 'D', 'teacher': 'x'}
LABEL = {'hybrid': 'hybrid (circuit)', 'surrogate': 'separable surrogate', 'classical_branch': 'classical branch',
         'classical': 'MLP only', 'teacher': 'teacher (DDIM)', 'no_entanglement': 'no entanglement'}
plt.rcParams.update({'font.size': 8, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True,
                     'grid.color': '#e4e3df', 'grid.linewidth': 0.5, 'axes.edgecolor': '#8a8984', 'lines.linewidth': 1.5,
                     'legend.frameon': False, 'pdf.fonttype': 42})

load = lambda p: json.load(open(REP / p))
fid = lambda r: r['fid_like_classifier_features'] if 'fid_like_classifier_features' in r else r['fid_like']
ms = lambda v: (float(np.mean(v)), float(np.std(v)))          # population SD, as in v2/RESULTS.md
fmt = lambda v, d=1: f"{np.mean(v):.{d}f} $\\pm$ {np.std(v):.{d}f}"
S = {}                                                        # everything written to stats.json


def ratio_ci(target, control):
    """Paired log-ratio log(target/control); mean, 90% t-interval (TOST at 5% per side), as percentages."""
    seeds = sorted(set(target) & set(control))
    r = np.array([math.log(target[s] / control[s]) for s in seeds])
    n = len(r); m = r.mean(); se = r.std(ddof=1) / math.sqrt(n) if n > 1 else float('nan')
    tc = stats.t.ppf(0.95, n - 1); lo, hi = m - tc * se, m + tc * se
    # TOST p-value: max of the two one-sided tests against +-MARGIN
    p = max(stats.t.sf((m + MARGIN) / se, n - 1), stats.t.cdf((m - MARGIN) / se, n - 1)) if se > 0 else 0.0
    pct = lambda x: 100 * (math.exp(x) - 1)
    return {'n': n, 'mean_pct': pct(m), 'lo_pct': pct(lo), 'hi_pct': pct(hi), 'equivalent': bool(lo > -MARGIN and hi < MARGIN),
            'tost_p': float(p), 'log_ratios': r.tolist()}


def pooled(pairs):
    """Pool paired log-ratios from several cells (each pair = one dataset/seed)."""
    t = {i: a for i, (a, _) in enumerate(pairs)}; c = {i: b for i, (_, b) in enumerate(pairs)}
    return ratio_ci(t, c)


def clean_log(ax, x=False):
    from matplotlib.ticker import FuncFormatter, NullFormatter, LogLocator
    f = FuncFormatter(lambda v, _: f'{v:g}')
    ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1, 2, 5)))
    ax.yaxis.set_major_formatter(f); ax.yaxis.set_minor_formatter(NullFormatter())
    if x:
        ax.xaxis.set_major_formatter(f); ax.xaxis.set_minor_formatter(NullFormatter())


def save(fig, name):
    fig.savefig(FIG / name, bbox_inches='tight'); plt.close(fig)


# ---------------------------------------------------------------- campaign (4 datasets x 3 teachers)
camp = load('campaign-datasets-teachers/report.mnist-fashion-kmnist.json')['rows'] + \
       load('campaign-datasets-teachers/report.pneumoniamnist.json')['rows']
C = defaultdict(dict)   # (dataset, teacher_kind, model, steps) -> {seed: fid}
T = defaultdict(dict)   # same -> median seconds
for r in camp:
    k = (r['dataset'], r['teacher_kind'], r['model'], r['steps'])
    C[k][r['seed']] = fid(r); T[k][r['seed']] = r['timing']['median_seconds']

rows = []
for d in DATASETS:
    for tk in ('hybrid', 'classical', 'no_entanglement'):
        cells = [fmt(list(C[(d, tk, 'teacher', s)].values())) for s in (1, 4, 32)] + \
                [fmt(list(C[(d, tk, 'classical', s)].values())) for s in (1, 4, 32)]
        rows.append(f"{DNAME[d] if tk == 'hybrid' else ''} & {LABEL[tk] if tk != 'classical' else 'classical'} & " + ' & '.join(cells) + r' \\')
    rows.append(r'\addlinespace')
(GEN / 'tab_campaign.tex').write_text('\n'.join(rows[:-1]) + '\n')

# Students of the hybrid teacher at 1 step, three kinds
rows = []
for d in DATASETS:
    cells = [fmt(list(C[(d, 'hybrid', k, 1)].values())) for k in ('hybrid', 'classical', 'surrogate')]
    tm = [1000 * np.mean(list(T[(d, 'hybrid', k, 1)].values())) for k in ('hybrid', 'classical', 'surrogate')]
    rows.append(f"{DNAME[d]} & {fmt(list(C[(d, 'hybrid', 'teacher', 32)].values()))} & " + ' & '.join(cells) +
                f" & {tm[0]:.1f} / {tm[1]:.1f} / {tm[2]:.1f}" + r' \\')
(GEN / 'tab_students.tex').write_text('\n'.join(rows) + '\n')

# Equivalence: pooled over 4 datasets x 3 seeds, plus per dataset
eq = {}
def pairs_of(a, b):
    out = []
    for d in DATASETS:
        A, B = C[a(d)], C[b(d)]
        out += [(A[s], B[s]) for s in sorted(set(A) & set(B))]
    return out
eq['teacher32_hybrid_vs_classical'] = pooled(pairs_of(lambda d: (d, 'hybrid', 'teacher', 32), lambda d: (d, 'classical', 'teacher', 32)))
eq['teacher32_hybrid_vs_noent'] = pooled(pairs_of(lambda d: (d, 'hybrid', 'teacher', 32), lambda d: (d, 'no_entanglement', 'teacher', 32)))
eq['student1_hybrid_vs_classical'] = pooled(pairs_of(lambda d: (d, 'hybrid', 'hybrid', 1), lambda d: (d, 'hybrid', 'classical', 1)))
eq['student1_hybrid_vs_surrogate'] = pooled(pairs_of(lambda d: (d, 'hybrid', 'hybrid', 1), lambda d: (d, 'hybrid', 'surrogate', 1)))
eq['student1_classical_from_hybrid_vs_from_classical_teacher'] = pooled(pairs_of(lambda d: (d, 'hybrid', 'classical', 1), lambda d: (d, 'classical', 'classical', 1)))
eq['student1_vs_teacher32'] = pooled(pairs_of(lambda d: (d, 'hybrid', 'hybrid', 1), lambda d: (d, 'hybrid', 'teacher', 32)))
for d in DATASETS:
    eq[f'teacher32_hybrid_vs_classical/{d}'] = ratio_ci(C[(d, 'hybrid', 'teacher', 32)], C[(d, 'classical', 'teacher', 32)])
    eq[f'student1_hybrid_vs_classical/{d}'] = ratio_ci(C[(d, 'hybrid', 'hybrid', 1)], C[(d, 'hybrid', 'classical', 1)])
S['equivalence_campaign'] = eq
EQLAB = {'teacher32_hybrid_vs_classical': ('Teacher, 32 steps', 'hybrid vs.\\ classical teacher'),
         'teacher32_hybrid_vs_noent': ('Teacher, 32 steps', 'hybrid vs.\\ no-entanglement teacher'),
         'student1_hybrid_vs_classical': ('Student, 1 step', 'hybrid vs.\\ classical student'),
         'student1_hybrid_vs_surrogate': ('Student, 1 step', 'hybrid vs.\\ separable surrogate'),
         'student1_classical_from_hybrid_vs_from_classical_teacher': ('Student, 1 step', 'classical student: hybrid vs.\\ classical teacher')}
rows = []
for k, (role, lab) in EQLAB.items():
    e = eq[k]
    rows.append(f"{role} & {lab} & {e['n']} & {e['mean_pct']:+.1f}\\% & [{e['lo_pct']:+.1f}\\%, {e['hi_pct']:+.1f}\\%] & {e['tost_p']:.3f} & {'yes' if e['equivalent'] else 'no'}" + r' \\')
(GEN / 'tab_equivalence.tex').write_text('\n'.join(rows) + '\n')

# Figure: FID-like vs steps, per dataset (hybrid teacher and its students)
fig, axes = plt.subplots(1, 4, figsize=(7.0, 1.9))
steps = [1, 2, 4, 8, 16, 32]
for ax, d in zip(axes, DATASETS):
    for k in ('teacher', 'hybrid', 'surrogate', 'classical'):
        y = [np.mean(list(C[(d, 'hybrid', k, s)].values())) for s in steps]
        e = [np.std(list(C[(d, 'hybrid', k, s)].values())) for s in steps]
        ax.errorbar(steps, y, yerr=e, color=COL[k], marker=MARK[k], ms=3.5, capsize=1.5, lw=1.2,
                    label=LABEL[k] if k != 'classical' else 'classical student (MLP only)')
    ax.set_xscale('log', base=2); ax.set_yscale('log'); ax.set_xticks(steps); ax.set_xticklabels(steps)
    ax.set_title(DNAME[d]); ax.set_xlabel('sampling steps'); clean_log(ax)
axes[0].set_ylabel('FID-like (lower is better)')
h, l = axes[0].get_legend_handles_labels(); fig.legend(h, l, loc='lower center', ncol=4, bbox_to_anchor=(.5, -.2), fontsize=7)
save(fig, 'fig_steps.pdf')

# ---------------------------------------------------------------- budget diagnostic (MNIST pilot)
bud = load('diag-student-budget/report.json')['rows']
pil = load('pilot-mnist/report.json')['rows']
B = defaultdict(dict)
for r in bud: B[(r['variant'], r['model'], r['steps'])][r['seed']] = fid(r)
for r in pil: B[('pilot', r['model'], r['steps'])][r['seed']] = fid(r)
rows = []
for v, lab in (('pilot', '50 / 0.999 (pilot)'), ('e50_ema095', '50 / 0.95'), ('e200_ema0999', '200 / 0.999'), ('e200_ema095', '200 / 0.95')):
    for k in ('hybrid', 'classical'):
        rows.append(f"{lab if k == 'hybrid' else ''} & {k} & " + ' & '.join(fmt(list(B[(v, k, s)].values())) for s in (1, 4, 32)) + r' \\')
(GEN / 'tab_budget.tex').write_text('\n'.join(rows) + '\n')

# ---------------------------------------------------------------- E0 branch contribution
e0 = load('extension-circuit/e0-branch-contribution.json')['results'] + load('extension-circuit/e0-branch-contribution.pneumoniamnist.json')['results']
E0 = defaultdict(lambda: defaultdict(list))
for r in e0:
    if r['teacher_kind'] != 'hybrid' or r['kind'] != 'hybrid': continue
    key = (r['dataset'], r['role'])
    for f in r['fid']:
        if f['steps'] == (32 if r['role'] == 'teacher' else 1): E0[key][f['variant']].append(f['fid_like'])
    E0[key]['ratio'].append(float(np.median([x['ratio'] for x in r['ratios']])))
    E0[key]['varying'].append(float(np.median([x['branch_varying_rms'] / x['backbone_rms'] for x in r['ratios']])))
rows = []
for d in DATASETS:
    for role, st in (('teacher', 32), ('student', 1)):
        v = E0[(d, role)]
        rows.append(f"{DNAME[d] if role == 'teacher' else ''} & {role} ({st}) & {np.mean(v['full']):.1f} & {np.mean(v['features_mean']):.1f} & {np.mean(v['readout_zeroed']):.1f} & {np.mean(v['ratio']):.2f} & {np.mean(v['varying']):.2f}" + r' \\')
(GEN / 'tab_e0.tex').write_text('\n'.join(rows) + '\n')
S['e0_varying_range'] = [min(np.mean(E0[k]['varying']) for k in E0), max(np.mean(E0[k]['varying']) for k in E0)]

# ---------------------------------------------------------------- E1 / E2 sweeps
def sweep(path):
    D, P = defaultdict(dict), {}
    for r in load(path)['rows']:
        role = 'teacher' if r['model'] == 'teacher' else 'student'
        k = (r['dataset'], r['kind'], r['hidden'], r['qubits'], r['depth'], r['reupload'], role, r['steps'])
        D[k][r['seed']] = fid(r); P[k] = r['parameters']
    return D, P
E1, P1 = sweep('extension-circuit/e1-capacity.json')
E2, P2 = sweep('extension-circuit/e2-scaling.json')
# E1 reuses the hidden=0 MNIST cells in E2 and vice versa; merge for lookups
ALL = defaultdict(dict); ALL.update(E2); ALL.update(E1); PALL = {**P2, **P1}

def verdict(t, c):
    seeds = sorted(set(t) & set(c)); gain = 1 - np.mean([t[s] for s in seeds]) / np.mean([c[s] for s in seeds])
    return bool(all(t[s] < c[s] for s in seeds) and gain >= .10), gain

forest = []   # (label, comparison, ratio dict, primary?)
crit = {'e1': {'circuit': [], 'entanglement': []}, 'e2': {'circuit': [], 'entanglement': []}}
rows = []
for d in ('mnist', 'fashion_mnist'):
    for h in (0, 4, 8, 16, 32, 128):
        g = lambda kind: ALL[(d, kind, h, 4, 2, False, 'student', 1)]
        cl = ALL.get((d, 'classical', h, 4, 2, False, 'student', 1), {})
        vc, gc = verdict(g('hybrid'), g('classical_branch')); ve, ge = verdict(g('hybrid'), g('surrogate'))
        crit['e1']['circuit'].append(vc); crit['e1']['entanglement'].append(ve)
        forest.append((f"E1 {DNAME[d]} h={h}", 'circuit', ratio_ci(g('hybrid'), g('classical_branch'))))
        forest.append((f"E1 {DNAME[d]} h={h}", 'entanglement', ratio_ci(g('hybrid'), g('surrogate'))))
        rows.append(f"{DNAME[d] if h == 0 else ''} & {h} & {PALL.get((d, 'hybrid', h, 4, 2, False, 'student', 1), '---')} & "
                    f"{fmt(list(g('hybrid').values()))} & {fmt(list(g('surrogate').values()))} & {fmt(list(g('classical_branch').values()))} & "
                    f"{fmt(list(cl.values())) if cl else '---'} & {1 - gc:.2f}{'$^\\ast$' if vc else ''} & {1 - ge:.2f}{'$^\\ast$' if ve else ''}" + r' \\')
    rows.append(r'\addlinespace')
(GEN / 'tab_e1.tex').write_text('\n'.join(rows[:-1]) + '\n')

rows = []
for q in (4, 6, 8):
    for dp in (2, 4):
        for re in (False, True):
            hy = E2[('mnist', 'hybrid', 0, q, dp, re, 'student', 1)]; su = E2[('mnist', 'surrogate', 0, q, dp, re, 'student', 1)]
            cb = E2[('mnist', 'classical_branch', 0, q, 2, False, 'student', 1)]
            th = E2[('mnist', 'hybrid', 0, q, dp, re, 'teacher', 32)]
            vc, gc = verdict(hy, cb); ve, ge = verdict(hy, su)
            crit['e2']['circuit'].append(vc); crit['e2']['entanglement'].append(ve)
            forest.append((f"E2 q={q} d={dp}{' RU' if re else ''}", 'circuit', ratio_ci(hy, cb)))
            forest.append((f"E2 q={q} d={dp}{' RU' if re else ''}", 'entanglement', ratio_ci(hy, su)))
            rows.append(f"{q} & {dp} & {'yes' if re else 'no'} & {fmt(list(hy.values()))} & {fmt(list(su.values()))} & {fmt(list(cb.values()))} & "
                        f"{fmt(list(th.values()))} & {1 - gc:.2f}{'$^\\ast$' if vc else ''} & {1 - ge:.2f}{'$^\\ast$' if ve else ''}" + r' \\')
(GEN / 'tab_e2.tex').write_text('\n'.join(rows) + '\n')
S['criteria_primary'] = {k: {kk: int(sum(vv)) for kk, vv in v.items()} | {'cells': len(v['circuit'])} for k, v in crit.items()}

# E1 figure: FID-like vs hidden width, 1-step student
fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.3), sharey=False)
for ax, d in zip(axes, ('mnist', 'fashion_mnist')):
    for k in ('hybrid', 'surrogate', 'classical_branch', 'classical'):
        hs = [h for h in (0, 2, 4, 8, 16, 32, 128) if ALL.get((d, k, h, 4, 2, False, 'student', 1))]
        y = [np.mean(list(ALL[(d, k, h, 4, 2, False, 'student', 1)].values())) for h in hs]
        e = [np.std(list(ALL[(d, k, h, 4, 2, False, 'student', 1)].values())) for h in hs]
        x = [max(h, 1) for h in hs]
        ax.errorbar(x, y, yerr=e, color=COL[k], marker=MARK[k], ms=3.5, capsize=1.5, lw=1.2, label=LABEL[k])
    ax.set_xscale('log', base=2); ax.set_yscale('log')
    ax.set_xticks([1, 2, 4, 8, 16, 32, 128]); ax.set_xticklabels(['0', '2', '4', '8', '16', '32', '128'])
    ax.set_title(DNAME[d]); ax.set_xlabel('backbone width (hidden units; 0 = no backbone)'); clean_log(ax)
axes[0].set_ylabel('1-step student FID-like'); axes[1].legend(fontsize=6.5)
save(fig, 'fig_capacity.pdf')

# Forest plot of all primary comparisons (E1 + E2), ratio hybrid/control with 90% CI
fig, axes = plt.subplots(1, 2, figsize=(7.0, 5.6), sharey=True)
labels = [f[0] for f in forest if f[1] == 'circuit']
for ax, comp, title in zip(axes, ('circuit', 'entanglement'), ('hybrid / classical branch', 'hybrid / separable surrogate')):
    items = [f for f in forest if f[1] == comp]
    for i, (lab, _, r) in enumerate(items):
        y = len(items) - 1 - i
        lo, hi, m = 1 + r['lo_pct'] / 100, 1 + r['hi_pct'] / 100, 1 + r['mean_pct'] / 100
        col = COL['hybrid'] if comp == 'circuit' else COL['surrogate']
        ax.plot([max(lo, .2), min(hi, 5)], [y, y], color=col, lw=1.2)
        ax.plot(m, y, marker='o', color=col, ms=3.5)
    ax.axvspan(1 / 1.1, 1.1, color='#e4e3df', alpha=.7, lw=0)
    ax.axvline(1, color='#52514e', lw=.8)
    ax.set_xscale('log'); ax.set_xlim(.25, 4)
    from matplotlib.ticker import FixedLocator, NullFormatter, NullLocator
    ax.xaxis.set_major_locator(FixedLocator([.25, .5, 1, 2, 4])); ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xticklabels(['0.25', '0.5', '1', '2', '4'])
    ax.set_title(title); ax.set_xlabel('FID-like ratio (<1: circuit better)')
    ax.set_yticks(range(len(items))); ax.set_yticklabels(labels[::-1], fontsize=6.5); ax.grid(axis='y', visible=False)
save(fig, 'fig_forest.pdf')
S['forest'] = [{'cell': f[0], 'comparison': f[1], **{k: v for k, v in f[2].items() if k != 'log_ratios'}} for f in forest]
S['forest_any_equivalent'] = sum(f[2]['equivalent'] for f in forest)

# ---------------------------------------------------------------- E3 shots / noise
e3 = load('extension-circuit/e3-noise-shots.h128.json')['rows'] + load('extension-circuit/e3-noise-shots.h0.json')['rows']
E3 = defaultdict(list)
for r in e3:
    E3[(r['hidden'], r['role'], r['steps'], r['depolarizing_p'], r['shots_per_evaluation'])].append(r['fid_like'])
rows = []
for h in (128, 0):
    for p, sh in ((0.0, None), (0.0, 64), (0.03, None), (0.03, 64)):
        cells = [np.mean(E3[(h, 'teacher', s, p, sh)]) for s in (4, 8, 32)] + [np.mean(E3[(h, 'student', s, p, sh)]) for s in (1, 2, 4)]
        rows.append(f"{h} & {p:g} & {sh if sh else 'exact'} & " + ' & '.join(f'{c:.1f}' for c in cells) + r' \\')
    if h == 128: rows.append(r'\addlinespace')
(GEN / 'tab_e3.tex').write_text('\n'.join(rows) + '\n')

fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.2))
for ax, h in zip(axes, (128, 0)):
    for role, stp, k in (('teacher', (4, 8, 32), 'teacher'), ('student', (1, 2, 4), 'hybrid')):
        for p, ls in ((0.0, '-'), (0.03, '--')):
            # FID-like vs total shots per sample at the 64-shot setting, both noise levels
            x = [s * 64 for s in stp]; y = [np.mean(E3[(h, role, s, p, 64)]) for s in stp]
            ax.plot(x, y, ls=ls, marker=MARK[k], ms=3.5, color=COL[k], label=f"{'teacher' if role == 'teacher' else 'student'}, p={p:g}")
    ax.set_xscale('log', base=2); ax.set_yscale('log'); ax.set_xlabel('total shots per sample (steps x 64)')
    ax.set_title('with backbone (hidden 128)' if h else 'no backbone (hidden 0)'); clean_log(ax)
axes[0].set_ylabel('FID-like'); axes[0].legend(fontsize=6.5)
save(fig, 'fig_shots.pdf')

# ---------------------------------------------------------------- E4a / E4b
e4 = load('extension-circuit/e4a-step-degradation.json')['latent_energy']
E4 = defaultdict(list)
for r in e4: E4[(r['dataset'], r['model'], r['steps'])].append(r['fraction_above_4'])
rows = []
for d in ('mnist', 'pneumoniamnist'):
    for m in ('teacher', 'classical'):
        rows.append(f"{DNAME[d] if m == 'teacher' else ''} & {'teacher DDIM' if m == 'teacher' else 'classical student'} & " +
                    ' & '.join(f"{np.mean(E4[(d, m, s)]):.3f}" for s in (1, 4, 8, 16, 32)) + r' \\')
(GEN / 'tab_e4a.tex').write_text('\n'.join(rows) + '\n')

e4b = load('extension-circuit/e4b-kmnist-latent.json')
R = defaultdict(list); G = defaultdict(list)
for r in e4b['reconstruction']: R[(r['model'], r['latent'])].append(fid(r))
for r in e4b['generation']: G[(r['latent'], r['model'], r['steps'])].append(fid(r))
S['e4b_floor'] = float(np.mean(R[('real test images', None)])) if R[('real test images', None)] else None
for s_ in (1, 4, 32):   # latent 16 = the campaign pipeline (hybrid teacher, classical student)
    G[(16, 'teacher', s_)] = list(C[('kmnist', 'hybrid', 'teacher', s_)].values())
    G[(16, 'classical', s_)] = list(C[('kmnist', 'hybrid', 'classical', s_)].values())
rows = []
for lat in (16, 32, 64):
    rec = [v for (m, l), v in R.items() if l == lat]
    rec = rec[0] if rec else []
    rows.append(f"{lat} & {fmt(rec) if rec else '---'} & {fmt(G[(lat, 'teacher', 32)])} & {fmt(G[(lat, 'teacher', 4)])} & {fmt(G[(lat, 'classical', 1)])} & {fmt(G[(lat, 'classical', 4)])}" + r' \\')
(GEN / 'tab_e4b.tex').write_text('\n'.join(rows) + '\n')

# ---------------------------------------------------------------- classifier validation
cv = {d: json.load(open(p)) for d, p in (('mnist', REP / 'pilot-mnist/mnist_classifier.validation.json'),
      ('fashion_mnist', REP / 'campaign-datasets-teachers/fashion_mnist_classifier.validation.json'),
      ('kmnist', REP / 'campaign-datasets-teachers/kmnist_classifier.validation.json'),
      ('pneumoniamnist', REP / 'campaign-datasets-teachers/pneumoniamnist_classifier.validation.json'))}
S['classifier_accuracy'] = {d: v['test_accuracy'] for d, v in cv.items()}

(GEN / 'stats.json').write_text(json.dumps(S, indent=1) + '\n')
print(json.dumps({k: S[k] for k in ('criteria_primary', 'forest_any_equivalent', 'e0_varying_range', 'classifier_accuracy', 'e4b_floor')}, indent=1))
for k, e in eq.items(): print(f"{k:60s} {e['mean_pct']:+6.1f}% [{e['lo_pct']:+6.1f}, {e['hi_pct']:+6.1f}] p={e['tost_p']:.3f} eq={e['equivalent']}")
