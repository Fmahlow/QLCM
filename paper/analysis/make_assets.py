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


def cluster_ci(cells):
    """Cell- (or dataset-) clustered interval: mean paired log-ratio per cell, t-interval across cells."""
    means = np.array([np.mean([math.log(t[s] / c[s]) for s in sorted(set(t) & set(c))]) for t, c in cells])
    k = len(means); m = means.mean(); se = means.std(ddof=1) / math.sqrt(k); tc = stats.t.ppf(0.95, k - 1)
    pct = lambda x: 100 * (math.exp(x) - 1)
    return {'k': k, 'mean_pct': pct(m), 'lo_pct': pct(m - tc * se), 'hi_pct': pct(m + tc * se),
            'equivalent': bool(m - tc * se > -MARGIN and m + tc * se < MARGIN)}


def ptxt(p): return '$<$0.001' if p < 0.001 else f'{p:.3f}'


def ci_txt(e): return f"[{e['lo_pct']:+.1f}\\%, {e['hi_pct']:+.1f}\\%]"


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
EQKEYS = {'teacher32_hybrid_vs_classical': ((lambda d: (d, 'hybrid', 'teacher', 32)), (lambda d: (d, 'classical', 'teacher', 32))),
          'teacher32_hybrid_vs_noent': ((lambda d: (d, 'hybrid', 'teacher', 32)), (lambda d: (d, 'no_entanglement', 'teacher', 32))),
          'student1_hybrid_vs_classical': ((lambda d: (d, 'hybrid', 'hybrid', 1)), (lambda d: (d, 'hybrid', 'classical', 1))),
          'student1_hybrid_vs_surrogate': ((lambda d: (d, 'hybrid', 'hybrid', 1)), (lambda d: (d, 'hybrid', 'surrogate', 1))),
          'student1_classical_from_hybrid_vs_from_classical_teacher': ((lambda d: (d, 'hybrid', 'classical', 1)), (lambda d: (d, 'classical', 'classical', 1))),
          'student1_vs_teacher32': ((lambda d: (d, 'hybrid', 'hybrid', 1)), (lambda d: (d, 'hybrid', 'teacher', 32)))}
for k, (a, b) in EQKEYS.items():
    eq[k + '/by_dataset'] = cluster_ci([(C[a(d)], C[b(d)]) for d in DATASETS])
    exk = []
    for d in DATASETS:
        if d != 'kmnist': exk += [(C[a(d)][x], C[b(d)][x]) for x in sorted(C[a(d)])]
    eq[k + '/excl_kmnist'] = pooled(exk)
    eq[k + '/max_abs_pct'] = max(abs(eq[k]['lo_pct']), abs(eq[k]['hi_pct']), abs(eq[k + '/by_dataset']['lo_pct']), abs(eq[k + '/by_dataset']['hi_pct']))
rows = []
for k, (role, lab) in EQLAB.items():
    e, dd, ex = eq[k], eq[k + '/by_dataset'], eq[k + '/excl_kmnist']
    rows.append(f"{role} & {lab} & {e['mean_pct']:+.1f}\\% & {ci_txt(e)} & {ptxt(e['tost_p'])} & {ci_txt(dd)} & {ci_txt(ex)}" + r' \\')
(GEN / 'tab_equivalence.tex').write_text('\n'.join(rows) + '\n')
# Student (1 step) vs 32-step teacher, per dataset: ratio of geometric means and seeds within 10%
rows = []
for d in DATASETS:
    e = ratio_ci(C[(d, 'hybrid', 'hybrid', 1)], C[(d, 'hybrid', 'teacher', 32)])
    e4 = ratio_ci(C[(d, 'hybrid', 'hybrid', 4)], C[(d, 'hybrid', 'teacher', 32)])
    within = lambda st: sum(C[(d, 'hybrid', 'hybrid', st)][x] <= 1.1 * C[(d, 'hybrid', 'teacher', 32)][x] for x in range(3))
    rows.append(f"{DNAME[d]} & {e['mean_pct']:+.1f}\\% & {ci_txt(e)} & {within(1)}/3 & {e4['mean_pct']:+.1f}\\% & {within(4)}/3" + r' \\')
    eq[f'student1_vs_teacher32/{d}'] = e; eq[f'student4_vs_teacher32/{d}'] = e4
(GEN / 'tab_student_teacher.tex').write_text('\n'.join(rows) + '\n')

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
        forest.append((f"E1 {DNAME[d]} h={h}" + (' (= E2 q=4 d=2)' if (d, h) == ('mnist', 0) else ''), 'circuit', ratio_ci(g('hybrid'), g('classical_branch'))))
        forest.append((f"E1 {DNAME[d]} h={h}" + (' (= E2 q=4 d=2)' if (d, h) == ('mnist', 0) else ''), 'entanglement', ratio_ci(g('hybrid'), g('surrogate'))))
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
            if not (q == 4 and dp == 2 and not re):   # same runs as E1 MNIST h=0
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
plt.close(fig)  # superseded by the parameter-axis version below

# Forest plot of all primary comparisons (E1 + E2), ratio hybrid/control with 90% CI
fig, axes = plt.subplots(1, 2, figsize=(7.0, 5.4), sharey=True)
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

# ---------------------------------------------------------------- sweep robustness (added after review)
# Unique primary cells (E1 MNIST h=0 is the same run as E2 q=4 d=2 without re-uploading).
st1 = lambda d, kind, h, q=4, dp=2, re=False: ALL[(d, kind, h, q, dp, re, 'student', 1)]
e1cells = [(d, h) for d in ('mnist', 'fashion_mnist') for h in (0, 4, 8, 16, 32, 128)]
e2cells = [(q, dp, re) for q in (4, 6, 8) for dp in (2, 4) for re in (False, True) if not (q == 4 and dp == 2 and not re)]
fam = {
 'E1 hybrid/surrogate (12 cells)': [(st1(d, 'hybrid', h), st1(d, 'surrogate', h)) for d, h in e1cells],
 'E1 hybrid/classical branch, $h\\ge16$ (6)': [(st1(d, 'hybrid', h), st1(d, 'classical_branch', h)) for d, h in e1cells if h >= 16],
 'E1 hybrid/classical branch, $h\\le8$ (6)': [(st1(d, 'hybrid', h), st1(d, 'classical_branch', h)) for d, h in e1cells if h <= 8],
 'E2 hybrid/surrogate, no RU (6)': [(st1('mnist', 'hybrid', 0, q, dp, re), st1('mnist', 'surrogate', 0, q, dp, re)) for q, dp, re in [(4, 2, False)] + e2cells if not re],
 'E2 hybrid/surrogate, RU (6)': [(st1('mnist', 'hybrid', 0, q, dp, re), st1('mnist', 'surrogate', 0, q, dp, re)) for q, dp, re in e2cells if re],
 'E2 hybrid/classical branch, no RU (6)': [(st1('mnist', 'hybrid', 0, q, dp, re), st1('mnist', 'classical_branch', 0, q, 2, False)) for q, dp, re in [(4, 2, False)] + e2cells if not re],
 'E2 hybrid/classical branch, RU (6)': [(st1('mnist', 'hybrid', 0, q, dp, re), st1('mnist', 'classical_branch', 0, q, 2, False)) for q, dp, re in e2cells if re],
}
rows = []; S['sweep_families'] = {}
for lab, cells in fam.items():
    e = cluster_ci(cells); S['sweep_families'][lab] = e
    rows.append(f"{lab} & {e['mean_pct']:+.1f}\\% & {ci_txt(e)} & {'yes' if e['equivalent'] else 'no'}" + r' \\')
(GEN / 'tab_sweep_pooled.tex').write_text('\n'.join(rows) + '\n')
S['unique_primary_cells'] = len(e1cells) + len(e2cells)

# Preregistered parameter frontier: hybrid vs best MLP-only with <= registered parameters (one-step student)
rows = []; S['frontier'] = []
for d in ('mnist', 'fashion_mnist'):
    for h in (0, 4, 8, 16, 32, 128):
        kh = (d, 'hybrid', h, 4, 2, False, 'student', 1); ph = PALL[kh]; fh = np.mean(list(ALL[kh].values()))
        opts = [(np.mean(list(ALL[k].values())), k[2], PALL[k]) for k in ALL if k[0] == d and k[1] == 'classical' and k[6:] == ('student', 1) and PALL[k] <= ph]
        best = min(opts) if opts else None
        S['frontier'].append({'dataset': d, 'h': h, 'params': ph, 'hybrid': fh, 'best_mlp': best})
        rows.append(f"{DNAME[d] if h == 0 else ''} & {h} & {ph} & {fh:.1f} & " + (f"{best[1]} ({best[2]}) & {best[0]:.1f} & {'hybrid' if fh < best[0] else 'MLP'}" if best else '--- & --- & ---') + r' \\')
    rows.append(r'\addlinespace')
(GEN / 'tab_frontier.tex').write_text('\n'.join(rows[:-1]) + '\n')

# Capacity figure against registered parameters
fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.3))
for ax, d in zip(axes, ('mnist', 'fashion_mnist')):
    for k in ('hybrid', 'surrogate', 'classical_branch', 'classical'):
        hs = [h for h in (0, 2, 4, 8, 16, 32, 128) if ALL.get((d, k, h, 4, 2, False, 'student', 1))]
        x = [PALL[(d, k, h, 4, 2, False, 'student', 1)] for h in hs]
        y = [np.mean(list(ALL[(d, k, h, 4, 2, False, 'student', 1)].values())) for h in hs]
        e = [np.std(list(ALL[(d, k, h, 4, 2, False, 'student', 1)].values())) for h in hs]
        ax.errorbar(x, y, yerr=e, color=COL[k], marker=MARK[k], ms=3.5, capsize=1.5, lw=1.2, label=LABEL[k])
    ax.set_xscale('log'); ax.set_yscale('log'); clean_log(ax, x=True)
    ax.set_title(DNAME[d]); ax.set_xlabel('registered parameters of the network')
axes[0].set_ylabel('1-step student FID-like'); axes[1].legend(fontsize=6.5)
save(fig, 'fig_capacity.pdf')

# Power of the preregistered per-cell rule (3 seeds, all better and mean >= 10% lower), by simulation
rng = np.random.default_rng(0)
sds = [0.5 * (math.log(1 + f['hi_pct'] / 100) - math.log(1 + f['lo_pct'] / 100)) / stats.t.ppf(.95, 2) * math.sqrt(3) for f in S['forest']]
S['cell_logratio_sd'] = {'median': float(np.median(sds)), 'q25': float(np.percentile(sds, 25)), 'q75': float(np.percentile(sds, 75))}
def power(effect, sd, n=200000):
    x = rng.normal(math.log(1 - effect), sd, size=(n, 3))   # log(hybrid/control) per seed
    return float(np.mean((x < 0).all(1) & (np.exp(x).mean(1) <= 0.9)))   # ratio-of-means approx via mean of ratios
S['power'] = {f'{int(e*100)}%': {'median_sd': power(e, S['cell_logratio_sd']['median']), 'sd_0.20': power(e, 0.20)} for e in (0.0, 0.10, 0.15, 0.20, 0.30)}
rows = [f"{k.replace('%', chr(92) + '%')} & {v['median_sd']:.2f} & {v['sd_0.20']:.2f}" + r' \\' for k, v in S['power'].items()]
(GEN / 'tab_power.tex').write_text('\n'.join(rows) + '\n')

# Label entropy (diversity) of the campaign models
LE = defaultdict(dict)
for r in camp: LE[(r['dataset'], r['teacher_kind'], r['model'], r['steps'])][r['seed']] = r['label_entropy']
rows = []
for d in DATASETS:
    cells = [LE[(d, 'hybrid', 'teacher', 32)]] + [LE[(d, 'hybrid', k, 1)] for k in ('hybrid', 'classical', 'surrogate')] + [LE[(d, tk, 'teacher', 32)] for tk in ('classical', 'no_entanglement')]
    rows.append(f"{DNAME[d]} & " + ' & '.join(f"{np.mean(list(c.values())):.2f}" for c in cells) + r' \\')
(GEN / 'tab_entropy.tex').write_text('\n'.join(rows) + '\n')

# E0 per-seed relative changes, all branch kinds
E0R = defaultdict(lambda: defaultdict(list))
for r in e0:
    st = 32 if r['role'] == 'teacher' else 1
    f = {x['variant']: x['fid_like'] for x in r['fid'] if x['steps'] == st}
    key = (r['dataset'], r['role'], r['kind'])
    E0R[key]['fm'].append(100 * (f['features_mean'] / f['full'] - 1)); E0R[key]['rz'].append(100 * (f['readout_zeroed'] / f['full'] - 1))
rows = []
for d in DATASETS:
    first = True
    for role, kinds in (('teacher', ('hybrid', 'no_entanglement')), ('student', ('hybrid', 'surrogate'))):
        for k in kinds:
            v = E0R[(d, role, k)]
            if not v['fm']: continue
            rows.append(f"{DNAME[d] if first else ''} & {role} & {LABEL[k]} & {np.mean(v['fm']):+.1f} [{min(v['fm']):+.1f}, {max(v['fm']):+.1f}] & {np.mean(v['rz']):+.0f} [{min(v['rz']):+.0f}, {max(v['rz']):+.0f}]" + r' \\')
            first = False
(GEN / 'tab_e0_relative.tex').write_text('\n'.join(rows) + '\n')
S['e0_features_mean_range_pct'] = [min(min(v['fm']) for v in E0R.values()), max(max(v['fm']) for v in E0R.values())]

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

# ================================================================ review experiments (2026-11-04 preregistration)
RV = 'review-2026-11/'
def pct_(x): return 100 * (math.exp(x) - 1)
def sig(e): return e['lo_pct'] > 0 or e['hi_pct'] < 0          # 90% interval excludes 0
def detect(e): return sig(e) and abs(e['mean_pct']) >= 10       # preregistered E6 detection rule
def pm(e): return f"{e['mean_pct']:+.1f}"

# ---- retrained campaign vs original (appendix) and E5 replication
def camp_dict(rows_):
    D_ = defaultdict(dict)
    for r in rows_: D_[(r['dataset'], r['teacher_kind'], r['model'], r['steps'])][r['seed']] = fid(r)
    return D_
CR = camp_dict(load(RV + 'campaign-retrained.json')['rows'])
REPL = camp_dict(load(RV + 'e5-mnist-replication.json')['rows'])
rows = []; S['reproduction'] = {}
for d in DATASETS:
    for lab, key in (('teacher (32)', ('hybrid', 'teacher', 32)), ('hybrid student (1)', ('hybrid', 'hybrid', 1)),
                     ('classical student (1)', ('hybrid', 'classical', 1)), ('classical teacher (32)', ('classical', 'teacher', 32))):
        o, n = C[(d,) + key], CR[(d,) + key]; e = ratio_ci(n, o)
        S['reproduction'][f'{d}/{lab}'] = e
        rows.append(f"{DNAME[d] if lab.startswith('teacher') else ''} & {lab} & {fmt(list(o.values()))} & {fmt(list(n.values()))} & {pm(e)}\\%" + r' \\')
    rows.append(r'\addlinespace')
(GEN / 'tab_repro.tex').write_text('\n'.join(rows[:-1]) + '\n')
S['reproduction_max_abs_pct'] = max(abs(e['mean_pct']) for e in S['reproduction'].values())

m_ = lambda C_, tk, k, st: C_[('mnist', tk, k, st)]
E5 = {}
E5['student1_vs_teacher32'] = ratio_ci(m_(REPL, 'hybrid', 'hybrid', 1), m_(REPL, 'hybrid', 'teacher', 32))
E5['student4_vs_teacher32'] = ratio_ci(m_(REPL, 'hybrid', 'hybrid', 4), m_(REPL, 'hybrid', 'teacher', 32))
E5['within10_1'] = int(sum(m_(REPL, 'hybrid', 'hybrid', 1)[s] <= 1.1 * m_(REPL, 'hybrid', 'teacher', 32)[s] for s in m_(REPL, 'hybrid', 'teacher', 32)))
E5['within10_4'] = int(sum(m_(REPL, 'hybrid', 'hybrid', 4)[s] <= 1.1 * m_(REPL, 'hybrid', 'teacher', 32)[s] for s in m_(REPL, 'hybrid', 'teacher', 32)))
COMP5 = {'Teacher (32): hybrid vs.\\ classical': (('hybrid', 'teacher', 32), ('classical', 'teacher', 32)),
         'Teacher (32): hybrid vs.\\ no entanglement': (('hybrid', 'teacher', 32), ('no_entanglement', 'teacher', 32)),
         'Student (1): hybrid vs.\\ classical': (('hybrid', 'hybrid', 1), ('hybrid', 'classical', 1)),
         'Student (1): hybrid vs.\\ surrogate': (('hybrid', 'hybrid', 1), ('hybrid', 'surrogate', 1)),
         'Student (1) vs.\\ teacher (32)': (('hybrid', 'hybrid', 1), ('hybrid', 'teacher', 32)),
         'Student (4) vs.\\ teacher (32)': (('hybrid', 'hybrid', 4), ('hybrid', 'teacher', 32))}
rows = []
for lab, (a, b) in COMP5.items():
    e = ratio_ci(m_(REPL, *a), m_(REPL, *b))
    both = {**{s: m_(REPL, *a)[s] for s in m_(REPL, *a)}, **{s: m_(CR, *a)[s] for s in m_(CR, *a)}}
    bothc = {**{s: m_(REPL, *b)[s] for s in m_(REPL, *b)}, **{s: m_(CR, *b)[s] for s in m_(CR, *b)}}
    e8 = ratio_ci(both, bothc); E5[lab] = e; E5[lab + '/8seeds'] = e8
    rows.append(f"{lab} & {pm(e)}\\% & {ci_txt(e)} & {'yes' if e['equivalent'] else 'no'} & {pm(e8)}\\% & {ci_txt(e8)} & {'yes' if e8['equivalent'] else 'no'}" + r' \\')
(GEN / 'tab_e5.tex').write_text('\n'.join(rows) + '\n')
E5['means'] = {k: ms(list(m_(REPL, *k_).values())) for k, k_ in (('teacher32', ('hybrid', 'teacher', 32)), ('hybrid1', ('hybrid', 'hybrid', 1)),
               ('classical1', ('hybrid', 'classical', 1)), ('hybrid4', ('hybrid', 'hybrid', 4)))}
S['e5'] = E5

# ---- review evaluation: 5,000 samples, floors, P/R, latent metrics, positive control, solvers
RE = load(RV + 'e6-e7-e8-review-eval.json')['results']
V = defaultdict(dict)
for res in RE:
    for x in res['rows']:
        V[(x['model'], x['sampler'], x['steps'], x.get('teacher_kind'), x.get('perturbation'), x.get('value'))][(res['dataset'], res['seed'])] = x
def rv(key, d, metric='fid_like'): return {s: V[key][(d, s)][metric] for s in range(3) if (d, s) in V[key] and metric in V[key][(d, s)]}
def rvall(key, metric='fid_like'):
    out = {}
    for i, d in enumerate(DATASETS):
        for s, v in rv(key, d, metric).items(): out[(i, s)] = v
    return out
T32k = ('teacher-hybrid', 'ddim', 32, 'hybrid', None, None); S1k = ('student-hybrid', 'consistency', 1, 'hybrid', None, None)
S4k = ('student-hybrid', 'consistency', 4, 'hybrid', None, None)
key_ = lambda m, smp, st, tk: (m, smp, st, tk, None, None)
FLOORS = [('real unseen train (floor, post hoc)', 'Real, held-out train'), ('AE reconstruction of unseen train (R4)', 'AE recon., held-out train'),
          ('real (floor)', 'Real, test (prereg.)'), ('AE reconstruction (floor)', 'AE recon., test (prereg.)'), ('AE reconstruction of reference (R4)', 'AE recon., reference')]
rows = []; S['floors'] = {}
for d in DATASETS:
    first = True
    for lab, k in [(l, (m, '-', 0, None, None, None)) for m, l in FLOORS] + [('Teacher (32)', T32k), ('Hybrid student (1)', S1k), ('Teacher DDIM (4)', key_('teacher-hybrid', 'ddim', 4, 'hybrid'))]:
        f_ = rv(k, d)
        if not f_: continue
        if d == 'pneumoniamnist': lab = {'Real, test (prereg.)': 'Real, validation (prereg.)', 'AE recon., test (prereg.)': 'AE recon., validation (prereg.)'}.get(lab, lab)
        cells = [ms(list(f_.values()))[0]] + [np.mean(list(rv(k, d, mm).values())) for mm in ('precision', 'recall')]
        lf = rv(k, d, 'latent_frechet'); lm = rv(k, d, 'latent_mmd')
        S['floors'][f'{d}/{lab}'] = cells + ([np.mean(list(lf.values())), np.mean(list(lm.values()))] if lf else [])
        tail = f" & {np.mean(list(lf.values())):.2f} & {1000 * np.mean(list(lm.values())):.1f}" if lf else ' & --- & ---'
        rows.append(f"{DNAME[d] if first else ''} & {lab} & {cells[0]:.1f} & {cells[1]:.2f} & {cells[2]:.2f}" + tail + r' \\'); first = False
    rows.append(r'\addlinespace')
(GEN / 'tab_floors.tex').write_text('\n'.join(rows[:-1]) + '\n')

COMP7 = {'teacher32_hybrid_vs_classical': ('Teacher (32)', 'hybrid vs.\\ classical teacher', T32k, key_('teacher-classical', 'ddim', 32, 'classical')),
         'teacher32_hybrid_vs_noent': ('Teacher (32)', 'hybrid vs.\\ no-entanglement teacher', T32k, key_('teacher-no_entanglement', 'ddim', 32, 'no_entanglement')),
         'student1_hybrid_vs_classical': ('Student (1)', 'hybrid vs.\\ classical student', S1k, key_('student-classical', 'consistency', 1, 'hybrid')),
         'student1_hybrid_vs_surrogate': ('Student (1)', 'hybrid vs.\\ separable surrogate', S1k, key_('student-surrogate', 'consistency', 1, 'hybrid')),
         'student1_classical_teacher': ('Student (1)', 'classical student: hybrid vs.\\ classical teacher', key_('student-classical', 'consistency', 1, 'hybrid'), key_('student-classical', 'consistency', 1, 'classical')),
         'student1_vs_teacher32': ('Student (1)', 'hybrid student vs.\\ its teacher (32)', S1k, T32k),
         'student4_vs_teacher32': ('Student (4)', 'hybrid student vs.\\ its teacher (32)', S4k, T32k)}
rows = []; S['e7_equivalence'] = {}
for k, (role, lab, a, b) in COMP7.items():
    ef = ratio_ci(rvall(a), rvall(b)); el = ratio_ci(rvall(a, 'latent_frechet'), rvall(b, 'latent_frechet'))
    ed = cluster_ci([(rv(a, d), rv(b, d)) for d in DATASETS])
    dp = 100 * np.mean([rvall(a, 'precision')[x] - rvall(b, 'precision')[x] for x in rvall(a)])
    dr = 100 * np.mean([rvall(a, 'recall')[x] - rvall(b, 'recall')[x] for x in rvall(a)])
    dm = 1000 * np.array([rvall(a, 'latent_mmd')[x] - rvall(b, 'latent_mmd')[x] for x in rvall(a)])
    tcm = stats.t.ppf(.95, len(dm) - 1) * dm.std(ddof=1) / math.sqrt(len(dm))
    S['e7_equivalence'][k] = {'fid': ef, 'fid_by_dataset': ed, 'latent_frechet': el, 'precision_pp': dp, 'recall_pp': dr, 'mmd_x1000': [float(dm.mean()), float(dm.mean() - tcm), float(dm.mean() + tcm)]}
    rows.append(f"{role} & {lab} & {pm(ef)}\\% & {ci_txt(ef)} & {ci_txt(ed)} & {pm(el)}\\% & {ci_txt(el)} & {dm.mean():+.2f} [{dm.mean() - tcm:+.2f}, {dm.mean() + tcm:+.2f}] & {dp:+.1f} & {dr:+.1f}" + r' \\')
(GEN / 'tab_e7.tex').write_text('\n'.join(rows) + '\n')
for d in DATASETS:
    S['e7_equivalence'][f'student1_vs_teacher32/{d}'] = ratio_ci(rv(S1k, d), rv(T32k, d))

# Positive control (E6). Ratios against the unperturbed model of the same seed; 3 seeds per dataset.
PERT = [('Branch: input-dependent part removed ($a=0$)', 'perturb_scale', 'scale', 0.0), ('Branch: input-dependent part $\\times2$', 'perturb_scale', 'scale', 2.0),
        ('Branch: input-dependent part $\\times3$', 'perturb_scale', 'scale', 3.0), ('Branch: permuted across samples', 'perturb_shuffle', 'shuffle', None),
        ('Latent noise, $s=0.1$', 'perturb_latent_noise', 'latent_noise', 0.1), ('Latent noise, $s=0.4$', 'perturb_latent_noise', 'latent_noise', 0.4),
        ('Branch removed (readout zeroed)$^\\dagger$', 'perturb_readout_zeroed', 'readout_zeroed', None),
        ('Contamination 2\\%$^\\dagger$', 'perturb_contamination', 'contamination', 0.02), ('Contamination 5\\%$^\\dagger$', 'perturb_contamination', 'contamination', 0.05),
        ('Contamination 10\\%$^\\dagger$', 'perturb_contamination', 'contamination', 0.1)]
rows = []; S['e6'] = {}
for lab, smp, pert, val in PERT:
    cells = []
    for base, st in ((T32k, 32), (S1k, 1)):
        for d in DATASETS:
            k = (base[0], smp, st, 'hybrid', pert, val); e = ratio_ci(rv(k, d), rv(base, d))
            S['e6'][f'{base[0]}/{pert}/{val}/{d}'] = e
            cells.append(f"{e['mean_pct']:+.0f}{'$^\\ast$' if detect(e) else ('$^\\circ$' if sig(e) else '')}")
    rows.append(f"{lab} & " + ' & '.join(cells) + r' \\')
(GEN / 'tab_e6.tex').unlink(missing_ok=True)   # superseded by tab_e6_full.tex
rows = []
for lab, smp, pert, vals in (('Scale $a$', 'perturb_scale', 'scale', (0.0, 0.5, 1.5, 2.0, 3.0)), ('Permuted', 'perturb_shuffle', 'shuffle', (None,)),
                              ('Readout zeroed$^\\dagger$', 'perturb_readout_zeroed', 'readout_zeroed', (None,)), ('Latent noise $s$', 'perturb_latent_noise', 'latent_noise', (0.05, 0.1, 0.2, 0.4)),
                              ('Contamination$^\\dagger$', 'perturb_contamination', 'contamination', (0.02, 0.05, 0.1, 0.2))):
    for v in vals:
        cells = []
        for base, st in ((T32k, 32), (S1k, 1)):
            for d in DATASETS:
                e = ratio_ci(rv((base[0], smp, st, 'hybrid', pert, v), d), rv(base, d))
                cells.append(f"{e['mean_pct']:+.0f}{'$^\\ast$' if detect(e) else ('$^\\circ$' if sig(e) else '')}")
        rows.append(f"{lab}{'' if v is None else ' = ' + format(v, 'g')} & " + ' & '.join(cells) + r' \\')
(GEN / 'tab_e6_full.tex').write_text('\n'.join(rows) + '\n')
S['e6_contamination5_range'] = [min(S['e6'][f'{b}/contamination/0.05/{d}']['mean_pct'] for b in ('teacher-hybrid', 'student-hybrid') for d in DATASETS),
                                max(S['e6'][f'{b}/contamination/0.05/{d}']['mean_pct'] for b in ('teacher-hybrid', 'student-hybrid') for d in DATASETS)]
S['e6_contamination2_all_sig'] = all(sig(S['e6'][f'{b}/contamination/0.02/{d}']) for b in ('teacher-hybrid', 'student-hybrid') for d in DATASETS)
S['e6_branch_any_detect'] = any(detect(S['e6'][f'{b}/{p}/{v}/{d}']) for b in ('teacher-hybrid', 'student-hybrid') for d in DATASETS for _, _, p, v in PERT[:4])
S['e6_branch_range'] = [min(S['e6'][f'{b}/{p}/{v}/{d}']['mean_pct'] for b in ('teacher-hybrid', 'student-hybrid') for d in DATASETS for _, _, p, v in PERT[:4]),
                        max(S['e6'][f'{b}/{p}/{v}/{d}']['mean_pct'] for b in ('teacher-hybrid', 'student-hybrid') for d in DATASETS for _, _, p, v in PERT[:4])]

# Few-step baselines (E8)
SOLV = [('DDIM', [(n, key_('teacher-hybrid', 'ddim', n, 'hybrid')) for n in (2, 4, 8, 32)]),
        ('DPM-Solver++(2M)', [(n, key_('teacher-hybrid', 'dpm_solver_pp_2m', n, 'hybrid')) for n in (3, 4, 6, 8)]),
        ('Heun', [(2 * st - 2, key_('teacher-hybrid', 'heun', st, 'hybrid')) for st in (3, 4, 5)]),
        ('Endpoint regression (10k pairs)', [(1, key_('student-endpoint_regression', 'direct', 1, 'hybrid'))]),
        ('Endpoint regression (100k pairs)', [(1, key_('student-endpoint_regression_100k', 'direct', 1, 'hybrid'))]),
        ('Consistency, classical student', [(1, key_('student-classical', 'consistency', 1, 'hybrid')), (4, key_('student-classical', 'consistency', 4, 'hybrid'))]),
        ('Consistency, hybrid student', [(1, S1k), (4, S4k)])]
SOLVK = {name: items for name, items in SOLV}
rows = []
for name, items in SOLV:
    for n, k in items:
        rows.append(f"{name} & {n} & " + ' & '.join(f"{np.mean(list(rv(k, d).values())):.1f}" for d in DATASETS) + r' \\')
    rows.append(r'\addlinespace')
(GEN / 'tab_e8.tex').write_text('\n'.join(rows[:-1]) + '\n')
rows = []; S['e8'] = {}
for d in DATASETS:
    cands = {f'DPM-Solver++ {n}': k for n, k in SOLV[1][1] if n <= 4}; cands.update({f'Heun {n}': k for n, k in SOLV[2][1] if n <= 4}); cands['DDIM 4'] = SOLV[0][1][1][1]
    bname = min(cands, key=lambda c: np.mean(list(rv(cands[c], d).values())))
    e1 = ratio_ci(rv(S1k, d), rv(cands[bname], d)); e2 = ratio_ci(rv(SOLVK['Endpoint regression (10k pairs)'][0][1], d), rv(S1k, d))
    e2b = ratio_ci(rv(SOLVK['Endpoint regression (100k pairs)'][0][1], d), rv(S1k, d))
    e3 = ratio_ci(rv(S1k, d), rv(key_('teacher-hybrid', 'dpm_solver_pp_2m', 8, 'hybrid'), d))
    S['e8'][d] = {'best_le4': bname, 'student1_vs_best_le4': e1, 'endpoint_vs_student1': e2, 'endpoint100k_vs_student1': e2b, 'student1_vs_dpm8': e3}
    rows.append(f"{DNAME[d]} & {bname} & {pm(e1)}\\% & {ci_txt(e1)} & {pm(e3)}\\% & {ci_txt(e3)} & {pm(e2)}\\% & {pm(e2b)}\\% {ci_txt(e2b)}" + r' \\')
(GEN / 'tab_e8_ratios.tex').write_text('\n'.join(rows) + '\n')

fig, axes = plt.subplots(1, 4, figsize=(7.0, 1.9))
SC = {'DDIM': ('teacher', 'x', '-'), 'DPM-Solver++(2M)': ('#7a5195', 'v', '-'), 'Heun': ('#8a8984', 'P', '-')}
for ax, d in zip(axes, DATASETS):
    for name, items in SOLV[:3]:
        col, mk, ls = SC[name]; col = COL.get(col, col)
        xs = [n for n, _ in items if n <= 8]; ys = [np.mean(list(rv(k, d).values())) for n, k in items if n <= 8]
        ax.plot(xs, ys, color=col, marker=mk, ms=3.5, lw=1.2, label=f'teacher, {name}')
    for name, kind, items in (('consistency student (hybrid)', 'hybrid', SOLVK['Consistency, hybrid student']), ('endpoint regression (100k)', 'classical', SOLVK['Endpoint regression (100k pairs)'])):
        xs = [n for n, _ in items]; ys = [np.mean(list(rv(k, d).values())) for _, k in items]
        ax.plot(xs, ys, color=COL[kind], marker=MARK[kind], ms=4, lw=0 if len(xs) == 1 else 1.2, label=name)
    ax.axhline(np.mean(list(rv(T32k, d).values())), color='#52514e', lw=.7, ls=':')
    ax.set_xscale('log', base=2); ax.set_yscale('log'); ax.set_xticks([1, 2, 4, 8]); ax.set_xticklabels([1, 2, 4, 8])
    ax.set_title(DNAME[d]); ax.set_xlabel('network evaluations'); clean_log(ax)
axes[0].set_ylabel('FID-like (5,000 samples)')
h_, l_ = axes[0].get_legend_handles_labels(); fig.legend(h_, l_, loc='lower center', ncol=5, bbox_to_anchor=(.5, -.2), fontsize=6.5)
save(fig, 'fig_solvers.pdf')

# ---- E9: correlator readout
def sweep2(path):
    D_, P_ = defaultdict(dict), {}
    for r in load(path)['rows']:
        role = 'teacher' if r['model'] == 'teacher' else 'student'
        k = (r['dataset'], r['kind'], r['hidden'], r['qubits'], r['depth'], r.get('readout', 'zz' if r['pipeline'].endswith('-zz') else 'z'), role, r['steps'])
        D_[k][r['seed']] = fid(r); P_[k] = r['parameters']
    return D_, P_
E9, P9 = sweep2(RV + 'e9-correlators.json')
E9CELLS = [(0, 4, 2), (4, 4, 2), (16, 4, 2), (0, 6, 4)]
rows = []; S['e9'] = {'cells': [], 'passes': defaultdict(int)}
for d in ('mnist', 'fashion_mnist'):
    for h, q, dp in E9CELLS:
        g = lambda kind, ro, role='student', st=1: E9[(d, kind, h, q, dp, ro, role, st)]
        out = {}
        for role, st in (('student', 1), ('teacher', 32)):
            out[(role, 'ent')] = verdict(g('hybrid', 'zz', role, st), g('surrogate', 'zz', role, st))
            out[(role, 'circ')] = verdict(g('hybrid', 'zz', role, st), g('classical_branch', 'zz', role, st))
            out[(role, 'zz')] = verdict(g('hybrid', 'zz', role, st), g('hybrid', 'z', role, st))
            for c in ('ent', 'circ', 'zz'): S['e9']['passes'][f'{role}/{c}'] += int(out[(role, c)][0])
        S['e9']['cells'].append({'dataset': d, 'h': h, 'q': q, 'depth': dp, **{f'{a}/{b}': {'pass': v[0], 'gain': v[1]} for (a, b), v in out.items()}})
        mk = lambda v: f"{1 - v[1]:.2f}{'$^\\ast$' if v[0] else ''}"
        rows.append(f"{DNAME[d] if (h, q) == (0, 4) else ''} & {h} & {q}/{dp} & {fmt(list(g('hybrid', 'zz').values()))} & {fmt(list(g('surrogate', 'zz').values()))} & "
                    f"{fmt(list(g('classical_branch', 'zz').values()))} & {fmt(list(g('hybrid', 'z').values()))} & {mk(out[('student', 'ent')])} & {mk(out[('student', 'circ')])} & {mk(out[('student', 'zz')])} & "
                    f"{mk(out[('teacher', 'ent')])} & {mk(out[('teacher', 'circ')])} & {mk(out[('teacher', 'zz')])}" + r' \\')
    rows.append(r'\addlinespace')
(GEN / 'tab_e9.tex').write_text('\n'.join(rows[:-1]) + '\n')
S['e9']['passes'] = dict(S['e9']['passes'])
cells9 = [(d, h, q, dp) for d in ('mnist', 'fashion_mnist') for h, q, dp in E9CELLS]
S['e9']['pooled'] = {
    'hybrid_zz/surrogate_zz': cluster_ci([(E9[(d, 'hybrid', h, q, dp, 'zz', 'student', 1)], E9[(d, 'surrogate', h, q, dp, 'zz', 'student', 1)]) for d, h, q, dp in cells9]),
    'hybrid_zz/classical_branch_zz': cluster_ci([(E9[(d, 'hybrid', h, q, dp, 'zz', 'student', 1)], E9[(d, 'classical_branch', h, q, dp, 'zz', 'student', 1)]) for d, h, q, dp in cells9]),
    'hybrid_zz/hybrid_z': cluster_ci([(E9[(d, 'hybrid', h, q, dp, 'zz', 'student', 1)], E9[(d, 'hybrid', h, q, dp, 'z', 'student', 1)]) for d, h, q, dp in cells9]),
    'surrogate_zz/surrogate_z': cluster_ci([(E9[(d, 'surrogate', h, q, dp, 'zz', 'student', 1)], E9[(d, 'surrogate', h, q, dp, 'z', 'student', 1)]) for d, h, q, dp in cells9]),
    'hybrid_z/surrogate_z': cluster_ci([(E9[(d, 'hybrid', h, q, dp, 'z', 'student', 1)], E9[(d, 'surrogate', h, q, dp, 'z', 'student', 1)]) for d, h, q, dp in cells9])}
S['e9']['pooled_teacher'] = {
    'hybrid_zz/surrogate_zz': cluster_ci([(E9[(d, 'hybrid', h, q, dp, 'zz', 'teacher', 32)], E9[(d, 'surrogate', h, q, dp, 'zz', 'teacher', 32)]) for d, h, q, dp in cells9]),
    'hybrid_zz/hybrid_z': cluster_ci([(E9[(d, 'hybrid', h, q, dp, 'zz', 'teacher', 32)], E9[(d, 'hybrid', h, q, dp, 'z', 'teacher', 32)]) for d, h, q, dp in cells9])}
rows = []
for lab, e in [('Student (1): hybrid / surrogate, \\texttt{zz}', S['e9']['pooled']['hybrid_zz/surrogate_zz']),
               ('Student (1): hybrid / classical branch, \\texttt{zz}', S['e9']['pooled']['hybrid_zz/classical_branch_zz']),
               ('Student (1): hybrid \\texttt{zz} / hybrid \\texttt{z}', S['e9']['pooled']['hybrid_zz/hybrid_z']),
               ('Student (1): surrogate \\texttt{zz} / surrogate \\texttt{z}', S['e9']['pooled']['surrogate_zz/surrogate_z']),
               ('Teacher (32): hybrid / surrogate, \\texttt{zz}', S['e9']['pooled_teacher']['hybrid_zz/surrogate_zz']),
               ('Teacher (32): hybrid \\texttt{zz} / hybrid \\texttt{z}', S['e9']['pooled_teacher']['hybrid_zz/hybrid_z'])]:
    rows.append(f"{lab} & {pm(e)}\\% & {ci_txt(e)} & {'yes' if e['equivalent'] else 'no'}" + r' \\')
(GEN / 'tab_e9_pooled.tex').write_text('\n'.join(rows) + '\n')

# ---- E10: E1 grid with six seeds (0-2 original E1 runs, 3-5 new)
E10, P10 = sweep(RV + 'e10-capacity-seeds.json')
def six(d, kind, h):
    k = (d, kind, h, 4, 2, False, 'student', 1); a = dict(E1.get(k, ALL.get(k, {}))); a.update(E10.get(k, {})); return a
def verdict6(t, c, need=5):
    seeds = sorted(set(t) & set(c)); gain = 1 - np.mean([t[s] for s in seeds]) / np.mean([c[s] for s in seeds])
    wins = sum(t[s] < c[s] for s in seeds)
    return bool(wins >= need and gain >= .10), gain, wins
rows = []; S['e10'] = {'circuit': 0, 'entanglement': 0, 'cells': [], 'circuit_against': 0}
for d in ('mnist', 'fashion_mnist'):
    for h in (0, 4, 8, 16, 32, 128):
        hy, su, cb = six(d, 'hybrid', h), six(d, 'surrogate', h), six(d, 'classical_branch', h)
        vc = verdict6(hy, cb); ve = verdict6(hy, su); rc_ = verdict6(cb, hy)
        ec = ratio_ci(hy, cb); ee = ratio_ci(hy, su)
        S['e10']['circuit'] += vc[0]; S['e10']['entanglement'] += ve[0]; S['e10']['circuit_against'] += rc_[0]
        S['e10']['cells'].append({'dataset': d, 'h': h, 'n': len(hy), 'circuit': vc, 'entanglement': ve, 'ci_circuit': ec, 'ci_entanglement': ee})
        rows.append(f"{DNAME[d] if h == 0 else ''} & {h} & {fmt(list(hy.values()))} & {fmt(list(su.values()))} & {fmt(list(cb.values()))} & "
                    f"{pm(ec)}\\% {ci_txt(ec)}{'$^\\ast$' if vc[0] else ''} & {vc[2]}/6 & {pm(ee)}\\% {ci_txt(ee)}{'$^\\ast$' if ve[0] else ''} & {ve[2]}/6" + r' \\')
    rows.append(r'\addlinespace')
(GEN / 'tab_e10.tex').write_text('\n'.join(rows[:-1]) + '\n')
e10cells = [(d, h) for d in ('mnist', 'fashion_mnist') for h in (0, 4, 8, 16, 32, 128)]
S['e10']['pooled'] = {'hybrid/surrogate': cluster_ci([(six(d, 'hybrid', h), six(d, 'surrogate', h)) for d, h in e10cells]),
                      'hybrid/classical_branch h>=16': cluster_ci([(six(d, 'hybrid', h), six(d, 'classical_branch', h)) for d, h in e10cells if h >= 16]),
                      'hybrid/classical_branch h<=8': cluster_ci([(six(d, 'hybrid', h), six(d, 'classical_branch', h)) for d, h in e10cells if h <= 8])}
S['e10']['equivalent_cells'] = sum(c['ci_circuit']['equivalent'] for c in S['e10']['cells']) + sum(c['ci_entanglement']['equivalent'] for c in S['e10']['cells'])
# Replication of the new seeds alone (3-5) for the original E1 criteria
S['e10']['new_seeds_only'] = {'circuit': 0, 'entanglement': 0}
for d, h in e10cells:
    g = lambda kind: E10[(d, kind, h, 4, 2, False, 'student', 1)]
    S['e10']['new_seeds_only']['circuit'] += verdict(g('hybrid'), g('classical_branch'))[0]
    S['e10']['new_seeds_only']['entanglement'] += verdict(g('hybrid'), g('surrogate'))[0]

# ================================================================ review round 1 (R1-R6 of the round-1 preregistration)
def rescore(path):
    D_ = defaultdict(dict)
    for res in load(path)['results']:
        for x in res['rows']:
            D_[(res['dataset'], x['kind'], x['hidden'], x['qubits'], x['depth'], x['readout'], x['role'], x.get('perturbation'), x.get('value'))][res['seed']] = x
    return D_
R1 = {'E9': rescore(RV + 'r1-e9-rescore.json'), 'E10': rescore(RV + 'r1-e10-rescore-r2-perturb.json')}
def r1get(sw, d, kind, h, q, dp, ro, role, metric, pert=None, val=None):
    return {s: v[metric] for s, v in R1[sw][(d, kind, h, q, dp, ro, role, pert, val)].items()}
R1CELLS = {'E9': [(d, h, q, dp, ro) for d in ('mnist', 'fashion_mnist') for h, q, dp in E9CELLS for ro in ('zz', 'z')],
           'E10': [(d, h, 4, 2, 'z') for d in ('mnist', 'fashion_mnist') for h in (0, 4, 8, 16, 32, 128)]}
S['r1'] = {}; rows = []
FAMS = [('E9', 'hybrid/surrogate, \\texttt{z}', 'surrogate', lambda c: c[4] == 'z'), ('E9', 'hybrid/surrogate, \\texttt{zz}', 'surrogate', lambda c: c[4] == 'zz'),
        ('E9', 'hybrid/classical branch, \\texttt{z}', 'classical_branch', lambda c: c[4] == 'z'), ('E9', 'hybrid/classical branch, \\texttt{zz}', 'classical_branch', lambda c: c[4] == 'zz'),
        ('E10', 'hybrid/surrogate', 'surrogate', lambda c: True), ('E10', 'hybrid/classical branch, $h\\ge16$', 'classical_branch', lambda c: c[1] >= 16),
        ('E10', 'hybrid/classical branch, $h\\le8$', 'classical_branch', lambda c: c[1] <= 8)]
for sw, lab, ctrl, keep in FAMS:
    cells = [c for c in R1CELLS[sw] if keep(c)]; out = []
    for role in ('student', 'teacher'):
        for metric in ('fid_like', 'latent_frechet'):
            e = cluster_ci([(r1get(sw, c[0], 'hybrid', *c[1:], role, metric), r1get(sw, c[0], ctrl, *c[1:], role, metric)) for c in cells])
            S['r1'][f'{sw}/{lab}/{role}/{metric}'] = e; out.append(e)
    dprec = 100 * np.mean([np.mean(list(r1get(sw, c[0], 'hybrid', *c[1:], 'student', 'precision').values())) - np.mean(list(r1get(sw, c[0], ctrl, *c[1:], 'student', 'precision').values())) for c in cells])
    drec = 100 * np.mean([np.mean(list(r1get(sw, c[0], 'hybrid', *c[1:], 'student', 'recall').values())) - np.mean(list(r1get(sw, c[0], ctrl, *c[1:], 'student', 'recall').values())) for c in cells])
    mmd = []
    for role in ('student', 'teacher'):
        dm = np.array([np.mean([1000 * (r1get(sw, c[0], 'hybrid', *c[1:], role, 'latent_mmd')[x] - r1get(sw, c[0], ctrl, *c[1:], role, 'latent_mmd')[x]) for x in r1get(sw, c[0], 'hybrid', *c[1:], role, 'latent_mmd')]) for c in cells])
        tcm = stats.t.ppf(.95, len(dm) - 1) * dm.std(ddof=1) / math.sqrt(len(dm)); mmd.append((float(dm.mean()), float(dm.mean() - tcm), float(dm.mean() + tcm)))
    S['r1'][f'{sw}/{lab}/mmd_x1000'] = mmd
    rows.append(f"{sw} & {lab} ({len(cells)}) & " + ' & '.join(f"{pm(e)}\\% {ci_txt(e)}" for e in out) + f" & {mmd[0][0]:+.2f} [{mmd[0][1]:+.2f}, {mmd[0][2]:+.2f}] & {dprec:+.1f} / {drec:+.1f}" + r' \\')
(GEN / 'tab_r1.tex').write_text('\n'.join(rows) + '\n')
# per-cell preregistered rules at 5,000 samples (3 seeds)
S['r1_rules'] = {}
for sw in ('E9', 'E10'):
    for ctrl in ('surrogate', 'classical_branch'):
        for role in ('student', 'teacher'):
            S['r1_rules'][f'{sw}/{ctrl}/{role}'] = int(sum(verdict(r1get(sw, c[0], 'hybrid', *c[1:], role, 'fid_like'), r1get(sw, c[0], ctrl, *c[1:], role, 'fid_like'))[0] for c in R1CELLS[sw]))
            S['r1_rules'][f'{sw}/{ctrl}/{role}/against'] = int(sum(verdict(r1get(sw, c[0], ctrl, *c[1:], role, 'fid_like'), r1get(sw, c[0], 'hybrid', *c[1:], role, 'fid_like'))[0] for c in R1CELLS[sw]))

S['r1_teacher_sur_vs_cb'] = {sw: cluster_ci([(r1get(sw, c[0], 'surrogate', *c[1:], 'teacher', 'fid_like'), r1get(sw, c[0], 'classical_branch', *c[1:], 'teacher', 'fid_like')) for c in R1CELLS[sw]]) for sw in ('E9', 'E10')}
S['r1_teacher_sur_vs_cb_rule'] = {sw: int(sum(verdict(r1get(sw, c[0], 'surrogate', *c[1:], 'teacher', 'fid_like'), r1get(sw, c[0], 'classical_branch', *c[1:], 'teacher', 'fid_like'))[0] for c in R1CELLS[sw])) for sw in ('E9', 'E10')}
S['r1_teacher_circuit_pass_cells'] = {sw: [list(map(str, c)) for c in R1CELLS[sw] if verdict(r1get(sw, c[0], 'hybrid', *c[1:], 'teacher', 'fid_like'), r1get(sw, c[0], 'classical_branch', *c[1:], 'teacher', 'fid_like'))[0]] for sw in ('E9', 'E10')}
S['r1_teacher_ent_pass_cells'] = {sw: [list(map(str, c)) for c in R1CELLS[sw] if verdict(r1get(sw, c[0], 'hybrid', *c[1:], 'teacher', 'fid_like'), r1get(sw, c[0], 'surrogate', *c[1:], 'teacher', 'fid_like'))[0]] for sw in ('E9', 'E10')}
# R2: branch perturbations where the branch carries the model
R2P = [('$a=0$', 'scale', 0.0), ('$a=0.5$', 'scale', 0.5), ('$a=1.5$', 'scale', 1.5), ('$a=2$', 'scale', 2.0), ('$a=3$', 'scale', 3.0), ('permuted', 'shuffle', None), ('readout zeroed', 'readout_zeroed', None)]
rows = []; S['r2'] = {}
for d in ('mnist', 'fashion_mnist'):
    for h in (0, 4, 16):
        for role, st in (('student', 1), ('teacher', 32)):
            base = r1get('E10', d, 'hybrid', h, 4, 2, 'z', role, 'fid_like'); cells = []
            for lab, pert, val in R2P:
                e = ratio_ci(r1get('E10', d, 'hybrid', h, 4, 2, 'z', role, 'fid_like', pert, val), base); S['r2'][f'{d}/{h}/{role}/{pert}/{val}'] = e
                cells.append(f"{e['mean_pct']:+.0f}{'$^\\ast$' if detect(e) else ('$^\\circ$' if sig(e) else '')}")
            rows.append(f"{DNAME[d] if (h, role) == (0, 'student') else ''} & {h} & {role} ({st}) & " + ' & '.join(cells) + r' \\')
    rows.append(r'\addlinespace')
(GEN / 'tab_r2.tex').write_text('\n'.join(rows[:-1]) + '\n')
S['r2_detected'] = sum(detect(e) for e in S['r2'].values()); S['r2_total'] = len(S['r2'])
S['r2_detected_improvements'] = sum(detect(e) and e['mean_pct'] < 0 for e in S['r2'].values())
S['r2_min_pct'] = min(e['mean_pct'] for e in S['r2'].values())
S['r2_a0_detected_student'] = sum(detect(S['r2'][f'{d}/{h}/student/scale/0.0']) for d in ('mnist', 'fashion_mnist') for h in (0, 4, 16))
S['r2_a0_range_student'] = [min(S['r2'][f'{d}/{h}/student/scale/0.0']['mean_pct'] for d in ('mnist', 'fashion_mnist') for h in (0, 4, 16)),
                            max(S['r2'][f'{d}/{h}/student/scale/0.0']['mean_pct'] for d in ('mnist', 'fashion_mnist') for h in (0, 4, 16))]

# R3: model-level positive control, pairs of independently trained pipelines with a known capacity difference
R3P = [('MLP only, $h=8$ vs.\\ $h=16$', 'classical', 8, 16), ('MLP only, $h=16$ vs.\\ $h=32$', 'classical', 16, 32), ('MLP only, $h=32$ vs.\\ $h=128$', 'classical', 32, 128),
       ('Hybrid, $h=16$ vs.\\ $h=32$', 'hybrid', 16, 32), ('Hybrid, $h=32$ vs.\\ $h=128$', 'hybrid', 32, 128)]
rows = []; S['r3'] = {}
for lab, kind, ha, hb in R3P:
    cells = []
    for d in ('mnist', 'fashion_mnist'):
        e5 = ratio_ci(r1get('E10', d, kind, ha, 4, 2, 'z', 'student', 'fid_like'), r1get('E10', d, kind, hb, 4, 2, 'z', 'student', 'fid_like'))
        e6 = ratio_ci(six(d, kind, ha), six(d, kind, hb)); S['r3'][f'{d}/{kind}/{ha}/{hb}'] = {'n5000_3seeds': e5, 'n512_6seeds': e6}
        cells += [f"{pm(e5)}\\% {ci_txt(e5)}", f"{pm(e6)}\\% {ci_txt(e6)}"]
    rows.append(f"{lab} & " + ' & '.join(cells) + r' \\')
(GEN / 'tab_r3.tex').write_text('\n'.join(rows) + '\n')

# R6: bootstrap over seeds for pooled intervals (cells of one seed share the AE), geometric E10 rule, surrogate/classical branch
def seed_t(cells):
    # Seed-clustered interval: mean log-ratio over cells for each seed (cells of a seed share the AE), t-interval over seeds.
    # (Replaces a percentile bootstrap over seeds, which is anti-conservative with 3-6 seeds; round-2 review.)
    seeds = sorted(set.intersection(*[set(t) & set(c) for t, c in cells]))
    per = np.array([np.mean([math.log(t[s] / c[s]) for t, c in cells]) for s in seeds]); n = len(per)
    m = per.mean(); se = per.std(ddof=1) / math.sqrt(n); tc = stats.t.ppf(.95, n - 1); pct = lambda x: 100 * (math.exp(x) - 1)
    return {'mean_pct': pct(m), 'lo_pct': pct(m - tc * se), 'hi_pct': pct(m + tc * se), 'equivalent': bool(m - tc * se > -MARGIN and m + tc * se < MARGIN),
            'seeds': n, 'per_seed_pct': [pct(x) for x in per]}
boot_pool = seed_t
S['r6_seed_t'] = {'E10 hybrid/surrogate': boot_pool([(six(d, 'hybrid', h), six(d, 'surrogate', h)) for d, h in e10cells]),
                     'E10 hybrid/classical_branch h>=16': boot_pool([(six(d, 'hybrid', h), six(d, 'classical_branch', h)) for d, h in e10cells if h >= 16]),
                     'E10 surrogate/classical_branch h>=16': boot_pool([(six(d, 'surrogate', h), six(d, 'classical_branch', h)) for d, h in e10cells if h >= 16]),
                     'E10 hybrid/classical_branch h<=8': boot_pool([(six(d, 'hybrid', h), six(d, 'classical_branch', h)) for d, h in e10cells if h <= 8]),
                     'E1 hybrid/surrogate (3 seeds)': boot_pool([(st1(d, 'hybrid', h), st1(d, 'surrogate', h)) for d, h in e1cells])}
S['e10']['pooled']['surrogate/classical_branch h>=16'] = cluster_ci([(six(d, 'surrogate', h), six(d, 'classical_branch', h)) for d, h in e10cells if h >= 16])
def verdict6g(t, c, need=5):
    seeds = sorted(set(t) & set(c)); r = [math.log(t[s] / c[s]) for s in seeds]
    return bool(sum(x < 0 for x in r) >= need and math.exp(np.mean(r)) <= 0.9)
S['e10']['geometric_rule'] = {'circuit': int(sum(verdict6g(six(d, 'hybrid', h), six(d, 'classical_branch', h)) for d, h in e10cells)),
                              'entanglement': int(sum(verdict6g(six(d, 'hybrid', h), six(d, 'surrogate', h)) for d, h in e10cells)),
                              'circuit_against': int(sum(verdict6g(six(d, 'classical_branch', h), six(d, 'hybrid', h)) for d, h in e10cells))}
S['e10']['significant_cells'] = [{'dataset': c['dataset'], 'h': c['h'], 'comparison': comp, 'mean_pct': c[k]['mean_pct'], 'lo': c[k]['lo_pct'], 'hi': c[k]['hi_pct']}
                                 for c in S['e10']['cells'] for comp, k in (('circuit', 'ci_circuit'), ('entanglement', 'ci_entanglement')) if sig(c[k])]

# Entangling contribution of trained circuits (post hoc, round-2 review)
EC = defaultdict(list)
for r in load(RV + 'r2-entangling-contribution.json')['rows']:
    EC[(r['sweep'], r['dataset'], r['hidden'], r['qubits'], r['depth'], r['readout'], r['role'])].append((r['seed'], r['relative_rms_entangling']))
rows = []; xs, ys = [], []
for sw, swn in (('v2-e10-capacity-seeds', 'E10'), ('v2-e9-correlators', 'E9')):
    keys = sorted({k[:6] for k in EC if k[0] == sw}, key=lambda k: (k[1], k[5], k[3], k[2]))
    for k in keys:
        vals = {role: np.mean([v for _, v in EC[k + (role,)]]) for role in ('init', 'teacher', 'student')}
        ratio = ratio_ci(r1get(swn, k[1], 'hybrid', k[2], k[3], k[4], k[5], 'student', 'fid_like'), r1get(swn, k[1], 'surrogate', k[2], k[3], k[4], k[5], 'student', 'fid_like'))
        for sd, v in EC[k + ('student',)]:
            h_ = r1get(swn, k[1], 'hybrid', k[2], k[3], k[4], k[5], 'student', 'fid_like'); s_ = r1get(swn, k[1], 'surrogate', k[2], k[3], k[4], k[5], 'student', 'fid_like')
            xs.append(v); ys.append(math.log(h_[sd] / s_[sd]))
        rows.append(f"{swn} & {DNAME[k[1]]} & {k[2]} & {k[3]}/{k[4]} & \\texttt{{{k[5]}}} & {vals['init']:.2f} & {vals['teacher']:.2f} & {vals['student']:.2f} & {pm(ratio)}\\%" + r' \\')
(GEN / 'tab_entangling.tex').write_text('\n'.join(rows) + '\n')
rho, prho = stats.spearmanr(xs, ys)
allv = {role: [v for k, l in EC.items() if k[6] == role for _, v in l] for role in ('init', 'teacher', 'student')}
S['entangling'] = {'init_range': [float(min(allv['init'])), float(max(allv['init']))], 'teacher_range': [float(min(allv['teacher'])), float(max(allv['teacher']))],
                   'student_range': [float(min(allv['student'])), float(max(allv['student']))], 'student_median': float(np.median(allv['student'])),
                   'spearman_student_vs_logratio': float(rho), 'spearman_p': float(prho), 'n': len(xs)}
fig, ax = plt.subplots(figsize=(3.4, 2.3))
ax.scatter(xs, [100 * (math.exp(y) - 1) for y in ys], s=9, color=COL['hybrid'], alpha=.75, lw=0)
ax.axhspan(-100 * (1 - 1 / 1.1), 10, color='#e4e3df', alpha=.7, lw=0); ax.axhline(0, color='#52514e', lw=.8)
ax.set_xlabel('entangling contribution (RMS, rel. to feature SD)'); ax.set_ylabel('hybrid / surrogate, % (1-step)')
save(fig, 'fig_entangling.pdf')

# Forest plot with six seeds for the E1 cells (E10) and three for E2 (overwrites the three-seed version)
forest6 = []
for d in ('mnist', 'fashion_mnist'):
    for h in (0, 4, 8, 16, 32, 128):
        lab = f"E1 {DNAME[d]} h={h} (6 seeds)"
        forest6.append((lab, 'circuit', ratio_ci(six(d, 'hybrid', h), six(d, 'classical_branch', h))))
        forest6.append((lab, 'entanglement', ratio_ci(six(d, 'hybrid', h), six(d, 'surrogate', h))))
forest6 += [f for f in forest if f[0].startswith('E2')]
fig, axes = plt.subplots(1, 2, figsize=(7.0, 5.4), sharey=True)
labels6 = [f[0] for f in forest6 if f[1] == 'circuit']
from matplotlib.ticker import FixedLocator, NullLocator
for ax, comp, title in zip(axes, ('circuit', 'entanglement'), ('hybrid / classical branch', 'hybrid / separable surrogate')):
    items = [f for f in forest6 if f[1] == comp]
    for i, (lab, _, r) in enumerate(items):
        y = len(items) - 1 - i; col = COL['hybrid'] if comp == 'circuit' else COL['surrogate']
        ax.plot([max(1 + r['lo_pct'] / 100, .2), min(1 + r['hi_pct'] / 100, 5)], [y, y], color=col, lw=1.2); ax.plot(1 + r['mean_pct'] / 100, y, marker='o', color=col, ms=3.5)
    ax.axvspan(1 / 1.1, 1.1, color='#e4e3df', alpha=.7, lw=0); ax.axvline(1, color='#52514e', lw=.8)
    ax.set_xscale('log'); ax.set_xlim(.25, 4); ax.xaxis.set_major_locator(FixedLocator([.25, .5, 1, 2, 4])); ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xticklabels(['0.25', '0.5', '1', '2', '4']); ax.set_title(title); ax.set_xlabel('FID-like ratio (<1: circuit better)')
    ax.set_yticks(range(len(items))); ax.set_yticklabels(labels6[::-1], fontsize=6.5); ax.grid(axis='y', visible=False)
save(fig, 'fig_forest.pdf')
S['forest6_equivalent'] = sum(f[2]['equivalent'] for f in forest6); S['forest6_n'] = len(forest6)

# ---------------------------------------------------------------- classifier validation
cv = {d: json.load(open(p)) for d, p in (('mnist', REP / 'pilot-mnist/mnist_classifier.validation.json'),
      ('fashion_mnist', REP / 'campaign-datasets-teachers/fashion_mnist_classifier.validation.json'),
      ('kmnist', REP / 'campaign-datasets-teachers/kmnist_classifier.validation.json'),
      ('pneumoniamnist', REP / 'campaign-datasets-teachers/pneumoniamnist_classifier.validation.json'))}
S['classifier_accuracy'] = {d: v['test_accuracy'] for d, v in cv.items()}

(GEN / 'stats.json').write_text(json.dumps(S, indent=1) + '\n')
print(json.dumps({k: S[k] for k in ('criteria_primary', 'forest_any_equivalent', 'e0_varying_range', 'classifier_accuracy', 'e4b_floor')}, indent=1))
for k, e in eq.items():
    if isinstance(e, dict): print(f"{k:60s} {e['mean_pct']:+6.1f}% [{e['lo_pct']:+6.1f}, {e['hi_pct']:+6.1f}] eq={e['equivalent']}")
