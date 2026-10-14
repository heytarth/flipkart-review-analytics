"""
Stage 7: Render the pipeline architecture diagram.

The project guidelines require an architecture figure in the Dataset &
Methodology section, and recommend the pipeline:

    Data Collection -> Preprocessing -> Feature Representation
    -> Analytics/ML -> Evaluation -> Visualization -> Insights

This script draws that pipeline as implemented here, with the three analytical
components shown as parallel branches and the validation loop shown explicitly,
since external validation of the clustering is the part that distinguishes this
project from a standard three-component pipeline. Those validation labels are
LLM-generated, not human annotation -- see 06_score_validation.py.

Generated rather than drawn by hand so it stays consistent with the code.
"""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path(__file__).resolve().parent.parent
FIGURES = ROOT / 'outputs' / 'figures'
FIGURES.mkdir(parents=True, exist_ok=True)

STAGE = '#2d5f8b'       # pipeline stages
COMPONENT = '#3f7d5a'   # the three analytical components
EVAL = '#8b5a2d'        # evaluation
VALID = '#7a3b52'       # external validation
TEXT = 'white'

fig, ax = plt.subplots(figsize=(15, 8.5))
ax.set_xlim(0, 15)
ax.set_ylim(0, 8.5)
ax.axis('off')


def box(x, y, w, h, label, sub='', color=STAGE, fontsize=10):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle='round,pad=0.08',
                                facecolor=color, edgecolor='none'))
    if sub:
        ax.text(x + w / 2, y + h * 0.63, label, ha='center', va='center',
                color=TEXT, fontsize=fontsize, fontweight='bold')
        ax.text(x + w / 2, y + h * 0.26, sub, ha='center', va='center',
                color=TEXT, fontsize=fontsize - 2.2)
    else:
        ax.text(x + w / 2, y + h / 2, label, ha='center', va='center',
                color=TEXT, fontsize=fontsize, fontweight='bold')
    return (x, y, w, h)


def arrow(p1, p2, style='-|>', color='#555555', rad=0.0):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle=style, color=color,
                                 mutation_scale=16, lw=1.4,
                                 connectionstyle=f'arc3,rad={rad}'))


ax.text(7.5, 8.15, 'E-Commerce Review Analytics — Pipeline Architecture',
        ha='center', fontsize=14, fontweight='bold')

# --- linear front of the pipeline -------------------------------------------
b1 = box(0.3, 6.3, 2.6, 1.0, 'Data Collection',
         'Kaggle: 189,874 reviews', STAGE)
b2 = box(3.4, 6.3, 2.9, 1.0, 'Preprocessing',
         '7 steps -> 132,410 rows', STAGE)
b3 = box(6.8, 6.3, 3.0, 1.0, 'Feature Representation',
         'TF-IDF · counts · behavioural', STAGE)

arrow((2.9, 6.8), (3.4, 6.8))
arrow((6.3, 6.8), (6.8, 6.8))

# --- the three analytical components ----------------------------------------
ax.text(7.5, 5.75, 'Three analytical components', ha='center',
        fontsize=10, style='italic', color='#444444')

c1 = box(0.3, 4.0, 4.4, 1.35, 'Component 1 — Sentiment Classification',
         'TF-IDF (1,2)-grams · NB · DT · LogReg · ComplementNB', COMPONENT, 9.5)
c2 = box(5.2, 4.0, 4.4, 1.35, 'Component 2 — Topic Discovery',
         'LDA, k=6 · counts · coherence sweep', COMPONENT, 9.5)
c3 = box(10.1, 4.0, 4.6, 1.35, 'Component 3 — Low-Information Detection',
         'K-Means, K=4 · behavioural features', COMPONENT, 9.5)

# Fan out below the feature box so no connector crosses an upstream stage.
arrow((8.3, 6.3), (8.3, 5.95))
arrow((8.3, 5.95), (2.5, 5.35), rad=-0.18)
arrow((8.3, 5.95), (7.4, 5.35), rad=-0.05)
arrow((8.3, 5.95), (12.4, 5.35), rad=0.18)

# --- evaluation --------------------------------------------------------------
e1 = box(0.3, 2.35, 4.4, 1.1, 'Evaluation',
         'Accuracy · P/R/F1 · macro F1 · confusion · error analysis', EVAL, 9.5)
e2 = box(5.2, 2.35, 4.4, 1.1, 'Evaluation',
         'UMass coherence · topic–sentiment · soft-mixture check', EVAL, 9.5)
e3 = box(10.1, 2.35, 4.6, 1.1, 'Evaluation',
         'Silhouette · non-degeneracy rule', EVAL, 9.5)

for x in (2.5, 7.4, 12.4):
    arrow((x, 4.0), (x, 3.45))

# --- external validation loop ---------------------------------------------------
v = box(10.1, 0.75, 4.6, 1.1, 'External Validation  (200 labelled reviews)',
        'LLM-labelled, blind, pre-registered criterion · precision 0.980',
        VALID, 9.5)
arrow((12.4, 2.35), (12.4, 1.85))
arrow((14.7, 1.3), (14.95, 3.0), rad=-0.45)
ax.text(14.85, 2.15, 'validates', rotation=90, ha='center', va='center',
        fontsize=8, color=VALID, style='italic')

# --- insights ----------------------------------------------------------------
ins = box(0.3, 0.75, 9.3, 1.1, 'Integrated Insights',
          'service-related topic is most negative (46.4%) · '
          '14.8% of reviews carry no information', '#333333', 10)
arrow((2.5, 2.35), (2.5, 1.85))
arrow((7.4, 2.35), (7.4, 1.85))

ax.text(0.3, 0.35,
        'Component 3 is validated against external labels, not internal metrics alone. Those '
        'labels are LLM-generated, not human annotation -- see the report.',
        fontsize=8.5, style='italic', color='#444444')

plt.tight_layout()
out = FIGURES / 'pipeline_architecture.png'
plt.savefig(out, dpi=150, facecolor='white', bbox_inches='tight')
print(f"Saved {out.relative_to(ROOT)}")
