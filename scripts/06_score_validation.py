"""
Stage 5b: Score the clustering flag against the manual labels.

Converts Component 3 from "silhouette says the clusters are separated" into a
external evaluation: how well does cluster membership agree with a judgement
made by reading the review text?

PROVENANCE OF THE LABELS -- READ BEFORE QUOTING ANY FIGURE FROM THIS SCRIPT:
the labels in validation_sample_labeled.csv were produced by an LLM (Claude)
applying the written criterion in validation/LABELLING_INSTRUCTIONS.md. They
are NOT human annotation. Describe them as LLM-assisted labelling, never as
"hand-labelled" or "human judgement".

This still compares two independent signals: K-Means never reads the words (it
uses length, repetition frequency and rating), while the labeller read only the
text. But the independence is weaker than human annotation would give, because
an LLM's notion of "low-information" plausibly tracks brevity, which is also
most of what the clustering keys on. A shared bias of that kind would inflate
the agreement. State this whenever the figures are reported.


Run 05_export_validation_sample.py first, produce labels following
validation/LABELLING_INSTRUCTIONS.md, and save them as
validation/validation_sample_labeled.csv.
"""
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import (precision_recall_fscore_support, confusion_matrix,
                             accuracy_score, classification_report)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path(__file__).resolve().parent.parent
VALDIR = ROOT / 'validation'
FIGURES = ROOT / 'outputs' / 'figures'
TABLES = ROOT / 'outputs' / 'tables'
for _d in (FIGURES, TABLES):
    _d.mkdir(parents=True, exist_ok=True)

LABELED = VALDIR / 'validation_sample_labeled.csv'
KEY = VALDIR / 'validation_key.csv'

if not LABELED.exists():
    sys.exit(
        f"Not found: {LABELED}\n\n"
        "Produce labels for validation/validation_sample.csv first (fill the\n"
        "human_label column with 1 or 0, following LABELLING_INSTRUCTIONS.md)\n"
        "and save it under the name above."
    )

lab = pd.read_csv(LABELED)
key = pd.read_csv(KEY)

# --- Validate the labelling before scoring anything ---
if 'human_label' not in lab.columns:
    sys.exit("The labelled file has no 'human_label' column.")

blank = lab['human_label'].isna().sum()
if blank:
    sys.exit(f"{blank} of {len(lab)} rows have no label. Label every row, then re-run.")

bad = set(lab['human_label'].astype(str).str.strip().unique()) - {'0', '1'}
if bad:
    sys.exit(f"human_label must contain only 0 or 1. Found: {sorted(bad)}")

merged = lab.merge(key, on='row_id', how='inner')
if len(merged) != len(lab):
    sys.exit(f"Only {len(merged)} of {len(lab)} rows matched the key on row_id. "
             "Did row_id get edited or re-sorted?")

# The low-information cluster is identified in 05 and stored in the key as the
# smallest-mean-length group; recover it the same way rather than hard-coding.
clustered = pd.read_csv(ROOT / 'data' / 'processed' / 'flipkart_with_clusters.csv',
                        usecols=['word_count', 'log_text_freq', 'cluster'])
prof = clustered.groupby('cluster').agg(word_count=('word_count', 'mean'),
                                        log_text_freq=('log_text_freq', 'mean'))
z = lambda s: (s - s.mean()) / s.std(ddof=0)
FLAGGED = (z(prof['log_text_freq']) - z(prof['word_count'])).idxmax()

y_true = merged['human_label'].astype(int)
y_pred = (merged['cluster'] == FLAGGED).astype(int)

print(f"Labelled reviews scored: {len(merged)}")
print(f"Low-information cluster: {FLAGGED}")
print(f"Reference labels marked low-information: {y_true.sum()} of {len(y_true)} "
      f"({100*y_true.mean():.1f}%)")
print(f"Flag predicted low-information: {y_pred.sum()} of {len(y_pred)} "
      f"({100*y_pred.mean():.1f}%)")

p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary',
                                              zero_division=0)
acc = accuracy_score(y_true, y_pred)
cm = confusion_matrix(y_true, y_pred)

print("\n" + "=" * 52)
print("VALIDATION OF THE LOW-INFORMATION FLAG")
print("Reference labels are LLM-generated, NOT human annotation.")
print("=" * 52)
print(f"Precision : {p:.3f}   of reviews the flag catches, this share are "
      f"genuinely low-information")
print(f"Recall    : {r:.3f}   of genuinely low-information reviews, this "
      f"share are caught")
print(f"F1        : {f1:.3f}")
print(f"Accuracy  : {acc:.3f}")
print(f"\nConfusion matrix (rows = reference label, cols = flag):")
print(pd.DataFrame(cm,
                   index=['ref: informative', 'ref: low-info'],
                   columns=['flag: not flagged', 'flag: flagged']).to_string())
print()
print(classification_report(y_true, y_pred,
                            target_names=['informative', 'low-information'],
                            zero_division=0))

results = pd.DataFrame([{
    'n_labelled': len(merged),
    'flagged_cluster': FLAGGED,
    'precision': round(p, 4),
    'recall': round(r, 4),
    'f1': round(f1, 4),
    'accuracy': round(acc, 4),
    'true_negatives': int(cm[0, 0]), 'false_positives': int(cm[0, 1]),
    'false_negatives': int(cm[1, 0]), 'true_positives': int(cm[1, 1]),
}])
results.to_csv(TABLES / 'validation_results.csv', index=False)
print(f"Saved outputs/tables/validation_results.csv")

# --- Error analysis: the rows worth discussing in the report ---
merged['correct'] = (y_true == y_pred)
fp = merged[(y_pred == 1) & (y_true == 0)]
fn = merged[(y_pred == 0) & (y_true == 1)]

print(f"\nFALSE POSITIVES ({len(fp)}): flagged, but labelled informative")
if len(fp):
    print(fp[['Summary_clean', 'Rate']].head(10).to_string(index=False))
print(f"\nFALSE NEGATIVES ({len(fn)}): missed, but labelled low-information")
if len(fn):
    print(fn[['Summary_clean', 'Rate', 'cluster']].head(10).to_string(index=False))

pd.concat([fp.assign(error='false_positive'),
           fn.assign(error='false_negative')]).to_csv(
    VALDIR / 'validation_errors.csv', index=False)
print(f"\nSaved validation/validation_errors.csv for the error-analysis section")

# --- Figure ---
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=axes[0],
            xticklabels=['not flagged', 'flagged'],
            yticklabels=['informative', 'low-info'])
axes[0].set_title(f'Flag vs LLM reference labels (n={len(merged)})')
axes[0].set_xlabel('K-Means flag')
axes[0].set_ylabel('Reference label')

axes[1].bar(['Precision', 'Recall', 'F1', 'Accuracy'], [p, r, f1, acc],
            color=['steelblue', 'seagreen', 'goldenrod', 'slategrey'])
axes[1].set_ylim(0, 1)
axes[1].set_title('Validation metrics for the low-information flag')
for i, v in enumerate([p, r, f1, acc]):
    axes[1].text(i, v + 0.02, f'{v:.3f}', ha='center')

plt.tight_layout()
plt.savefig(FIGURES / 'validation_results.png', dpi=120)
print("Saved outputs/figures/validation_results.png")

print("\nREMINDERS FOR THE REPORT:")
print("1. The reference labels are LLM-generated, not human annotation. This is")
print("   NOT human validation and must not be described as such.")
print("2. The sample is balanced 100/100 by design while only ~14.8% of the full")
print("   dataset is flagged, so these figures describe the sample, not the corpus.")
print("3. The LLM labeller and the clustering may share a bias toward brevity,")
print("   which would inflate agreement. Independence is weaker than it looks.")
