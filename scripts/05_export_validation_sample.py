"""
Stage 5a: Export a blind sample for manual validation of the clustering.

WHY THIS EXISTS:
The silhouette score says the clusters are geometrically well separated. It
says nothing about whether the low-information cluster actually contains
low-information reviews. That claim needs an external judgement made by
reading the text, so this script exports a blind sample for labelling and
06_score_validation.py scores the result.

NOTE: the labels currently in validation_sample_labeled.csv were produced by an
LLM applying the written criterion, not by a human annotator. See the
provenance note in 06_score_validation.py.

BLINDING:
The cluster ID is deliberately NOT written to the exported file, and the rows
are shuffled, so whoever or whatever labels the sample cannot tell which group a
review came from. The mapping back to clusters is recovered later via `row_id`.
"""
import pandas as pd
from pathlib import Path

RANDOM_STATE = 42
N_PER_GROUP = 100

ROOT = Path(__file__).resolve().parent.parent
CLUSTERED = ROOT / 'data' / 'processed' / 'flipkart_with_clusters.csv'
VALDIR = ROOT / 'validation'
VALDIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(CLUSTERED)
print(f"Loaded {len(df)} clustered reviews")

# Identify the low-information cluster the same way 04 does, rather than
# hard-coding an index -- cluster numbering is not stable if the script is
# re-run with different parameters.
import numpy as np
prof = df.groupby('cluster').agg(word_count=('word_count', 'mean'),
                                 log_text_freq=('log_text_freq', 'mean'))
z = lambda s: (s - s.mean()) / s.std(ddof=0)
FLAGGED = (z(prof['log_text_freq']) - z(prof['word_count'])).idxmax()
print(f"Low-information cluster identified as: cluster {FLAGGED}")
print(f"  mean length {prof.loc[FLAGGED,'word_count']:.2f} words, "
      f"{(df['cluster']==FLAGGED).sum()} reviews "
      f"({100*(df['cluster']==FLAGGED).mean():.2f}% of data)")

flagged = df[df['cluster'] == FLAGGED].sample(N_PER_GROUP, random_state=RANDOM_STATE)
other = df[df['cluster'] != FLAGGED].sample(N_PER_GROUP, random_state=RANDOM_STATE)

sample = (pd.concat([flagged, other])
          .sample(frac=1, random_state=1)          # shuffle: order leaks nothing
          .reset_index()
          .rename(columns={'index': 'row_id'}))

out = sample[['row_id', 'ProductName_clean', 'Summary_clean', 'Rate']].copy()
out['human_label'] = ''                            # <-- you fill this in: 1 or 0

sample_path = VALDIR / 'validation_sample.csv'
out.to_csv(sample_path, index=False)
print(f"\nSaved {len(out)} reviews to {sample_path.relative_to(ROOT)}")
print("The 'cluster' column is intentionally absent so labelling stays blind.")

# Keep the key separately so scoring can recover the cluster assignment.
sample[['row_id', 'cluster']].to_csv(VALDIR / 'validation_key.csv', index=False)
print(f"Saved cluster key to {(VALDIR / 'validation_key.csv').relative_to(ROOT)} "
      f"-- DO NOT open this before labelling.")

print(f"\nSample composition (hidden from the labeller):")
print(sample['cluster'].value_counts().sort_index().to_string())
print(f"\nLength distribution in the sample:")
print(out.assign(words=out['Summary_clean'].fillna('').str.split().str.len())
         ['words'].describe().round(2).to_string())
