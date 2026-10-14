"""
Stage 4 / Component 3: Unsupervised segmentation of reviews by
information quality, using K-Means on behavioural (non-topical) features.

FRAMING -- READ BEFORE QUOTING THIS IN THE REPORT:
This component identifies reviews that are statistically unusual and low in
information content. It does NOT establish that any review is fake, paid or
fraudulent. The dataset carries no ground-truth labels for authenticity, so
no such claim can be evaluated here. Describe the output as
"low-information / anomalous reviews", never as "fake review detection".
"""
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

# Paths resolved relative to this file, not the working directory.
ROOT = Path(__file__).resolve().parent.parent
CLEANED = ROOT / 'data' / 'processed' / 'flipkart_cleaned.csv'
FIGURES = ROOT / 'outputs' / 'figures'
TABLES = ROOT / 'outputs' / 'tables'
PROCESSED = ROOT / 'data' / 'processed'
for _d in (FIGURES, TABLES, PROCESSED):
    _d.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
SIL_SAMPLE = 10000          # subsample for silhouette (full 132k is O(n^2))
MIN_CLUSTER_FRAC = 0.01     # reject solutions with a cluster below 1% of data

df = pd.read_csv(CLEANED)
print(f"Rows loaded: {len(df)}")

# ----------------------------------------------------------------------
# Feature engineering
# ----------------------------------------------------------------------
# 1. word_count        - length proxy; one-word reviews carry little signal
# 2. exclamation_ratio - punctuation intensity per word
# 3. log_text_freq     - how often this exact review text occurs in the RAW
#                        dataset, counted in 01_preprocessing.py BEFORE
#                        deduplication (see note below)
# 4. Rate              - the star rating itself
#
# WHY text frequency is computed upstream, not here:
# 01_preprocessing.py removes exact duplicate reviews. An earlier version of
# this script counted duplicates AFTER that removal, which returned 1 for
# virtually every row -- the feature was effectively constant and contributed
# nothing, leaving the clustering to run on word_count and Rate alone. The
# count is now recorded before deduplication and across all products.
#
# WHY the log transform:
# Raw text frequency spans 1 to 15,818 ("Good" occurs 15,818 times) with a
# median of 1. K-Means minimises squared Euclidean distance, so on the raw
# scale this one feature dominates all others even after standardisation.
# log1p compresses 15,818 to about 9.7, making the feature comparable to the
# rest while preserving its ordering.
#
# WHY mismatch_flag is NOT a clustering input:
# It is a binary flag true for only ~0.9% of rows. A near-constant binary
# feature splits the space into two perfectly separable sheets, inflating the
# silhouette score (K=2 scored 0.83) while producing a partition that merely
# recovers the flag itself and carries no analytical content. It is retained
# below as a descriptive statistic reported per cluster.

df['exclamation_count'] = df['Summary_clean'].str.count('!')
df['exclamation_ratio'] = df['exclamation_count'] / df['word_count'].replace(0, 1)

negative_words = ['bad', 'worst', 'poor', 'waste', 'useless',
                  'disappointed', 'defective']
df['negative_word_count'] = df['Summary_clean'].fillna('').str.lower().apply(
    lambda x: sum(word in x for word in negative_words))
df['mismatch_flag'] = ((df['Rate'] >= 4) & (df['negative_word_count'] > 0)).astype(int)

if 'text_freq_global' not in df.columns:
    raise SystemExit(
        "flipkart_cleaned.csv has no 'text_freq_global' column.\n"
        "Re-run 01_preprocessing.py first -- it records pre-deduplication "
        "text frequency, which this script depends on."
    )
df['log_text_freq'] = np.log1p(df['text_freq_global'])

feature_cols = ['word_count', 'exclamation_ratio', 'log_text_freq', 'Rate']
X = df[feature_cols].fillna(0)

print("\nFeature summary (before scaling):")
print(X.describe().loc[['mean', 'std', 'min', '50%', 'max']].round(3).to_string())

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ----------------------------------------------------------------------
# Choosing K: highest silhouette among non-degenerate solutions
# ----------------------------------------------------------------------
# Silhouette alone can reward a solution that isolates a tiny outlier pocket
# (K=5 gains only +0.005 by splitting off 191 rows, 0.14% of the data). Any
# solution containing a cluster below MIN_CLUSTER_FRAC is therefore rejected
# as degenerate: such a cluster describes too few reviews to support an
# interpretation. The rule is applied automatically and reported below.
print(f"\nSelecting K (rejecting any solution with a cluster below "
      f"{MIN_CLUSTER_FRAC:.0%} of {len(df)} rows):")

K_RANGE = [2, 3, 4, 5, 6, 7, 8]
sil_scores, min_fracs, valid = {}, {}, {}
for k in K_RANGE:
    km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
    labels = km.fit_predict(X_scaled)
    sil_scores[k] = silhouette_score(X_scaled, labels,
                                     sample_size=SIL_SAMPLE,
                                     random_state=RANDOM_STATE)
    min_fracs[k] = np.bincount(labels).min() / len(labels)
    valid[k] = min_fracs[k] >= MIN_CLUSTER_FRAC
    print(f"  K={k}: silhouette={sil_scores[k]:.4f}   "
          f"smallest cluster={min_fracs[k]:7.2%}   "
          f"{'ok' if valid[k] else 'REJECTED (degenerate)'}")

best_k = max((k for k in K_RANGE if valid[k]), key=lambda k: sil_scores[k])
print(f"\nK selected: {best_k} (silhouette {sil_scores[best_k]:.4f}, highest "
      f"among non-degenerate solutions)")

# ----------------------------------------------------------------------
# Final clustering
# ----------------------------------------------------------------------
kmeans = KMeans(n_clusters=best_k, random_state=RANDOM_STATE, n_init=10)
df['cluster'] = kmeans.fit_predict(X_scaled)
final_silhouette = silhouette_score(X_scaled, df['cluster'],
                                    sample_size=SIL_SAMPLE,
                                    random_state=RANDOM_STATE)
print(f"Final silhouette score: {final_silhouette:.4f}")

# ----------------------------------------------------------------------
# Cluster profiles
# ----------------------------------------------------------------------
profile = df.groupby('cluster')[feature_cols].mean()
profile['text_freq_global'] = df.groupby('cluster')['text_freq_global'].mean()
profile['mismatch_flag'] = df.groupby('cluster')['mismatch_flag'].mean()
profile['n_reviews'] = df['cluster'].value_counts().sort_index()
profile['pct_of_data'] = 100 * profile['n_reviews'] / len(df)

# Identify the low-information cluster on a standardised basis: high repeated
# text combined with short length. Z-scores keep the two features on a
# comparable footing -- adding their raw means would let text frequency
# (range 1-15,818) decide the outcome by itself, which is the bug that made
# an earlier version of this script flag 99% of the dataset.
z = lambda s: (s - s.mean()) / s.std(ddof=0)
lowinfo_score = z(profile['log_text_freq']) - z(profile['word_count'])
lowinfo_cluster = lowinfo_score.idxmax()
profile['lowinfo_score'] = lowinfo_score.round(3)

print("\nCluster profiles:")
print(profile.round(3).to_string())

profile.round(4).to_csv(TABLES / 'cluster_profiles.csv')
print("\nSaved cluster_profiles.csv")

# Sentiment composition per cluster -- links Component 3 back to Component 1
cluster_sent = pd.crosstab(df['cluster'], df['sentiment_label'], normalize='index')
cluster_sent.round(4).to_csv(TABLES / 'cluster_sentiment.csv')
print("\nSentiment composition within each cluster:")
print(cluster_sent.round(3).to_string())
print("Saved cluster_sentiment.csv")

n_low = int(profile.loc[lowinfo_cluster, 'n_reviews'])
print(f"\nCluster {lowinfo_cluster} shows the strongest low-information "
      f"profile: {n_low} reviews ({100*n_low/len(df):.2f}% of the dataset), "
      f"mean length {profile.loc[lowinfo_cluster, 'word_count']:.1f} words, "
      f"mean raw text repetition "
      f"{profile.loc[lowinfo_cluster, 'text_freq_global']:.0f}x.")
print("This indicates low information content, NOT confirmed inauthenticity.")

print("\nMost frequently repeated review texts in the raw dataset:")
top_rep = (df[['Summary_clean', 'text_freq_global', 'cluster']]
           .drop_duplicates('Summary_clean')
           .nlargest(8, 'text_freq_global'))
print(top_rep.to_string(index=False))

# ----------------------------------------------------------------------
# Visualisation
# ----------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

bar_colors = ['seagreen' if valid[k] else 'lightgrey' for k in K_RANGE]
axes[0, 0].bar([str(k) for k in K_RANGE],
               [sil_scores[k] for k in K_RANGE], color=bar_colors)
axes[0, 0].axvline(str(best_k), color='crimson', ls='--', lw=1.5,
                   label=f'selected K={best_k}')
axes[0, 0].set_title('Silhouette score by K\n(grey = rejected as degenerate)')
axes[0, 0].set_xlabel('Number of clusters (K)')
axes[0, 0].set_ylabel('Silhouette score')
axes[0, 0].legend()

axes[0, 1].bar(profile.index.astype(str), profile['n_reviews'], color='steelblue')
axes[0, 1].set_title(f'Reviews per cluster (K={best_k})')
axes[0, 1].set_xlabel('Cluster')
axes[0, 1].set_ylabel('Number of reviews')
for i, (_, row) in enumerate(profile.iterrows()):
    axes[0, 1].text(i, row['n_reviews'], f"{row['pct_of_data']:.1f}%",
                    ha='center', va='bottom', fontsize=9)

samp = df.sample(min(6000, len(df)), random_state=RANDOM_STATE)
sc = axes[1, 0].scatter(samp['word_count'], samp['log_text_freq'],
                        c=samp['cluster'], cmap='tab10', s=6, alpha=0.5)
axes[1, 0].set_title('Cluster structure (6,000-review sample)')
axes[1, 0].set_xlabel('Word count (symlog scale)')
axes[1, 0].set_ylabel('log(1 + text frequency)')
axes[1, 0].set_xscale('symlog')
axes[1, 0].legend(*sc.legend_elements(), title='Cluster',
                  loc='upper right', fontsize=8)

cluster_sent.plot(kind='bar', stacked=True, ax=axes[1, 1],
                  color=['indianred', 'goldenrod', 'mediumseagreen'])
axes[1, 1].set_title('Sentiment composition within each cluster')
axes[1, 1].set_xlabel('Cluster')
axes[1, 1].set_ylabel('Proportion of reviews')
axes[1, 1].legend(title='Sentiment', fontsize=8)
axes[1, 1].tick_params(axis='x', rotation=0)

plt.tight_layout()
plt.savefig(FIGURES / 'clustering_analysis.png', dpi=120)
print("\nSaved clustering_analysis.png")

df.to_csv(PROCESSED / 'flipkart_with_clusters.csv', index=False)
print("Saved flipkart_with_clusters.csv")
