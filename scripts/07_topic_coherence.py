"""
Stage 6: Justify the number of LDA topics with a coherence sweep.

WHY THIS EXISTS:
03_topic_modeling.py uses 6 topics. That number was originally a judgement
call, and the script's comment claimed it came from inspecting coherence when
no coherence had been computed. This script supplies the missing evidence.

WHY NOT GENSIM:
gensim is the usual source of coherence measures, but it is not installed and
installs awkwardly on Python 3.13. UMass coherence is implemented directly
here instead -- it needs only document co-occurrence counts, which come
straight from the same document-term matrix the LDA is fitted on, so no extra
dependency is required.

WHAT UMASS COHERENCE MEASURES:
For each topic, take its top N words. For every ordered pair (w_i, w_j), ask:
given that w_j appeared in a document, how often did w_i appear too? A topic
whose top words genuinely co-occur describes a real theme; one whose top words
rarely appear together is an artefact of the model. Scores are negative, and
VALUES CLOSER TO ZERO ARE BETTER.

    C_UMass(t) = sum_{i=2..N} sum_{j=1..i-1} log( (D(w_i, w_j) + 1) / D(w_j) )

averaged over topics. The +1 avoids log(0) for pairs that never co-occur.
"""
import sys

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

RANDOM_STATE = 42
K_RANGE = [3, 4, 5, 6, 7, 8, 10]
TOP_N = 10          # top words per topic used for the coherence calculation

# The sweep fits one LDA model per k and evaluates perplexity for each. On all
# 84,070 documents that takes hours, almost entirely in the perplexity step,
# which runs a full variational E-step over the corpus. A 25,000-document
# random subsample gives the same *ranking* of k at a fraction of the cost --
# and only the ranking is used, since the chosen k is then refitted on the full
# corpus in 03_topic_modeling.py. The subsample size is reported in the output
# so the figures are not mistaken for full-corpus values.
SAMPLE_DOCS = 25000

ROOT = Path(__file__).resolve().parent.parent
CLEANED = ROOT / 'data' / 'processed' / 'flipkart_cleaned.csv'
FIGURES = ROOT / 'outputs' / 'figures'
TABLES = ROOT / 'outputs' / 'tables'
for _d in (FIGURES, TABLES):
    _d.mkdir(parents=True, exist_ok=True)

# Re-styling the figure should not cost a full sweep: --plot-only redraws from
# the saved table. The sweep itself fits one LDA model per k and takes ~13 min.
PLOT_ONLY = '--plot-only' in sys.argv

def run_sweep():
    """Fit one LDA per k, score coherence and perplexity, save the table."""
    df = pd.read_csv(CLEANED)
    df_topic = df[df['word_count'] >= 4].copy()
    n_full = len(df_topic)
    if len(df_topic) > SAMPLE_DOCS:
        df_topic = df_topic.sample(SAMPLE_DOCS, random_state=RANDOM_STATE)
    print(f"Documents available (>=4 words, same filter as script 03): {n_full:,}")
    print(f"Documents used for this sweep: {len(df_topic):,} (random subsample)")

    # Identical vectoriser settings to 03, so the sweep describes the same model
    vectorizer = CountVectorizer(max_features=2000, stop_words='english',
                                 min_df=10, max_df=0.5)
    dtm = vectorizer.fit_transform(df_topic['Summary_clean'])
    feature_names = vectorizer.get_feature_names_out()
    print(f"Vocabulary: {len(feature_names)} terms")

    # Binary document-occurrence matrix: did term j appear in document i at all?
    # Co-occurrence counts come from its Gram matrix.
    binary = (dtm > 0).astype(np.int32)
    doc_freq = np.asarray(binary.sum(axis=0)).ravel()      # D(w) for every term


    def umass_coherence(model, top_n=TOP_N):
        """Mean UMass coherence across topics. Closer to zero is better."""
        topic_scores = []
        for topic_weights in model.components_:
            top_idx = topic_weights.argsort()[-top_n:][::-1]       # most probable first
            sub = binary[:, top_idx]
            co_doc = np.asarray((sub.T @ sub).todense())           # D(w_i, w_j)
            score = 0.0
            pairs = 0
            for i in range(1, len(top_idx)):
                for j in range(i):
                    # j is the more probable word, i the less probable one
                    score += np.log((co_doc[i, j] + 1) / doc_freq[top_idx[j]])
                    pairs += 1
            topic_scores.append(score / pairs)                     # mean per pair
        return float(np.mean(topic_scores)), topic_scores


    rows = []
    topic_words = {}
    for k in K_RANGE:
        lda = LatentDirichletAllocation(n_components=k, random_state=RANDOM_STATE,
                                        max_iter=15, learning_method='online')
        lda.fit(dtm)
        coh, per_topic = umass_coherence(lda)
        perp = lda.perplexity(dtm)
        rows.append({'k': k, 'umass_coherence': round(coh, 4),
                     'perplexity': round(perp, 1),
                     'worst_topic_coherence': round(min(per_topic), 4)})
        topic_words[k] = [
            ', '.join(feature_names[t.argsort()[-6:][::-1]]) for t in lda.components_
        ]
        print(f"  k={k:2d}: UMass coherence={coh:+.4f}  "
              f"(worst topic {min(per_topic):+.4f})  perplexity={perp:.1f}")

    res = pd.DataFrame(rows)
    best_k = int(res.loc[res['umass_coherence'].idxmax(), 'k'])      # closest to zero
    res['selected'] = res['k'].eq(best_k)
    res.to_csv(TABLES / 'topic_coherence.csv', index=False)

    print("\n" + "=" * 64)
    print(res.to_string(index=False))
    print("=" * 64)
    print(f"\nBest UMass coherence at k={best_k}")
    print(f"Script 03 currently uses k=6 "
          f"(coherence {res.loc[res.k.eq(6), 'umass_coherence'].iloc[0]:+.4f})")

    print(f"\nTop words per topic at the best k ({best_k}):")
    for i, words in enumerate(topic_words[best_k]):
        print(f"  Topic {i}: {words}")
    if best_k != 6:
        print("\nTop words per topic at k=6 (the value script 03 uses):")
        for i, words in enumerate(topic_words[6]):
            print(f"  Topic {i}: {words}")

    print("\nNOTE: perplexity is reported for completeness but correlates poorly")
    print("with human interpretability -- it typically keeps improving as k grows,")
    print("which is why coherence rather than perplexity drives the choice here.")
    print(f"NOTE: this sweep ran on a {len(df_topic):,}-document random subsample of")
    print(f"the {n_full:,} eligible reviews. It is used to RANK values of k; the")
    print("chosen k is refitted on the full corpus in 03_topic_modeling.py.")
    return res


if PLOT_ONLY:
    res = pd.read_csv(TABLES / 'topic_coherence.csv')
    valid = {int(k): True for k in res['k']}
    K_RANGE = res['k'].tolist()
    best_k = int(res.loc[res['umass_coherence'].idxmax(), 'k'])
    print(f"--plot-only: redrawn from the saved table, no models refitted")
else:
    res = run_sweep()
    best_k = int(res.loc[res['umass_coherence'].idxmax(), 'k'])

# The figure must show BOTH the value coherence prefers and the value actually
# retained in 03_topic_modeling.py. Marking only the coherence winner would let
# the figure argue against the project's own choice without explaining it.
RETAINED_K = 6

fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
colors = []
for k, sel in zip(res['k'], res['selected']):
    if k == RETAINED_K:
        colors.append('darkorange')
    elif sel:
        colors.append('seagreen')
    else:
        colors.append('steelblue')
bars = axes[0].bar(res['k'].astype(str), res['umass_coherence'], color=colors)
axes[0].set_title('UMass topic coherence by k (closer to zero is better)\n'
                  'green = best coherence · orange = retained for interpretability')
axes[0].set_xlabel('Number of topics (k)')
axes[0].set_ylabel('Mean UMass coherence')
# Value labels sit INSIDE the bars. Placing them past the bar end would put
# white text on the white canvas, where it is invisible.
for bar, v in zip(bars, res['umass_coherence']):
    axes[0].text(bar.get_x() + bar.get_width() / 2, v + 0.13, f'{v:.2f}',
                 ha='center', va='bottom', fontsize=9, color='white',
                 fontweight='bold')
axes[0].set_ylim(min(res['umass_coherence']) * 1.10, 0)
# The explanation for keeping k=6 lives in the caption and the report, not as an
# arrow across the bars.

axes[1].plot(res['k'], res['perplexity'], marker='o', color='indianred')
axes[1].set_title('Perplexity by k\n(lower is better, but a poor guide to '
                  'interpretability)')
axes[1].set_xlabel('Number of topics (k)')
axes[1].set_ylabel('Perplexity')

plt.tight_layout()
plt.savefig(FIGURES / 'topic_coherence.png', dpi=120)
print("\nSaved outputs/figures/topic_coherence.png and "
      "outputs/tables/topic_coherence.csv")
