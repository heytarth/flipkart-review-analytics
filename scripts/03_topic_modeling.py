"""
Stage 3 / Component 2: Topic Discovery via LDA
Discovers latent themes across reviews (e.g. delivery, quality, price).
"""
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from pathlib import Path

# Paths resolved relative to this file, not the working directory.
ROOT = Path(__file__).resolve().parent.parent
CLEANED = ROOT / 'data' / 'processed' / 'flipkart_cleaned.csv'
FIGURES = ROOT / 'outputs' / 'figures'
TABLES = ROOT / 'outputs' / 'tables'
PROCESSED = ROOT / 'data' / 'processed'
for _d in (FIGURES, TABLES, PROCESSED):
    _d.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(CLEANED)

# Use reviews with at least 4 words -- LDA needs enough tokens per
# document to find meaningful co-occurrence patterns; very short
# reviews ("Nice", "Good product") add noise without topical signal.
df_topic = df[df['word_count'] >= 4].copy()
print(f"Rows used for topic modeling: {len(df_topic)}")

# --- Feature Representation: Bag-of-Words (required by LDA, not TF-IDF) ---
# LDA is a probabilistic model over raw word counts, not weighted scores,
# so CountVectorizer (not TF-IDF) is the correct representation here.
vectorizer = CountVectorizer(max_features=2000, stop_words='english',
                              min_df=10, max_df=0.5)
doc_term_matrix = vectorizer.fit_transform(df_topic['Summary_clean'])
feature_names = vectorizer.get_feature_names_out()

# --- LDA Model ---
# CHOICE OF k=6 -- evidence and the argument against it, both stated.
#
# scripts/07_topic_coherence.py sweeps k over {3,4,5,6,7,8,10} and computes
# UMass coherence. The sweep does NOT select 6:
#
#     k= 3  coherence -2.8529        k= 7  coherence -3.2810
#     k= 4  coherence -2.8106  <-- best
#     k= 5  coherence -3.1019        k= 8  coherence -3.2703
#     k= 6  coherence -3.1112        k=10  coherence -3.2678
#
# k=6 is retained anyway, on interpretability grounds:
#
#   * At k=4 two of the four topics are near-duplicates -- "product, good,
#     flipkart, price, best, nice" and "good, product, nice, quality, money,
#     value" describe the same generic-praise theme twice, so the better
#     coherence score buys no additional insight.
#   * UMass coherence is known to favour fewer, broader topics: it rewards
#     co-occurrence among high-frequency terms, and in a corpus spanning 800+
#     product categories the generic terms ("good", "product", "quality")
#     co-occur with nearly everything.
#   * At k=6 distinguishable themes separate -- delivery/installation,
#     size/fit, and audio-visual product features each acquire their own
#     topic, and the delivery/installation topic carries the project's main
#     finding (highest negative rate of any topic).
#
# An earlier version of this comment claimed k=6 had been chosen by inspecting
# coherence when no coherence had been computed. The sweep now exists, it
# disagrees, and the disagreement is reported rather than hidden.
N_TOPICS = 6
lda = LatentDirichletAllocation(n_components=N_TOPICS, random_state=42,
                                 max_iter=15, learning_method='online')
lda.fit(doc_term_matrix)

# --- Display top words per topic ---
def get_top_words(model, feature_names, n_top=10):
    topics = {}
    for idx, topic in enumerate(model.components_):
        top_indices = topic.argsort()[-n_top:][::-1]
        topics[idx] = [feature_names[i] for i in top_indices]
    return topics

topics = get_top_words(lda, feature_names, n_top=10)
print("\nDiscovered Topics (most probable terms):")
for idx, words in topics.items():
    print(f"Topic {idx}: {', '.join(words)}")


# --- Distinctive terms per topic ---
# The most PROBABLE words in a topic are dominated by corpus-wide generics
# ("product", "good", "quality") that appear near the top of every topic and so
# distinguish nothing. Distinctiveness ranks each term by how much more
# probable it is in this topic than across topics on average, which surfaces
# the vocabulary that actually separates one topic from another. Used for
# figure labels so that a label is supported by the evidence behind it.
probs = lda.components_ / lda.components_.sum(axis=1, keepdims=True)
mean_prob = probs.mean(axis=0)
distinctive = {i: [feature_names[j] for j in (probs[i] / mean_prob).argsort()[-8:][::-1]]
               for i in range(N_TOPICS)}
print("\nDiscovered Topics (most DISTINCTIVE terms):")
for idx, words in distinctive.items():
    print(f"Topic {idx}: {', '.join(words)}")

# --- Assign dominant topic to each review ---
doc_topic_dist = lda.transform(doc_term_matrix)
df_topic['dominant_topic'] = doc_topic_dist.argmax(axis=1)

print("\nReview count per topic:")
print(df_topic['dominant_topic'].value_counts().sort_index())

# --- Soft topic mixtures: robustness check on the argmax simplification ---
# Assigning each review to its single most probable topic discards the mixture
# LDA actually infers. If the headline finding (which topic is most negative)
# survives using the full distribution as weights, the simplification is safe
# to report. If it does not, the argmax version is an artefact.
sentiment_dummies = pd.get_dummies(df_topic['sentiment_label']).astype(float)
weighted = doc_topic_dist.T @ sentiment_dummies.values
weighted = weighted / weighted.sum(axis=1, keepdims=True)
soft = pd.DataFrame(weighted, columns=sentiment_dummies.columns)
soft.index.name = 'topic'

hard = pd.crosstab(df_topic['dominant_topic'], df_topic['sentiment_label'],
                   normalize='index')

print("\nSoft (mixture-weighted) sentiment per topic:")
print(soft.round(3).to_string())
print(f"\nMost negative topic -- hard assignment (argmax): "
      f"{hard['negative'].idxmax()} ({100*hard['negative'].max():.1f}% negative)")
print(f"Most negative topic -- soft assignment (mixture): "
      f"{soft['negative'].idxmax()} ({100*soft['negative'].max():.1f}% negative)")
if hard['negative'].idxmax() == soft['negative'].idxmax():
    print("ROBUST: both methods identify the same most-negative topic, so the")
    print("argmax simplification does not drive the headline finding.")
else:
    print("WARNING: the two methods disagree. The argmax result may be an")
    print("artefact of the hard assignment and must not be reported alone.")
soft.round(4).to_csv(TABLES / 'topic_sentiment_soft.csv')
print("Saved topic_sentiment_soft.csv")

# --- Visualization 1: topic sizes and negativity, labelled by top words ---
# Topic IDs alone carry no meaning for a reader, so each bar is labelled with
# its leading terms and annotated with the share of negative reviews -- the
# figure then shows the finding rather than just the counts.
counts = df_topic['dominant_topic'].value_counts().sort_index()
labels = [f"T{i}: {', '.join(distinctive[i][:3])}" for i in counts.index]
neg_rate = hard['negative']

fig, axes = plt.subplots(1, 2, figsize=(15, 5))

bars = axes[0].barh(labels, counts.values, color='steelblue')
axes[0].set_title('Reviews assigned to each discovered topic')
axes[0].set_xlabel('Number of reviews')
axes[0].set_ylabel('Topic (with leading terms)')
axes[0].invert_yaxis()
for bar, v in zip(bars, counts.values):
    axes[0].text(v, bar.get_y() + bar.get_height() / 2, f' {v:,}',
                 va='center', fontsize=9)

colors = ['indianred' if v == neg_rate.max() else 'slategrey' for v in neg_rate]
axes[1].barh(labels, 100 * neg_rate.values, color=colors)
axes[1].set_title('Negative reviews within each topic\n'
                  '(red = highest, the service-related topic)')
axes[1].set_xlabel('Negative reviews (%)')
axes[1].set_ylabel('')
axes[1].invert_yaxis()
axes[1].set_yticklabels([])
for i, v in enumerate(100 * neg_rate.values):
    axes[1].text(v, i, f' {v:.1f}%', va='center', fontsize=9)

plt.tight_layout()
plt.savefig(FIGURES / 'topic_distribution.png', dpi=120)
print("Saved topic_distribution.png")

# --- Visualization 2: Word clouds per topic ---
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
for idx, ax in zip(range(N_TOPICS), axes.flatten()):
    word_freq = {word: lda.components_[idx][list(feature_names).index(word)]
                 for word in topics[idx]}
    wc = WordCloud(width=400, height=300, background_color='white').generate_from_frequencies(word_freq)
    ax.imshow(wc, interpolation='bilinear')
    ax.set_title(f'Topic {idx}')
    ax.axis('off')
plt.tight_layout()
plt.savefig(FIGURES / 'topic_wordclouds.png', dpi=120)
print("Saved topic_wordclouds.png")

# --- Cross-tabulation: topic vs sentiment (integrated insight) ---
crosstab = pd.crosstab(df_topic['dominant_topic'], df_topic['sentiment_label'], normalize='index')
crosstab.to_csv(TABLES / 'topic_sentiment_crosstab.csv')
print("\nTopic vs Sentiment (proportion within each topic):")
print(crosstab)

df_topic[['ProductName_clean', 'Summary_clean', 'sentiment_label', 'dominant_topic']].to_csv(
    PROCESSED / 'reviews_with_topics.csv', index=False)
print("\nSaved reviews_with_topics.csv")
