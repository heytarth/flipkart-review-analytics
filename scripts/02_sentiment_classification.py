"""
Stage 2 / Component 1: Sentiment Classification
Compares two classifiers (satisfies "compare 2+ methods" requirement).
"""
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                              classification_report, confusion_matrix)
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from pathlib import Path

# Paths resolved relative to this file, not the working directory.
ROOT = Path(__file__).resolve().parent.parent
CLEANED = ROOT / 'data' / 'processed' / 'flipkart_cleaned.csv'
FIGURES = ROOT / 'outputs' / 'figures'
TABLES = ROOT / 'outputs' / 'tables'
MODELS = ROOT / 'outputs' / 'models'
for _d in (FIGURES, TABLES, MODELS):
    _d.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(CLEANED)

# Use only rows with at least 2 words of review text -- single-word
# reviews like "Good" carry too little signal for meaningful TF-IDF features.
df = df[df['word_count'] >= 2].copy()
print(f"Rows used for classification: {len(df)}")

# --- Feature Representation: TF-IDF ---
# TF-IDF chosen over plain Bag-of-Words because it downweights very
# common words (e.g. "product", "good") that appear across most reviews
# regardless of sentiment, giving more weight to distinguishing terms.
tfidf = TfidfVectorizer(max_features=5000, stop_words='english',
                         ngram_range=(1, 2), min_df=5)
X = tfidf.fit_transform(df['Summary_clean'])
y = df['sentiment_label']

# Row positions are carried through the split so that misclassified reviews can
# be traced back to their original text for the error analysis at the end.
positions = np.arange(X.shape[0])
X_train, X_test, y_train, y_test, pos_train, pos_test = train_test_split(
    X, y, positions, test_size=0.2, random_state=42, stratify=y
)
print(f"Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")

results = {}
models = {}

# --- Classifier 1: Multinomial Naive Bayes ---
# Standard, fast baseline for text classification; works well with
# TF-IDF/word-count style features.
nb = MultinomialNB()
nb.fit(X_train, y_train)
results['Naive Bayes'] = {'pred': nb.predict(X_test)}
models['model_naive_bayes'] = nb

# --- Classifier 2: Decision Tree ---
# class_weight='balanced' compensates for the positive-class imbalance
# observed during preprocessing (positive reviews dominate the dataset).
dt = DecisionTreeClassifier(max_depth=30, class_weight='balanced', random_state=42)
dt.fit(X_train, y_train)
results['Decision Tree'] = {'pred': dt.predict(X_test)}
models['model_decision_tree'] = dt

# --- Classifier 3: Logistic Regression ---
# A discriminative linear model, in contrast to the generative Naive Bayes.
# class_weight='balanced' re-weights the loss by inverse class frequency,
# which directly targets the weak neutral class (only 9% of the data).
lr = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
lr.fit(X_train, y_train)
results['Logistic Regression'] = {'pred': lr.predict(X_test)}
models['model_logistic_regression'] = lr

# --- Classifier 4: Complement Naive Bayes ---
# A variant of Naive Bayes designed specifically for imbalanced text data:
# it estimates each class's parameters from the complement of that class,
# which reduces the bias toward the majority class that standard
# MultinomialNB shows on skewed corpora like this one.
cnb = ComplementNB()
cnb.fit(X_train, y_train)
results['Complement NB'] = {'pred': cnb.predict(X_test)}
models['model_complement_nb'] = cnb

# Metrics computed once per model and reused everywhere below.
# Weighted F1 is dominated by the large positive class; macro F1 treats all
# three classes equally and therefore exposes the weak neutral class, so both
# are reported. Neutral F1 is broken out separately because it is the single
# weakest result in the project and should not be hidden inside an average.
LABELS = ['negative', 'neutral', 'positive']
for name, res in results.items():
    res['accuracy'] = accuracy_score(y_test, res['pred'])
    wp, wr, wf1, _ = precision_recall_fscore_support(
        y_test, res['pred'], average='weighted', zero_division=0)
    res['precision'], res['recall'], res['f1'] = wp, wr, wf1
    res['macro_f1'] = precision_recall_fscore_support(
        y_test, res['pred'], average='macro', zero_division=0)[2]
    per_class_f1 = precision_recall_fscore_support(
        y_test, res['pred'], labels=LABELS, zero_division=0)[2]
    res['neutral_f1'] = per_class_f1[LABELS.index('neutral')]
    res['negative_f1'] = per_class_f1[LABELS.index('negative')]

# --- Evaluation: Accuracy, Precision, Recall, F1 for both models ---
print("\n" + "="*60)
for name, res in results.items():
    print(f"\n{name}")
    print("-" * 40)
    print(f"Accuracy: {res['accuracy']:.4f}")
    print(f"Weighted Precision: {res['precision']:.4f} | "
          f"Recall: {res['recall']:.4f} | F1: {res['f1']:.4f}")
    print(f"Macro F1: {res['macro_f1']:.4f} | Neutral F1: {res['neutral_f1']:.4f}")
    print(classification_report(y_test, res['pred']))

# --- Method comparison summary table ---
comparison = pd.DataFrame({
    name: {
        'Accuracy': res['accuracy'],
        'Precision': res['precision'],
        'Recall': res['recall'],
        'F1': res['f1'],
        'Macro_F1': res['macro_f1'],
        'Negative_F1': res['negative_f1'],
        'Neutral_F1': res['neutral_f1'],
    }
    for name, res in results.items()
}).T
comparison.to_csv(TABLES / 'classifier_comparison.csv')
print("\nComparison table saved to classifier_comparison.csv")
print(comparison.round(4).to_string())

# --- Confusion matrices (visualization) ---
n = len(results)
fig, axes = plt.subplots(2, 2, figsize=(13, 10))
for ax, (name, res) in zip(axes.flatten(), results.items()):
    cm = confusion_matrix(y_test, res['pred'], labels=LABELS)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=LABELS,
                yticklabels=LABELS, ax=ax, cbar=False)
    ax.set_title(f"{name}\nacc={res['accuracy']:.3f}  "
                 f"weighted F1={res['f1']:.3f}  macro F1={res['macro_f1']:.3f}",
                 fontsize=10)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
for ax in axes.flatten()[n:]:          # hide unused panels if models are added/removed
    ax.axis('off')
plt.tight_layout()
plt.savefig(FIGURES / 'confusion_matrices.png', dpi=120)
print("Saved confusion_matrices.png")

# --- Final model choice ---
# Selected on WEIGHTED F1, matching the stated justification. (An earlier
# version of this script selected on accuracy while claiming to use F1; with
# four models those two criteria can disagree, so the metric is now explicit.)
best_weighted = max(results, key=lambda k: results[k]['f1'])
best_macro = max(results, key=lambda k: results[k]['macro_f1'])
best_neutral = max(results, key=lambda k: results[k]['neutral_f1'])

print("\n" + "=" * 62)
print(f"Best on WEIGHTED F1 : {best_weighted} "
      f"({results[best_weighted]['f1']:.4f})")
print(f"Best on MACRO F1    : {best_macro} "
      f"({results[best_macro]['macro_f1']:.4f})")
print(f"Best on NEUTRAL F1  : {best_neutral} "
      f"({results[best_neutral]['neutral_f1']:.4f})")
print("=" * 62)
if best_weighted != best_macro:
    print("NOTE: weighted and macro F1 disagree. Weighted F1 rewards doing well\n"
          "on the dominant positive class; macro F1 weights all three classes\n"
          "equally and so favours a model that handles neutral better. State\n"
          "which objective was chosen and why, rather than quoting only one.")
print("\nJustification: the final model is selected on weighted F1, which balances\n"
      "precision and recall across the three imbalanced classes better than\n"
      "accuracy alone. Macro F1 is reported alongside it because weighted F1 can\n"
      "mask poor performance on the small neutral class.")

# ----------------------------------------------------------------------
# Error analysis (required: "class-wise performance and error analysis")
# ----------------------------------------------------------------------
# Aggregate metrics say how much a model gets wrong. They do not say WHAT it
# gets wrong, which is what determines whether the errors matter. The analysis
# below is run for the weighted-F1 winner and the macro-F1 winner, because
# their error profiles differ in a way that drives the final model choice.
print("\n" + "=" * 62)
print("ERROR ANALYSIS")
print("=" * 62)

test_text = df.iloc[pos_test][['Summary_clean', 'Rate', 'word_count']].reset_index(drop=True)
y_test_arr = y_test.reset_index(drop=True)

error_rows = []
# dict.fromkeys, not a set literal: a set's iteration order depends on string
# hashing, which Python randomises per process, so a set here made the row order
# of sentiment_error_analysis.csv differ between otherwise identical runs. This
# keeps weighted-F1 winner first and drops the duplicate if both are the model.
for name in dict.fromkeys([best_weighted, best_macro]):
    pred = pd.Series(results[name]['pred'])
    wrong = pred != y_test_arr

    print(f"\n--- {name} ---")
    print(f"Misclassified: {wrong.sum():,} of {len(pred):,} "
          f"({100*wrong.mean():.1f}%)")

    # Which confusions dominate?
    confusions = (pd.DataFrame({'actual': y_test_arr[wrong], 'predicted': pred[wrong]})
                  .groupby(['actual', 'predicted']).size()
                  .sort_values(ascending=False))
    print("\nMost frequent confusions (actual -> predicted):")
    for (a, p), n in confusions.items():
        print(f"  {a:>8} -> {p:<8} {n:6,}  ({100*n/wrong.sum():5.1f}% of all errors)")

    # Does error rate depend on review length? Short reviews carry less signal,
    # so if errors concentrate there the limitation is the data, not the model.
    bins = pd.cut(test_text['word_count'], [0, 2, 5, 10, 20, 1000],
                  labels=['1-2', '3-5', '6-10', '11-20', '20+'])
    by_len = wrong.groupby(bins, observed=False).mean()
    print("\nError rate by review length (words):")
    for label, rate in by_len.items():
        print(f"  {label:>6}: {100*rate:5.1f}%")

    # Concrete examples of the single largest confusion, for the report.
    (a, p) = confusions.index[0]
    examples = test_text[wrong & (y_test_arr == a) & (pred == p)].head(6)
    print(f"\nExamples of the largest confusion ({a} -> {p}):")
    for _, row in examples.iterrows():
        print(f"  [{row['Rate']}*] {row['Summary_clean'][:90]}")

    for (a, p), n in confusions.items():
        error_rows.append({'model': name, 'actual': a, 'predicted': p,
                           'count': int(n),
                           'pct_of_errors': round(100 * n / wrong.sum(), 2)})

pd.DataFrame(error_rows).to_csv(TABLES / 'sentiment_error_analysis.csv', index=False)
print(f"\nSaved outputs/tables/sentiment_error_analysis.csv")

joblib.dump(tfidf, MODELS / 'tfidf_vectorizer.pkl')
for fname, model in models.items():
    joblib.dump(model, MODELS / f'{fname}.pkl')
print(f"\nSaved vectorizer and {len(models)} models to outputs/models/")
