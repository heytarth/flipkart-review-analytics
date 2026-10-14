# Results Log

Verbatim terminal output of a full pipeline run. **Every number quoted in the report and
slides must come from this file.** Regenerate it after any script change:

```
python scripts/regenerate_results_log.py
```

- Generated: 2026-10-09 20:57
- Python 3.13.6 | pandas 3.0.1 | scikit-learn 1.8.0
- All scripts use `random_state=42` and reproduce exactly.
- `05_export_validation_sample.py` is not re-run here; it would rewrite the sample that the
  existing hand labels are matched against. See the note in this script.

---

## Stage 1 — Preprocessing

`scripts/01_preprocessing.py` — 12s

```
Raw shape: (189874, 5)
Dropped 5 rows with invalid/corrupted Rate values
Dropped 13 rows with missing Summary text

Pre-dedup text frequency recorded:
  max global repeats of one review text: 15818
  reviews whose text appears >1 time globally: 107776 (56.8%)
Dropped 57446 exact duplicate reviews

Final cleaned shape: (132410, 13)

Sentiment label distribution:
sentiment_label
positive    98170
negative    23207
neutral     11033
Name: count, dtype: int64

Word count stats:
count    132410.000000
mean         10.086051
std          14.346922
min           0.000000
25%           2.000000
50%           5.000000
75%          11.000000
max         109.000000
Name: word_count, dtype: float64

Saved cleaned data to data\processed\flipkart_cleaned.csv
```

## Stage 2 — Component 1: Sentiment classification

`scripts/02_sentiment_classification.py` — 38s

```
Rows used for classification: 122029
Train size: 97623, Test size: 24406

============================================================

Naive Bayes
----------------------------------------
Accuracy: 0.8474
Weighted Precision: 0.8187 | Recall: 0.8474 | F1: 0.8222
Macro F1: 0.6100 | Neutral F1: 0.1758
              precision    recall  f1-score   support

    negative       0.81      0.68      0.74      4472
     neutral       0.42      0.11      0.18      2014
    positive       0.87      0.97      0.92     17920

    accuracy                           0.85     24406
   macro avg       0.70      0.59      0.61     24406
weighted avg       0.82      0.85      0.82     24406


Decision Tree
----------------------------------------
Accuracy: 0.7249
Weighted Precision: 0.7946 | Recall: 0.7249 | F1: 0.7470
Macro F1: 0.5732 | Neutral F1: 0.2571
              precision    recall  f1-score   support

    negative       0.52      0.79      0.63      4472
     neutral       0.21      0.33      0.26      2014
    positive       0.93      0.75      0.83     17920

    accuracy                           0.72     24406
   macro avg       0.55      0.62      0.57     24406
weighted avg       0.79      0.72      0.75     24406


Logistic Regression
----------------------------------------
Accuracy: 0.7745
Weighted Precision: 0.8399 | Recall: 0.7745 | F1: 0.8002
Macro F1: 0.6394 | Neutral F1: 0.3137
              precision    recall  f1-score   support

    negative       0.72      0.75      0.73      4472
     neutral       0.23      0.49      0.31      2014
    positive       0.94      0.81      0.87     17920

    accuracy                           0.77     24406
   macro avg       0.63      0.68      0.64     24406
weighted avg       0.84      0.77      0.80     24406


Complement NB
----------------------------------------
Accuracy: 0.8142
Weighted Precision: 0.8243 | Recall: 0.8142 | F1: 0.8163
Macro F1: 0.6374 | Neutral F1: 0.2969
              precision    recall  f1-score   support

    negative       0.63      0.82      0.72      4472
     neutral       0.32      0.28      0.30      2014
    positive       0.93      0.87      0.90     17920

    accuracy                           0.81     24406
   macro avg       0.63      0.66      0.64     24406
weighted avg       0.82      0.81      0.82     24406


Comparison table saved to classifier_comparison.csv
                     Accuracy  Precision  Recall      F1  Macro_F1  Negative_F1  Neutral_F1
Naive Bayes            0.8474     0.8187  0.8474  0.8222    0.6100       0.7385      0.1758
Decision Tree          0.7249     0.7946  0.7249  0.7470    0.5732       0.6316      0.2571
Logistic Regression    0.7745     0.8399  0.7745  0.8002    0.6394       0.7329      0.3137
Complement NB          0.8142     0.8243  0.8142  0.8163    0.6374       0.7156      0.2969
Saved confusion_matrices.png

==============================================================
Best on WEIGHTED F1 : Naive Bayes (0.8222)
Best on MACRO F1    : Logistic Regression (0.6394)
Best on NEUTRAL F1  : Logistic Regression (0.3137)
==============================================================
NOTE: weighted and macro F1 disagree. Weighted F1 rewards doing well
on the dominant positive class; macro F1 weights all three classes
equally and so favours a model that handles neutral better. State
which objective was chosen and why, rather than quoting only one.

Justification: the final model is selected on weighted F1, which balances
precision and recall across the three imbalanced classes better than
accuracy alone. Macro F1 is reported alongside it because weighted F1 can
mask poor performance on the small neutral class.

==============================================================
ERROR ANALYSIS
==============================================================

--- Naive Bayes ---
Misclassified: 3,725 of 24,406 (15.3%)

Most frequent confusions (actual -> predicted):
   neutral -> positive  1,380  ( 37.0% of all errors)
  negative -> positive  1,339  ( 35.9% of all errors)
   neutral -> negative    410  ( 11.0% of all errors)
  positive -> negative    285  (  7.7% of all errors)
  positive -> neutral     203  (  5.4% of all errors)
  negative -> neutral     108  (  2.9% of all errors)

Error rate by review length (words):
     1-2:  14.2%
     3-5:  15.1%
    6-10:  15.8%
   11-20:  15.8%
     20+:  15.6%

Examples of the largest confusion (neutral -> positive):
  [3*] It will slowing down the mobile...
  [3*] Camera quality should be increased
  [3*] The cycle is good but the finishing is definitely not good But yeah I think for this amoun
  [3*] very good
  [3*] it's good one
  [3*] It is good ????

--- Logistic Regression ---
Misclassified: 5,504 of 24,406 (22.6%)

Most frequent confusions (actual -> predicted):
  positive -> neutral   2,492  ( 45.3% of all errors)
  positive -> negative    838  ( 15.2% of all errors)
  negative -> neutral     756  ( 13.7% of all errors)
   neutral -> positive    583  ( 10.6% of all errors)
   neutral -> negative    452  (  8.2% of all errors)
  negative -> positive    383  (  7.0% of all errors)

Error rate by review length (words):
     1-2:  19.1%
     3-5:  22.2%
    6-10:  23.9%
   11-20:  24.2%
     20+:  24.2%

Examples of the largest confusion (positive -> neutral):
  [4*] The nice one is their display but their range of connection to the phone is short .
  [5*] Affordable cost and efficient working
  [5*] Nys product as shown in the image...
  [5*] Cheap in cost but best
  [5*] Product is very good
  [4*] Delivery was too fast, product is too good and easy to use, but some little bit scratches 

Saved outputs/tables/sentiment_error_analysis.csv

Saved vectorizer and 4 models to outputs/models/
```

## Stage 3 — Component 2: Topic discovery

`scripts/03_topic_modeling.py` — 386s

```
Rows used for topic modeling: 84070

Discovered Topics (most probable terms):
Topic 0: like, super, poor, quality, machine, bought, properly, colour, think, just
Topic 1: use, good, tv, small, easy, size, perfect, day, working, works
Topic 2: worth, using, range, amazing, great, just, days, performance, overall, purchase
Topic 3: product, good, buy, quality, bad, don, delivery, installation, working, money
Topic 4: product, good, nice, flipkart, quality, money, happy, value, awesome, thanks
Topic 5: good, price, best, product, quality, sound, service, got, satisfied, low

Discovered Topics (most DISTINCTIVE terms):
Topic 0: like, super, poor, machine, bought, properly, colour, think
Topic 1: use, tv, small, easy, perfect, day, works, fine
Topic 2: worth, using, range, amazing, days, performance, overall, purchase
Topic 3: bad, don, installation, worst, ok, waste, picture, expected
Topic 4: nice, happy, value, thanks, thank, love, superb, looking
Topic 5: price, best, service, got, satisfied, low, bass, bit

Review count per topic:
dominant_topic
0     7008
1    13920
2     8904
3    15175
4    27290
5    11773
Name: count, dtype: int64

Soft (mixture-weighted) sentiment per topic:
       negative  neutral  positive
topic                             
0         0.237    0.100     0.662
1         0.212    0.112     0.676
2         0.187    0.082     0.731
3         0.359    0.100     0.541
4         0.069    0.054     0.877
5         0.145    0.088     0.767

Most negative topic -- hard assignment (argmax): 3 (46.4% negative)
Most negative topic -- soft assignment (mixture): 3 (35.9% negative)
ROBUST: both methods identify the same most-negative topic, so the
argmax simplification does not drive the headline finding.
Saved topic_sentiment_soft.csv
Saved topic_distribution.png
Saved topic_wordclouds.png

Topic vs Sentiment (proportion within each topic):
sentiment_label  negative   neutral  positive
dominant_topic                               
0                0.277825  0.116153  0.606022
1                0.209842  0.126868  0.663290
2                0.175314  0.079964  0.744721
3                0.463657  0.106161  0.430181
4                0.041407  0.043936  0.914657
5                0.115264  0.094538  0.790198

Saved reviews_with_topics.csv
```

## Stage 4 — Component 3: Low-information detection

`scripts/04_clustering_anomaly.py` — 59s

```
Rows loaded: 132410

Feature summary (before scaling):
      word_count  exclamation_ratio  log_text_freq   Rate
mean      10.086              0.005          1.852  3.986
std       14.347              0.080          2.166  1.416
min        0.000              0.000          0.693  1.000
50%        5.000              0.000          0.693  5.000
max      109.000             13.000          9.669  5.000

Selecting K (rejecting any solution with a cluster below 1% of 132410 rows):
  K=2: silhouette=0.4281   smallest cluster= 18.12%   ok
  K=3: silhouette=0.4845   smallest cluster= 15.49%   ok
  K=4: silhouette=0.5315   smallest cluster=  6.19%   ok
  K=5: silhouette=0.5364   smallest cluster=  0.14%   REJECTED (degenerate)
  K=6: silhouette=0.5071   smallest cluster=  0.14%   REJECTED (degenerate)
  K=7: silhouette=0.4994   smallest cluster=  0.03%   REJECTED (degenerate)
  K=8: silhouette=0.5109   smallest cluster=  0.03%   REJECTED (degenerate)

K selected: 4 (silhouette 0.5315, highest among non-degenerate solutions)
Final silhouette score: 0.5315

Cluster profiles:
         word_count  exclamation_ratio  log_text_freq   Rate  text_freq_global  mismatch_flag  n_reviews  pct_of_data  lowinfo_score
cluster                                                                                                                             
0             1.776              0.000          6.413  4.500          2349.228          0.007      19620       14.818          2.495
1             7.892              0.008          1.005  4.753             2.897          0.009      74460       56.234         -0.091
2            56.010              0.003          0.758  4.079             1.173          0.039       8196        6.190         -2.404
3             8.427              0.002          1.274  1.733            18.488          0.000      30134       22.758         -0.001

Saved cluster_profiles.csv

Sentiment composition within each cluster:
sentiment_label  negative  neutral  positive
cluster                                     
0                   0.030    0.082     0.888
1                   0.000    0.000     1.000
2                   0.130    0.101     0.769
3                   0.715    0.285     0.000
Saved cluster_sentiment.csv

Cluster 0 shows the strongest low-information profile: 19620 reviews (14.82% of the dataset), mean length 1.8 words, mean raw text repetition 2349x.
This indicates low information content, NOT confirmed inauthenticity.

Most frequently repeated review texts in the raw dataset:
Summary_clean  text_freq_global  cluster
         Good             15818        0
         good             15818        0
         GOOD             15818        0
         GooD             15818        0
         Nice              9049        0
         nice              9049        0
         NICE              9049        0
         NicE              9049        0

Saved clustering_analysis.png
Saved flipkart_with_clusters.csv
```

## Stage 5 — Validation against reference labels (LLM-generated)

`scripts/06_score_validation.py` — 27s

```
Labelled reviews scored: 200
Low-information cluster: 0
Reference labels marked low-information: 133 of 200 (66.5%)
Flag predicted low-information: 100 of 200 (50.0%)

====================================================
VALIDATION OF THE LOW-INFORMATION FLAG
Reference labels are LLM-generated, NOT human annotation.
====================================================
Precision : 0.980   of reviews the flag catches, this share are genuinely low-information
Recall    : 0.737   of genuinely low-information reviews, this share are caught
F1        : 0.841
Accuracy  : 0.815

Confusion matrix (rows = reference label, cols = flag):
                  flag: not flagged  flag: flagged
ref: informative                 65              2
ref: low-info                    35             98

                 precision    recall  f1-score   support

    informative       0.65      0.97      0.78        67
low-information       0.98      0.74      0.84       133

       accuracy                           0.81       200
      macro avg       0.81      0.85      0.81       200
   weighted avg       0.87      0.81      0.82       200

Saved outputs/tables/validation_results.csv

FALSE POSITIVES (2): flagged, but labelled informative
  Summary_clean  Rate
      Too small     3
worth for money     5

FALSE NEGATIVES (35): missed, but labelled low-information
                    Summary_clean  Rate  cluster
                Very nice quality     5        1
                    Not bad, good     3        3
                        Value for     5        1
                        Ok thanks     3        3
  Ratting 10/10...tnq Flipkart...     5        1
            It a very bad prodect     1        3
                          So cute     5        1
Very very very poor product .....     1        3
       Exlent.. Terrific purchase     5        1
              This product is bed     2        3

Saved validation/validation_errors.csv for the error-analysis section
Saved outputs/figures/validation_results.png

REMINDERS FOR THE REPORT:
1. The reference labels are LLM-generated, not human annotation. This is
   NOT human validation and must not be described as such.
2. The sample is balanced 100/100 by design while only ~14.8% of the full
   dataset is flagged, so these figures describe the sample, not the corpus.
3. The LLM labeller and the clustering may share a bias toward brevity,
   which would inflate agreement. Independence is weaker than it looks.
```

## Stage 6 — Topic coherence sweep

`scripts/07_topic_coherence.py` — 14603s

```
Documents available (>=4 words, same filter as script 03): 84,070
Documents used for this sweep: 25,000 (random subsample)
Vocabulary: 1970 terms
  k= 3: UMass coherence=-2.8529  (worst topic -2.9812)  perplexity=473.3
  k= 4: UMass coherence=-2.8106  (worst topic -3.1467)  perplexity=510.1
  k= 5: UMass coherence=-3.1019  (worst topic -3.4974)  perplexity=527.6
  k= 6: UMass coherence=-3.1112  (worst topic -3.7185)  perplexity=558.2
  k= 7: UMass coherence=-3.2810  (worst topic -3.9653)  perplexity=571.3
  k= 8: UMass coherence=-3.2703  (worst topic -4.0640)  perplexity=571.0
  k=10: UMass coherence=-3.2678  (worst topic -3.9732)  perplexity=609.6

================================================================
 k  umass_coherence  perplexity  worst_topic_coherence  selected
 3          -2.8529       473.3                -2.9812     False
 4          -2.8106       510.1                -3.1467      True
 5          -3.1019       527.6                -3.4974     False
 6          -3.1112       558.2                -3.7185     False
 7          -3.2810       571.3                -3.9653     False
 8          -3.2703       571.0                -4.0640     False
10          -3.2678       609.6                -3.9732     False
================================================================

Best UMass coherence at k=4
Script 03 currently uses k=6 (coherence -3.1112)

Top words per topic at the best k (4):
  Topic 0: product, good, flipkart, price, best, nice
  Topic 1: good, product, nice, quality, money, value
  Topic 2: buy, bad, quality, don, sound, tv
  Topic 3: working, use, using, easy, satisfied, worst

Top words per topic at k=6 (the value script 03 uses):
  Topic 0: product, buy, don, installation, using, worst
  Topic 1: small, size, like, good, got, water
  Topic 2: thank, tv, waste, sound, amazing, picture
  Topic 3: good, working, really, use, product, easy
  Topic 4: product, good, flipkart, best, quality, nice
  Topic 5: good, product, nice, quality, money, price

NOTE: perplexity is reported for completeness but correlates poorly
with human interpretability -- it typically keeps improving as k grows,
which is why coherence rather than perplexity drives the choice here.
NOTE: this sweep ran on a 25,000-document random subsample of
the 84,070 eligible reviews. It is used to RANK values of k; the
chosen k is refitted on the full corpus in 03_topic_modeling.py.

Saved outputs/figures/topic_coherence.png and outputs/tables/topic_coherence.csv
```

## Stage 7 — Pipeline architecture diagram

`scripts/08_pipeline_diagram.py` — 6s

```
Saved outputs\figures\pipeline_architecture.png
```
