# CBS3009 Course Project — Handoff for Literature Review and Presentation Slides

You are picking up a completed analytics project. The implementation is finished, tested and
verified. Two things are needed from you: **a Literature Review section** and **presentation
slides**. Do not change the code or the numbers.

---

## 0. READ THIS FIRST — a provenance correction that changes what can be claimed

The 200 validation labels behind the 0.980 precision figure were produced by **an LLM (Claude)
applying a written criterion**. They are **not** human annotation. No human has checked any of the
200 labels.

Earlier drafts of this project's documentation described them as "hand-labelled" and the result as
"validated against human judgement". **Both claims were wrong.** They have been corrected
throughout the code and docs, and they must not reappear in the report or the slides.

**Correct phrasing:** "validated against LLM-generated reference labels applying a pre-registered
criterion."
**Never write:** "hand-labelled", "human judgement", "human validation", "human annotation".

Why the result is still worth reporting: K-Means never reads the words — it uses only review
length, repetition frequency and star rating — while the labeller read only the text. The two
signals are genuinely different, so the agreement is not circular.

Why it is weaker than human annotation: an LLM's notion of "low-information" plausibly tracks
brevity, and brevity is also most of what the clustering keys on. A shared bias of that kind would
inflate agreement. Report the figure with that caveat attached.

If the student later hand-labels a subset and reports Cohen's kappa against the LLM labels, this
section must be updated to reflect that.

---

## 1. Context

- **Course:** CBS3009 — Advanced Social, Text and Media Analytics, VIT Vellore
- **Level:** Final-year B.Tech., 20 marks, mapped to CO5 and CO6
- **Project:** E-Commerce Review Analytics on a public Flipkart review dataset
- **Domain:** Text Analytics / Integrated Analytics

### Marking rubric (official)

| Component | Marks | CO |
|---|---|---|
| Problem Definition & Objectives | 2 | CO5 |
| Data Collection & Preprocessing | 3 | CO5 |
| Methodology & Algorithm Selection | 4 | CO5 |
| Implementation & Experimental Analysis | 4 | CO5 |
| Results, Evaluation & Interpretation | 3 | CO6 |
| Innovation / Problem-Solving | 2 | CO6 |
| Presentation & Demonstration | 2 | CO6 |

### Required report structure (exact)

1. Introduction — background, problem statement, motivation, objectives, scope
2. **Literature Review — relevant studies, existing approaches and gap** ← YOU WRITE THIS
3. Dataset & Methodology — dataset, collection, preprocessing, features, algorithms, architecture
4. Implementation & Results — experimental setup, implementation, results, tables, visualizations
5. Analysis & Discussion — interpretation, comparison, findings, limitations
6. Conclusion & Future Work — conclusions, contributions, enhancements
7. References

### Deliverables

| # | Deliverable | Status |
|---|---|---|
| 1 | Final Project Report | NOT STARTED |
| 2 | Source code / Jupyter Notebook | DONE |
| 3 | Dataset or documented dataset source | DONE |
| 4 | Results, evaluation tables and visualizations | DONE |
| 5 | Presentation slides | NOT STARTED ← YOU BUILD THIS |
| 6 | Working demonstration | DONE (interactive HTML walkthrough) |

---

## 2. Dataset

- **Source:** Kaggle, "Flipkart Products Review Dataset" by Mansi Thummar
  `https://www.kaggle.com/datasets/mansithummar67/flipkart-product-review-dataset`
- **Raw:** 189,874 rows x 5 columns, ~812 unique products; must be read with `encoding='latin1'`
- **78.1%** of raw reviews are rated 4 or 5 stars

| Column | Contents | Used for |
|---|---|---|
| `ProductName` | Full product title | Grouping by product |
| `Price` | Price with currency symbol and commas | Cleaned to integer; not modelled |
| `Rate` | Star rating, 1 (worst) to 5 (best) | Source of sentiment label; clustering feature |
| `Review` | Short headline, e.g. "Super!" | Not used — too short |
| `Summary` | Review body — **the main text field** | All three components |

**Ethics / provenance.** Live scraping was attempted and deliberately abandoned: Amazon.in blocked
access with CAPTCHA; Flipkart's API gateway returned **HTTP 403 with an `x-bifrost-error` header**
to non-browser clients even with a valid session cookie. A public Kaggle dataset was used instead,
which the guidelines permit. Only review text, ratings, product names and prices are used — no
personal identifiers.

---

## 3. Preprocessing — 189,874 → 132,410 rows

| Step | Defect addressed | Rows dropped |
|---|---|---|
| 1 | Non-numeric ratings (product text leaked in via unescaped commas) | 5 |
| 2 | Missing review text | 13 |
| 3 | Mojibake from encoding corruption | — |
| 4 | Price parsed to integer | — |
| 5 | **Record text repetition frequency BEFORE deduplication** | — |
| 6 | Exact duplicate (product, review) pairs removed | 57,446 |
| 7 | Derive sentiment label and word count | — |

Final: **132,410 rows** — positive 98,170 / negative 23,207 / neutral 11,033. Median length 5
words. **56.8%** of reviews share their text with at least one other. The most common review text
is **"Good", appearing 15,818 times**.

**Label mapping:** 4–5 stars → positive, 3 → neutral, 1–2 → negative. There is no human-annotated
sentiment, so the classifiers learn to predict the *star rating* from text.

**Why step 5 precedes step 6 — a real bug found and fixed.** Step 6 deletes duplicates. The
original code counted repetition *after* that deletion, so the count returned 1 for almost every
row; the feature was constant and silently reduced Component 3 from four working features to two.

---

## 4. Component 1 — Sentiment Classification

Rows with >=2 words (122,029), stratified 80/20 split (97,623 / 24,406), `random_state=42`.
TF-IDF, 5,000 features, 1–2 grams, English stop words, `min_df=5`. TF-IDF chosen over raw counts
because it down-weights words appearing across most reviews; bigrams keep "not good" intact.

| Model | Accuracy | Weighted F1 | Macro F1 | Negative F1 | Neutral F1 |
|---|---|---|---|---|---|
| Multinomial Naive Bayes | 0.8474 | **0.8222** | 0.6100 | 0.7385 | 0.1758 |
| Decision Tree (depth 30, balanced) | 0.7249 | 0.7470 | 0.5732 | 0.6316 | 0.2571 |
| Logistic Regression (balanced) | 0.7745 | 0.8002 | **0.6394** | 0.7329 | **0.3137** |
| Complement NB | 0.8142 | 0.8163 | 0.6374 | 0.7156 | 0.2969 |

**The key finding: the two criteria disagree.** Naive Bayes wins on weighted F1; Logistic
Regression wins on macro F1 and nearly doubles neutral F1 (0.176 → 0.314). Weighted F1 is dominated
by the positive class (73% of the test set), so a model can score well while ignoring the neutral
class — Naive Bayes predicts "neutral" only **535 times out of 24,406**. Logistic Regression
predicts it 4,227 times and recovers 979 of 2,014 true neutrals.

**Error analysis.** NB: 73% of errors are *something → positive*. LR: 45.3% are *positive →
neutral* — `class_weight='balanced'` over-corrects. Error rate is **flat across review length**
(14.2% for 1–2 words, 15.6% for 20+), so short text is not the cause; label ambiguity is. A review
reading **"very good"** carries a 3-star rating.

---

## 5. Component 2 — Topic Discovery (LDA)

Reviews with >=4 words (84,070), `CountVectorizer` 2,000 features, `min_df=10`, `max_df=0.5`,
LDA with 6 components, `max_iter=15`, online learning, `random_state=42`.

**Raw counts, not TF-IDF** — LDA is a generative model over word counts; TF-IDF breaks that
assumption. The contrast with Component 1 is a deliberate methodological point.

**Topics labelled by DISTINCTIVE terms**, not most-probable terms, because the most probable words
are corpus-wide generics ("product", "good", "quality") that top every topic and distinguish
nothing.

| Topic | Distinctive terms | Reviews | Negative | Interpretation |
|---|---|---|---|---|
| 0 | like, super, poor, machine, bought, properly, colour | 7,008 | 27.8% | Product defects |
| 1 | use, tv, small, easy, perfect, day, works, fine | 13,920 | 21.0% | Usability / fit |
| 2 | worth, using, range, amazing, days, performance | 8,904 | 17.5% | Value / performance |
| 3 | **bad, don, installation, worst, ok, waste, expected** | 15,175 | **46.4%** | Complaints / installation |
| 4 | nice, happy, value, thanks, love, superb, looking | 27,290 | 4.1% | Gratitude |
| 5 | price, best, service, got, satisfied, low, bass | 11,773 | 11.5% | Price / service |

**Headline:** topic 3 carries ten times the negativity of topic 4, and its distinctive vocabulary
is service-and-expectation language. **Service-side issues, not product quality, drive negative
sentiment** — which routes to a logistics team, not a product team.

**Robustness check (passed).** Re-weighting sentiment by LDA's full soft mixture instead of
`argmax` gives the same most-negative topic (3), at 35.9% vs 46.4%.

**Why k=6 when coherence says k=4.** UMass coherence over k in {3,4,5,6,7,8,10}: −2.85, **−2.81
(best)**, −3.10, **−3.11 (retained)**, −3.28, −3.27, −3.27. k=6 kept because at k=4 two of four
topics are near-duplicate generic-praise themes, and UMass coherence is known to favour fewer,
broader topics. **The disagreement is reported, not hidden** — present this as a strength.

---

## 6. Component 3 — Low-Information Review Detection

**Framing that must be preserved exactly:** this identifies reviews that are statistically unusual
and carry little information. It is **NOT fake-review detection**. The dataset has no authenticity
ground truth. Always say "low-information" or "anomalous", never "fake".

Features: word count, exclamation ratio, log(1 + global repetition frequency), star rating.

- **Log transform** because raw frequency spans 1–15,818 with median 1 and would otherwise dominate
  the Euclidean distance.
- **`mismatch_flag` excluded** — a binary flag true for ~0.9% of rows splits the space into two
  perfectly separable sheets, inflating silhouette (K=2 scored 0.83) while recovering only the flag.

**K selection:** highest silhouette among non-degenerate solutions, rejecting any partition with a
cluster below 1% of the data. K=5 scored higher (0.5364) by isolating 191 rows (0.14%).
**K=4 selected, silhouette 0.5315.**

| Cluster | Reviews | Share | Words | Mean repeats | Rating | Reading |
|---|---|---|---|---|---|---|
| 0 | 19,620 | **14.8%** | 1.8 | **2,349x** | 4.50 | Low-information |
| 1 | 74,460 | 56.2% | 7.9 | 2.9x | 4.75 | Ordinary positive |
| 2 | 8,196 | 6.2% | 56.0 | 1.2x | 4.08 | Long and detailed |
| 3 | 30,134 | 22.8% | 8.4 | 18.5x | 1.73 | Genuine complaints |

---

## 7. Validation — see section 0 on label provenance

200 reviews (100 flagged, 100 not) exported shuffled with cluster IDs hidden in a separate key
file, labelled against a criterion written down **before** labelling began. Labels: 133
low-information, 67 informative. **Produced by an LLM, not a human.**

| Metric | Value |
|---|---|
| Precision | **0.980** |
| Recall | 0.737 |
| F1 | 0.841 |
| Accuracy | 0.815 |

Confusion matrix: 65 TN, **2 FP**, 35 FN, 98 TP.

**Agreement by cluster — a clean monotonic gradient:**

| Cluster | In sample | Labelled low-information |
|---|---|---|
| 0 (flagged) | 100 | **98.0%** |
| 1 (ordinary positive) | 69 | 39.1% |
| 2 (long and detailed) | 3 | **0.0%** |
| 3 (complaints) | 28 | 28.6% |

**Error analysis.** Only 2 false positives. Of 35 misses, **27 sit in the ordinary-positive
cluster** at mean 3.3 words, just outside cluster 0's 1.8-word centre — a **boundary effect**, not
the sentiment bias one might predict. Secondary effect: 8 generic complaints are pulled into the
complaints cluster because rating is a clustering feature, so the flag is biased toward catching
generic *praise*.

**Claim to make:** highly precise (0.980) but conservative (0.737) — a high-confidence *subset*,
and checked against LLM-generated rather than human labels.

---

## 8. Integrated findings (the Innovation argument, 2 marks)

- **Dissatisfaction is operational, not product-level.** The installation/delivery topic carries
  ten times the negativity of the general-satisfaction topic.
- **A seventh of the review corpus is noise.** One-word text repeated thousands of times across
  unrelated products, inflating rating averages.
- **The quality filter looks reliable**, at 0.980 precision against independent external labels —
  with the provenance caveat in section 0.

**Innovation claim:** not a classifier, but an integrated pipeline in which an unsupervised quality
filter is validated against independent external labels under a pre-registered criterion, then used
to qualify the other two components' findings — plus three methodological defects found, diagnosed
and documented during development.

---

## 9. Limitations — state these honestly

1. **Labels derived from star ratings, not annotation** — classifiers predict ratings, not
   human-judged sentiment.
2. **Neutral class weak across all four models** (best F1 0.314); partly inherent to the problem
   definition.
3. **Topic labels are analyst interpretations** of term lists, not model output.
4. **`argmax` discards LDA's soft mixture** (mitigated by the robustness check); vocabulary capped
   at 2,000 terms.
5. **Cluster–sentiment relationship is partly circular** — rating is both a clustering input and
   the label source.
6. **No ground truth for authenticity.** Low-information != fake.
7. **Validation labels are LLM-generated, not human annotation.** No human has checked any label.
   Inter-rater agreement unmeasured. This is external validation, not human validation.
8. **Validation sample balanced 100/100 by design** while the flagged cluster is 14.8% of the
   corpus, so precision describes the sample, not the dataset.
9. **Deduplication is case-sensitive while repetition counting lowercases** — "Good" and "good"
   survive as separate rows.
10. **Word-pair co-occurrence is weaker than the topic story implies** ("delivery" co-occurs with
    "installation" in 16.9% of delivery reviews but with "bad" in only 2.6%).

### Three methodological corrections made during development (good report material)

1. **Dead feature.** `duplicate_count` computed after deduplication was constant; the clustering ran
   on two effective features. Fixed by moving the count upstream. **The flagged group grew from 4.8%
   to 14.8%.** Old output kept in `archive/`.
2. **Scale domination and silhouette inflation.** The corrected feature spanned 1–15,818 and
   dominated the distance metric; the binary flag inflated silhouette. Fixed by log transform and by
   demoting the flag to a reported statistic.
3. **Metric/justification mismatch.** Script 02 selected the best model by *accuracy* while claiming
   *weighted F1*. Now explicitly weighted F1, with macro F1 reported alongside.

---

## 10. What exists

```
ASTAMA/
├── data/raw/          flipkart_product.csv (download from Kaggle)
├── data/processed/    cleaned data, cluster and topic assignments (~109 MB, regenerable)
├── scripts/           01-08 pipeline + 5 tooling scripts
├── notebook/          Flipkart_Review_Analytics.ipynb — 20 code cells, executed, 0 errors
├── outputs/figures/   7 PNG figures, all titled with labelled axes
├── outputs/tables/    8 result tables as CSV
├── outputs/models/    4 trained classifiers + TF-IDF vectorizer
├── validation/        sample, labels, key, written criterion
├── demo/              index.html — 11-stage interactive walkthrough (deliverable 6)
├── archive/           superseded v1 clustering results
├── docs/              PROJECT_HANDOFF.md, results_log.md, HANDOFF_PROMPT.md
└── report/            empty — the report goes here
```

**Figures for the slides:**

| File | Shows |
|---|---|
| `pipeline_architecture.png` | Full pipeline, three parallel components, validation loop |
| `confusion_matrices.png` | All four classifiers; NB's near-empty neutral column is visible |
| `topic_distribution.png` | Topic sizes and negativity, labelled by distinctive terms |
| `topic_wordclouds.png` | Six topic word clouds |
| `topic_coherence.png` | Coherence by k; green = best score, orange = retained |
| `clustering_analysis.png` | K selection, cluster sizes, structure, sentiment composition |
| `validation_results.png` | Flag vs reference labels, and the four metrics |

**Verification:** `scripts/verify_submission.py` runs automated consistency checks. A full pipeline
re-run reproduces all 11 output files byte-for-byte.

**Environment:** Python 3.13.6 on Windows, pandas 3.0.1, scikit-learn 1.8.0, matplotlib 3.10.5,
seaborn 0.13.2, wordcloud 1.9.6, joblib 1.5.2, nbformat/nbclient/ipykernel. `random_state=42`.

---

## 11. WHAT IS NEEDED

### A. Literature Review (report section 2)

Position this project against existing work and identify the gap. Suggested coverage:

- **Text representation:** TF-IDF and the vector space model (Salton & Buckley, 1988)
- **Text classification:** Naive Bayes for text (McCallum & Nigam, 1998); Complement Naive Bayes
  for imbalanced text (Rennie et al., 2003) — directly relevant, it is one of the four compared
- **Topic modelling:** LDA (Blei, Ng & Jordan, 2003); topic coherence and the UMass measure
  (Mimno et al., 2011) — relevant because the coherence sweep is implemented and reported
- **Clustering and validation:** k-means (MacQueen / Lloyd); silhouette analysis (Rousseeuw, 1987)
- **Review quality and opinion spam:** Jindal & Liu (2008); Ott et al. (2011) — **critical for the
  gap argument**
- **Aspect-based sentiment analysis** as the natural extension (SemEval ABSA tasks)

**The gap argument.** Existing opinion-spam work frames the problem as *fake* review detection,
which requires labelled deceptive data — Ott et al. built a crowdsourced deceptive corpus precisely
because such labels do not occur naturally. This dataset has no authenticity ground truth, so a
fraud claim could not be evaluated. This project therefore addresses the weaker but **evaluable**
problem of information content, under a criterion fixed in advance. Present this as a deliberate,
defensible scoping decision.

**Note on the validation when writing this section:** do not claim human annotation. The labels are
LLM-generated. If anything, the literature on annotation quality and inter-rater agreement is worth
citing to explain *why* that distinction matters and what would be needed to close it.

**Constraints:** cite only works you are confident exist and can describe accurately. Do not
fabricate page numbers, DOIs, or specific claims. Accuracy matters more than volume.

### B. Presentation slides

Roughly 10–12 slides. Suggested flow: title → problem and objectives → dataset and ethics →
preprocessing with row counts → pipeline architecture → Component 1 → Component 2 → Component 3 →
validation → integrated insights → limitations → conclusion and future work.

**Rules:**
- Every number must come from the tables above — do not invent, round differently, or estimate
- Never describe Component 3 as fake-review detection
- Never describe the validation as human
- Include the limitations slide; the rubric rewards presenting limitations
- The duplicate-count correction (4.8% → 14.8%) deserves prominence — it is the strongest evidence
  of analytical decision-making rather than execution of a ready-made notebook
- Present the weighted-vs-macro F1 disagreement as a finding, not a problem

### C. Information not available

The guidelines describe this as a **team-based** project and require the team's own contribution to
be clearly identified. Whether this is solo or a team, and the member names, is unknown.
**Ask the user before writing any title page or contribution statement — do not invent names.**

### D. Viva questions the student must be able to answer

- Why TF-IDF for classification but raw counts for LDA?
- Why does Naive Bayes win on weighted F1 but lose on macro F1?
- Why is the neutral class so hard, and how is the label definition partly responsible?
- What does silhouette measure, and why K=4 when K=5 scored higher?
- Why was `mismatch_flag` removed from the clustering features?
- Why did coherence favour k=4, and why was k=6 kept anyway?
- Is the flagged cluster fake reviews? (No — and why not.)
- Why must duplicate text be counted before deduplication?
- Who produced the validation labels, and what does that mean for the 0.980 figure?
