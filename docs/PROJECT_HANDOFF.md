# CBS3009 Course Project — Handoff Document

Purpose: full context of everything done so far, so work can continue without re-explaining.

**Last updated:** 2026-10-07 — Component 3 (clustering) was corrected and re-run. All
Component 3 numbers below supersede any earlier version of this document.

---

## 1. Who / what / where

- **Student:** Hetarth, VIT Vellore
- **Course:** CBS3009 (Social, Text and Media Analytics), course project worth **20 marks**
- **Local environment:** Windows, PowerShell, Python 3.13.6 (pandas 3.0.1, scikit-learn 1.8.0)
- **Project folder:** `C:\Users\HETARTH\OneDrive\Desktop\ASTAMA`
- **Deliverable style:** a working analytics pipeline + report (6-section structure) + demo

## 2. Course guideline requirements

> **UNVERIFIED.** The guidelines `.pptx` is **not in the project folder**. The table below is a
> reconstruction from memory and has never been checked against the source. Obtain the pptx
> and re-verify before writing the report.

| Rubric item | Marks | Status |
|---|---|---|
| Problem Definition & Objectives | 2 | ⬜ not written |
| Data Collection & Preprocessing | 3 | ✅ step 1 |
| Methodology & Algorithm Selection | 4 | ✅ steps 2–4 |
| Implementation & Experimental Analysis | 4 | ✅ steps 2–4 |
| Results, Evaluation & Interpretation | 3 | ⬜ metrics computed, interpretation not written |
| Innovation / Problem-Solving (CO6) | 2 | ⬜ components exist, argument not written |
| Presentation & Demonstration | 2 | ⬜ nothing |

Key constraints: **at least 3 major analytical components**, **evaluation metrics reported and
interpreted**, **compare 2+ methods**, **dataset documentation**, **ethics/ToS note**, justify
the final model choice.

## 3. Decision history

1. **Original idea:** Cross-Source Omission Analyzer. Dropped — under 3 analytical components,
   no evaluation metrics.
2. **Pivot to E-Commerce Review Analytics:** sentiment + topic modelling + anomaly clustering.
3. **Live scraping abandoned.** Amazon.in blocked by CAPTCHA. Flipkart's internal endpoint
   (`https://1.rome.api.flipkart.com/api/4/page/fetch`) returned **403 with
   `x-bifrost-error: true`** to plain `requests` even with a browser cookie; would need
   Selenium/Playwright non-headless. Too risky for the deadline.
   - `flipkart_scraper_draft.py` contained a pasted session cookie. **It is no longer in the
     folder.** Do not reintroduce it.
4. **Final data decision:** public Kaggle dataset (no scraping risk, allowed by guidelines).

## 4. Dataset

- **Source:** Kaggle, "Flipkart Products Review Dataset" by Mansi Thummar
  `https://www.kaggle.com/datasets/mansithummar67/flipkart-product-review-dataset`
- **File:** `flipkart_product.csv` — 189,874 rows × 5 columns
- **Columns:** `ProductName`, `Price`, `Rate` (1–5), `Review` (short tag), `Summary` (free-text
  body — the main text field)
- **Products:** ~812 unique, many categories
- **Quirks:** must be read with `encoding='latin1'`; a few rows have non-numeric `Rate`
  (product text leaked in via unescaped commas); many exact duplicate reviews; rating
  distribution heavily skewed to 5 stars.

## 5. Pipeline — files and what each does

**Folder layout (restructured 2026-10-07):**

```
ASTAMA/
├── data/raw/          flipkart_product.csv
├── data/processed/    flipkart_cleaned.csv, flipkart_with_clusters.csv,
│                      reviews_with_topics.csv        (~109 MB, all regenerable)
├── scripts/           01-07
├── outputs/           figures/  tables/  models/
├── validation/        sample, labels, key, instructions
├── archive/v1_clustering/   superseded results kept as before/after evidence
├── docs/              PROJECT_HANDOFF.md, results_log.md
├── report/            (empty)
└── notebook/          (empty)
```

Every script resolves paths from its own file location via
`ROOT = Path(__file__).resolve().parent.parent`, so the working directory does not matter.
Verified by running the full pipeline from `C:\Users\HETARTH` with identical results.

Run order, from the project folder:

```
python scripts/01_preprocessing.py
python scripts/02_sentiment_classification.py
python scripts/03_topic_modeling.py
python scripts/04_clustering_anomaly.py
python scripts/05_export_validation_sample.py    # then label by hand
python scripts/06_score_validation.py
python scripts/07_topic_coherence.py             # standalone sweep, ~13 min
python scripts/08_pipeline_diagram.py            # architecture figure
```

**Tooling** (not part of the analysis, but part of keeping it honest):

| Script | Purpose |
|---|---|
| `build_notebook.py` | Generates the notebook. Parses every code cell before writing, so a mangled escape cannot reach a 9-minute execution. |
| `run_notebook.py` | Executes the notebook via the `nbclient` API. The `jupyter nbconvert` CLI is not on PATH here and **exits 0 while reporting "command not found"**, which silently leaves the notebook unexecuted. |
| `regenerate_results_log.py` | Runs all 7 stages and rewrites `docs/results_log.md` from verbatim output. Fails loudly instead of writing a partial log. |
| `verify_submission.py` | 55 checks: files present, dependencies import, notebook fully executed with no errors, notebook/demo numbers agree with the output tables, results log not stale, figures present and referenced. Exits non-zero on failure. |

`07_topic_coherence.py --plot-only` redraws its figure from the saved table without refitting the
seven LDA models.

**Two traps worth remembering**, both of which cost time in development:
- Piping a command through `tail` replaces its exit code with `tail`'s, so a failed run looks
  like a success. Check exit codes before the pipe.
- Running the coherence sweep and the notebook at the same time starves both; the notebook's LDA
  cell hit a 40-minute timeout. Run heavy jobs sequentially.

### `01_preprocessing.py`
`data/raw/flipkart_product.csv` → `data/processed/flipkart_cleaned.csv`

1. Load with `latin1`
2. Coerce `Rate` to numeric, drop invalid, keep 1–5 (dropped **5** rows)
3. Drop missing `Summary` (dropped **13** rows)
4. Strip non-ASCII mojibake → `*_clean` columns
5. Clean `Price` to integer (`Price_clean`)
6. **Step 5a (added 2026-10-07): record pre-deduplication text frequency.**
   `text_freq_global` = times this exact text occurs anywhere in the dataset;
   `text_freq_product` = times it repeats on the same product. **Must run before dedup.**
7. Drop exact duplicates on (`ProductName_clean`, `Summary_clean`) (dropped **57,446** rows)
8. Derive `sentiment_label` from `Rate`: 4–5 positive, 3 neutral, 1–2 negative
9. Add `word_count`

Result: **132,410 rows**. Labels: positive 98,170 / negative 23,207 / neutral 11,033.
Word count median 5, mean 10.09, max 109.
Pre-dedup repetition: **56.8%** of reviews share text with at least one other; the most common
text ("Good") occurs **15,818** times.

### `02_sentiment_classification.py` (Component 1)
- `word_count >= 2` → **122,029 rows**; stratified 80/20 split (97,623 train / 24,406 test,
  `random_state=42`)
- TF-IDF: `max_features=5000`, `ngram_range=(1,2)`, `stop_words='english'`, `min_df=5`
- **Four classifiers compared** (extended 2026-10-07 from the original two)
- Outputs: `outputs/tables/classifier_comparison.csv`, `outputs/figures/confusion_matrices.png`,
  `outputs/models/` (vectorizer + 4 models)

| Model | Accuracy | Weighted F1 | Macro F1 | Negative F1 | Neutral F1 |
|---|---|---|---|---|---|
| Naive Bayes | 0.8474 | **0.8222** | 0.6100 | 0.7385 | 0.1758 |
| Decision Tree | 0.7249 | 0.7470 | 0.5732 | 0.6316 | 0.2571 |
| Logistic Regression | 0.7745 | 0.8002 | **0.6394** | 0.7329 | **0.3137** |
| Complement NB | 0.8142 | 0.8163 | 0.6374 | 0.7156 | 0.2969 |

**The two criteria disagree, and that disagreement is the most interesting result here:**
Naive Bayes wins on weighted F1, but Logistic Regression wins on macro F1 and nearly doubles
neutral F1 (0.176 → 0.314). Weighted F1 is dominated by the 73%-majority positive class; macro
F1 weights all three classes equally. The report must state which objective was chosen and why,
rather than quoting one number.

Recommended framing: Naive Bayes is the better model if overall accuracy on the realistic class
distribution is the goal; Logistic Regression is the better model if balanced performance across
all three classes matters. Pick one explicitly and justify it.

**Fixed 2026-10-07:** the script previously selected the "best" model by *accuracy* while the
printed justification claimed *weighted F1*. With two models they agreed; with four they need
not. Selection is now explicitly on weighted F1, with macro and neutral F1 reported alongside.

### `06_score_validation.py` — external validation of the clustering flag

200 blind, shuffled reviews labelled on 2026-10-07 (133 low-information, 67 informative).

> **LABEL PROVENANCE — READ FIRST.** These labels were produced by an **LLM (Claude)**
> applying the written criterion in `validation/LABELLING_INSTRUCTIONS.md`. They are **not**
> human annotation. Earlier versions of this document described them as "hand-labelled" and
> the result as "human judgement"; that was wrong and is corrected throughout.
>
> The comparison remains meaningful — K-Means never reads the words (it uses length,
> repetition frequency and rating) while the labeller read only the text, so agreement is not
> circular. But it is weaker than human annotation: an LLM's notion of "low-information"
> plausibly tracks brevity, which is also most of what the clustering keys on, and a shared
> bias of that kind inflates agreement.
>
> Say "validated against LLM-generated reference labels applying a pre-registered criterion".
> Never say "hand-labelled", "human judgement" or "human validation".

| Metric | Value |
|---|---|
| Precision | **0.980** |
| Recall | 0.737 |
| F1 | 0.841 |
| Accuracy | 0.815 |

Confusion matrix: 65 TN, 2 FP, 35 FN, 98 TP.

**Human agreement by cluster** — a clean monotonic gradient, and the strongest evidence that the
clustering found something real:

| Cluster | In sample | Human says low-info |
|---|---|---|
| 0 (flagged) | 100 | **98.0%** |
| 1 (ordinary positive) | 69 | 39.1% |
| 2 (long detailed) | 3 | **0.0%** |
| 3 (complaints) | 28 | 28.6% |

**Error analysis.** Only 2 false positives (`Too small`, `worth for money` — both borderline
calls). Of the 35 false negatives, **27 sit in cluster 1** (mean 3.3 words, mean rating 4.89) and
only 8 in cluster 3. So the recall gap is mainly a **boundary effect** — short generic praise just
outside cluster 0's 1.78-word centre — not the sentiment bias originally predicted. The sentiment
bias is real but secondary: 8 generic complaints (`Bad poor quality`, `This product is bed`) are
pulled into the complaints cluster because `Rate` is a clustering feature.

**Claim to make:** the flag is highly precise (0.98) but conservative (recall 0.74) — it
identifies a high-confidence subset of low-information reviews, not all of them.

**Caveats:**
1. The labels are LLM-generated, not human annotation (see the provenance note above).
2. The sample is balanced 100/100 by design while only 14.8% of the dataset is flagged, so
   these figures describe the sample, not the dataset.
3. One labeller only, so Cohen's kappa is unmeasured and the criterion's reproducibility is
   untested — across people and across labellers of any kind.

### `03_topic_modeling.py` (Component 2)
- `word_count >= 4` → **84,070 rows**
- `CountVectorizer(max_features=2000, stop_words='english', min_df=10, max_df=0.5)`
- `LatentDirichletAllocation(n_components=6, max_iter=15, learning_method='online', random_state=42)`
- Dominant topic per review via argmax
- Outputs: `topic_distribution.png`, `topic_wordclouds.png`, `topic_sentiment_crosstab.csv`,
  `reviews_with_topics.csv`

**Topic labels now use DISTINCTIVE terms, not merely the most probable ones.** The most probable
words are dominated by corpus-wide generics ("product", "good", "quality") that sit at the top of
every topic and distinguish nothing. Distinctiveness ranks a term by how much more probable it is
in one topic than across topics on average. This changed the interpretations materially — topic 3
previously showed as "product, good, buy", which did not support calling it a complaints topic.

| Topic | Distinctive terms | Reviews | Neg | Neu | Pos | Interpretation |
|---|---|---|---|---|---|---|
| 0 | like, super, poor, machine, bought, properly, colour | 7,008 | 27.8% | 11.6% | 60.6% | Product defects / quality |
| 1 | use, tv, small, easy, perfect, day, works, fine | 13,920 | 21.0% | 12.7% | 66.3% | Usability / fit |
| 2 | worth, using, range, amazing, days, performance, overall | 8,904 | 17.5% | 8.0% | 74.5% | Value / performance |
| 3 | **bad, don, installation, worst, ok, waste, picture, expected** | 15,175 | **46.4%** | 10.6% | 43.0% | Complaints / installation |
| 4 | nice, happy, value, thanks, thank, love, superb, looking | 27,290 | 4.1% | 4.4% | **91.5%** | Gratitude / satisfaction |
| 5 | price, best, service, got, satisfied, low, bass, bit | 11,773 | 11.5% | 9.5% | 79.0% | Price / service |

Key insight: Topic 3 has by far the highest negative rate, and its distinctive terms
(`installation`, `waste`, `worst`, `expected`) are service- and expectation-related rather than
product-feature related.

**Robustness check (added 2026-10-08).** Assigning each review to one topic by `argmax` discards
LDA's soft mixture. Re-weighting sentiment by the full topic distribution instead gives:

| | Most negative topic | Negative share |
|---|---|---|
| Hard assignment (argmax) | 3 | 46.4% |
| Soft assignment (mixture) | 3 | 35.9% |

Both methods identify the same topic, so the headline finding is **not** an artefact of the
simplification. Saved to `outputs/tables/topic_sentiment_soft.csv`. The absolute percentage is
lower under soft weighting, as expected — report the method alongside the number.

**Re-verified 2026-10-08: topic sizes and crosstab reproduce exactly.**

### `04_clustering_anomaly.py` (Component 3) — **REWRITTEN 2026-10-07**

Features: `word_count`, `exclamation_ratio`, `log_text_freq` (= `log1p(text_freq_global)`),
`Rate`. Standardised with `StandardScaler`. K searched over 2–8.

**K selection rule:** highest silhouette among **non-degenerate** solutions, where a solution is
rejected if any cluster holds under 1% of the data.

| K | Silhouette | Smallest cluster | Verdict |
|---|---|---|---|
| 2 | 0.4281 | 18.12% | ok |
| 3 | 0.4845 | 15.49% | ok |
| **4** | **0.5315** | 6.19% | **selected** |
| 5 | 0.5364 | 0.14% | rejected (degenerate) |
| 6 | 0.5071 | 0.14% | rejected |
| 7 | 0.4994 | 0.03% | rejected |
| 8 | 0.5109 | 0.03% | rejected |

**Cluster profiles (K=4, silhouette 0.5315):**

| Cluster | n | % | word_count | log_text_freq | mean raw repeats | Rate | mismatch | Reading |
|---|---|---|---|---|---|---|---|---|
| 0 | 19,620 | 14.82% | 1.78 | 6.41 | **2,349** | 4.50 | 0.007 | **Low-information: one-word, massively repeated** |
| 1 | 74,460 | 56.23% | 7.89 | 1.01 | 2.9 | 4.75 | 0.009 | Ordinary positive reviews |
| 2 | 8,196 | 6.19% | **56.01** | 0.76 | 1.2 | 4.08 | 0.039 | Long, detailed, unique reviews |
| 3 | 30,134 | 22.76% | 8.43 | 1.27 | 18.5 | 1.73 | 0.000 | Genuine complaints |

Outputs: `cluster_profiles.csv`, `cluster_sentiment.csv`, `clustering_analysis.png`,
`flipkart_with_clusters.csv`. Previous version preserved as `cluster_profiles_v1.csv` and
`clustering_analysis_v1.png` for the before/after discussion.

**Headline finding:** 14.8% of reviews are one-word texts repeated an average of 2,349 times
across the raw dataset, cleanly separated from both detailed reviews and genuine complaints.

## 6. The Component 3 correction (worth writing up as methodology)

Three defects were found and fixed on 2026-10-07. The before/after is itself reportable material
under Experimental Analysis.

1. **Dead feature.** `duplicate_count` was computed in step 4, *after* step 1 had already removed
   exact duplicates — so it returned 1 for nearly every row (cluster mean was exactly
   `1.000000`). The clustering was effectively running on `word_count` and `Rate` alone.
   *Fix:* count moved upstream into `01_preprocessing.py`, before dedup, and measured across all
   products rather than within one.
2. **Scale domination.** The corrected feature spans 1–15,818 with a median of 1. On the raw
   scale it overwhelmed every other feature in the Euclidean distance.
   *Fix:* `log1p` transform.
3. **Binary feature hijacking silhouette.** `mismatch_flag` is true for only ~0.9% of rows. A
   near-constant binary feature creates two perfectly separable sheets, so K=2 scored 0.83 while
   producing a split that merely recovered the flag. An intermediate run also flagged 99.13% of
   the dataset as suspicious, because the cluster-selection line summed raw column means of
   wildly different scales.
   *Fix:* `mismatch_flag` removed from clustering inputs and retained as a per-cluster
   descriptive statistic; cluster selection now uses z-scored features.

## 7. Remaining caveats — state these honestly in the report

1. **Sentiment composition per cluster is partly circular.** `sentiment_label` is derived from
   `Rate` (step 1), and `Rate` is a clustering input. So cluster 1 being 100% positive and
   cluster 3 being 0% positive is partly built in by construction, not a discovered finding.
   Report `cluster_sentiment.csv` as a description of the clusters, not as validation.
2. **No ground truth for authenticity.** Nothing here confirms any review is fake. Always say
   "low-information / anomalous", never "fake review detection".
3. **Validation exists** (200 labelled reviews, precision 0.980 / recall 0.737) but **the labels
   are LLM-generated, not human**. It is external validation, not human validation. Only one
   labeller, so inter-rater agreement is unmeasured. The criterion was also refined mid-task
   ("naming the product category counts as informative") — disclosed in
   `validation/LABELLING_INSTRUCTIONS.md` and it must be disclosed in the report too.
3a. **Recommended remedy:** have a human label a 50-row subset blind and report Cohen's kappa
   between the human and the LLM labels. That converts an unquantified weakness into a
   measured one and restores a genuine human element.
4. **Neutral class is weak across all four models** — best neutral F1 is 0.314 (Logistic
   Regression), worst 0.176 (Naive Bayes). 3-star reviews genuinely mix positive and negative
   language, and the label itself is derived from the star rating rather than from the text, so
   some of this ceiling is built into the problem definition. Report as a real limitation.
4a. **The sentiment labels are not independent ground truth.** `sentiment_label` is derived from
   `Rate`, so the classifiers are learning to predict the star rating from the review text, not
   to predict human-judged sentiment. This is a standard and acceptable approach, but say so
   plainly rather than implying the labels came from annotation.
5. **Topic labels are human interpretations.** Generic words ("product", "good", "quality") smear
   across topics because the corpus spans 800+ categories.
   **Coherence now exists and it disagrees with k=6** — `scripts/07_topic_coherence.py` sweeps
   k ∈ {3,4,5,6,7,8,10} and UMass coherence favours k=4 (−2.8106) over k=6 (−3.1112).

   k=6 is retained deliberately, and the report must present this honestly rather than quoting
   only the retained value:
   - at k=4, two of four topics are near-duplicate generic-praise themes, so the better score
     buys no extra insight;
   - UMass coherence is known to favour fewer, broader topics, since it rewards co-occurrence
     among high-frequency terms — and in an 800-category corpus the generic terms co-occur with
     almost everything;
   - at k=6 the delivery/installation theme separates out, and it carries the project's headline
     finding (46.4% negative, the highest of any topic).

   The script comment previously claimed k=6 came from inspecting coherence when none had been
   computed. That claim is now replaced with the actual table and the argument above.
6. **argmax discards LDA's soft mixture**; vocabulary capped at 2,000 words.
7. Word-pair co-occurrence is weaker than the topic story implies ("delivery" co-occurs with
   "installation" in 16.9% of delivery reviews but with "bad" in only 2.6%). LDA groups by
   corpus-level patterns, not simple pairs. Don't overclaim.
8. [01_preprocessing.py](01_preprocessing.py) step 7 comment says it filters short reviews; it
   only computes `word_count`. Filtering happens in steps 2 and 3.

## 8. Dependencies

```
pip install pandas numpy scikit-learn matplotlib seaborn wordcloud joblib
```
`gensim` is **not** installed and is awkward on Python 3.13 — if LDA coherence is wanted,
implement UMass coherence directly on the sklearn output rather than installing gensim.

## 8a. Demo walkthrough (deliverable 6)

`demo/index.html` — an 11-stage walkthrough built for the presentation. Each stage states what
happens at that step, why the choice was made, and the result, with the relevant figure.

- Stages: Problem · Data · Preprocess · Features · Sentiment · Topics · Quality · Validation ·
  Insights · Limits · Reproduce
- Navigate with the Next button, the numbered rail, or left/right arrow keys
- Opens by double-clicking the file; no server needed
- Also published at **https://claude.ai/artifact/H7EaRoup9NA3UxRiTb6Vaw** (private — share from
  the page's Share menu before anyone else can open it)
- `demo/figures/` holds copies of `outputs/figures/`; refresh with
  `cp outputs/figures/*.png demo/figures/` after regenerating figures

The walkthrough deliberately includes the limitations and the duplicate-count correction, because
the course guidelines reward presenting limitations and because the correction is a stronger
demonstration of analytical decision-making than a clean result with no history.

## 9. Next steps

1. **Get the guidelines `.pptx` into the folder and re-verify section 2.**
2. Write the report: 6-section structure, dataset documentation, ethics note (public Kaggle
   dataset, no PII), model justification, the section 6 correction story, the section 7
   limitations, and the innovation argument (integrated sentiment + topic + anomaly analysis).
3. Add `requirements.txt`, README with run order, `git init` + `.gitignore` excluding the
   ~230 MB of derived CSVs.
4. Optional: LDA coherence to justify k; Logistic Regression as a third classifier (would also
   help the neutral class); a 100–200 review manually labelled sample to give the
   low-information flag real precision/recall.
5. Assemble a notebook, run top to bottom, confirm it reproduces these numbers.
6. Demo prep: be able to explain TF-IDF vs counts, why LDA is unsupervised, how topics are
   interpreted, what silhouette means, why weighted F1, and why K=4 was chosen over K=5.

## 10. Working preferences

Wants step-by-step progress, one stage at a time, with plain-language explanations of *why* each
step exists. Dislikes redundancy (don't hand over both a script and its output) and being given
things that were not actually delivered as files. Prefers concrete examples and honesty about
uncertainty.
