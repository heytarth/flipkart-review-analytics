# E-Commerce Review Analytics — Flipkart Product Reviews

CBS3009 (Social, Text and Media Analytics) course project — VIT Vellore.

An integrated text-analytics pipeline over 132,410 Flipkart product reviews, combining three
analytical components: **sentiment classification**, **topic discovery**, and **unsupervised
detection of low-information reviews**. The third component is validated against 200 blind
labelled reviews rather than reported on internal metrics alone.

**Label provenance:** those 200 labels were produced by an LLM (Claude) applying a written
criterion, not by a human annotator. See "Validation and its limits" below before quoting any
figure from it.

---

## Results at a glance

**Sentiment classification** — four classifiers compared on a stratified 80/20 split
(97,623 train / 24,406 test):

| Model | Accuracy | Weighted F1 | Macro F1 | Neutral F1 |
|---|---|---|---|---|
| Naive Bayes | 0.8474 | **0.8222** | 0.6100 | 0.1758 |
| Decision Tree | 0.7249 | 0.7470 | 0.5732 | 0.2571 |
| Logistic Regression | 0.7745 | 0.8002 | **0.6394** | **0.3137** |
| Complement NB | 0.8142 | 0.8163 | 0.6374 | 0.2969 |

The two criteria disagree: Naive Bayes wins on weighted F1, Logistic Regression on macro F1 and
neutral F1. Weighted F1 is dominated by the 73%-majority positive class.

**Topic discovery** — LDA over 84,070 reviews with ≥4 words, 6 topics. Topics are labelled by
*distinctive* terms (how much more probable a term is in one topic than across topics on average)
rather than merely the most frequent ones, which are dominated by corpus-wide generics.

The complaints topic — distinctive terms `bad, don, installation, worst, waste, expected` —
carries by far the highest negative rate (46.4%, against 4.1% for the gratitude topic),
indicating that service-side issues rather than product-quality language drive negativity. The
finding survives replacing the hard `argmax` topic assignment with LDA's full soft mixture (same
topic, 35.9%), so it is not an artefact of that simplification.

A UMass coherence sweep over k ∈ {3…10} favours k=4 (−2.81) over k=6 (−3.11). k=6 is retained
because k=4 produces two near-duplicate generic topics, and UMass coherence is known to favour
fewer, broader topics. The trade-off is reported rather than hidden.

**Low-information review detection** — K-Means on behavioural features, K=4 selected by
silhouette among non-degenerate solutions (0.5315). The flagged cluster holds **14.8% of
reviews, averaging 1.8 words and repeated an average of 2,349 times** across the catalogue.

Validated against 200 blind labelled reviews (LLM-generated labels, see below):

| Metric | Value |
|---|---|
| Precision | **0.980** |
| Recall | 0.737 |
| F1 | 0.841 |

The flag is highly precise but conservative — it identifies a high-confidence subset of
low-information reviews, not all of them. **It does not detect fake reviews**; no ground truth
for authenticity exists in this dataset.

---

## Setup

```bash
python -m pip install -r requirements.txt
```

Verified on Python 3.13.6 (Windows).

## Getting the dataset

The raw data is not in this repository (~30 MB). Download it from Kaggle:

**["Flipkart Products Review Dataset" by Mansi Thummar](https://www.kaggle.com/datasets/mansithummar67/flipkart-product-review-dataset)**

Place `flipkart_product.csv` at `data/raw/flipkart_product.csv`.

The file must be read with `encoding='latin1'` — it contains non-UTF8 bytes from the original
scrape. The scripts handle this.

## Running the pipeline

Run from the project root, in order:

```bash
python scripts/01_preprocessing.py            # raw -> cleaned (132,410 rows)
python scripts/02_sentiment_classification.py # Component 1: 4 classifiers + error analysis
python scripts/03_topic_modeling.py           # Component 2: LDA, 6 topics + soft-mixture check
python scripts/04_clustering_anomaly.py       # Component 3: K-Means, K=4
```

Then, for the validation of Component 3:

```bash
python scripts/05_export_validation_sample.py # exports 200 blind reviews
#   ... label validation/validation_sample.csv by hand, following
#   ... validation/LABELLING_INSTRUCTIONS.md, save as
#   ... validation/validation_sample_labeled.csv
python scripts/06_score_validation.py         # precision / recall / F1 + error analysis
```

Supporting evidence and figures:

```bash
python scripts/07_topic_coherence.py          # UMass coherence sweep over k (slow)
python scripts/08_pipeline_diagram.py         # architecture figure
```

**Do not re-run `05` after labelling.** It rewrites the sample and key that the hand labels are
matched against by `row_id`. Re-run it only if the clustering itself changes — and then the 200
reviews must be re-labelled.

### Notebook

```bash
python scripts/build_notebook.py              # generates the .ipynb (syntax-checks every cell)
python scripts/run_notebook.py                # executes it in place, ~5 min
```

The notebook is generated from `build_notebook.py` rather than hand-edited, so it cannot drift
away from the scripts. `run_notebook.py` uses the `nbclient` API rather than the `jupyter
nbconvert` CLI, because the Jupyter console scripts are not on PATH in this environment and the
CLI reports "command not found" while still exiting 0.

### Demo walkthrough

`demo/index.html` is the working demonstration (deliverable 6): a stage-by-stage walkthrough of
the pipeline, from problem definition through to validated results and limitations. Each stage
states what happens, why that choice was made, and what the result was.

Open it by double-clicking the file — it needs no server and no internet beyond web fonts. Use
the **Next stage** button, the numbered rail, or the left/right arrow keys. It is also published
at the link in `docs/PROJECT_HANDOFF.md`.

`demo/figures/` holds copies of `outputs/figures/` so the demo stays self-contained. Refresh them
with `cp outputs/figures/*.png demo/figures/` after regenerating the figures.

### Regenerating the results log

```bash
python scripts/regenerate_results_log.py      # reruns the pipeline, rewrites docs/results_log.md
```

`docs/results_log.md` is the single source of truth for every number quoted in the report and the
demo. It is generated by script rather than by hand so the report cannot cite a stale figure.

### Verifying the submission

```bash
python scripts/verify_submission.py
```

Checks that every expected file exists, every declared dependency imports, the notebook executed
with no unexecuted cells and no errors, the numbers in the notebook and the demo agree with the
output tables, `results_log.md` is not older than the scripts it reports on, and every figure
exists and is referenced correctly. Exits non-zero on any failure, so it can gate a submission.

Scripts resolve paths from their own file location, so the working directory does not matter.
All use `random_state=42` and reproduce exactly — verified by running the full pipeline from an
unrelated directory and from the notebook, with identical results.

## Layout

```
data/raw/          source dataset (download separately)
data/processed/    generated — cleaned data, cluster and topic assignments
scripts/           01-08 pipeline, plus notebook and log tooling
notebook/          Flipkart_Review_Analytics.ipynb — executed, 20 cells, reproduces everything
outputs/figures/   7 PNG figures, all titled with labelled axes
outputs/tables/    8 result tables as CSV
outputs/models/    4 trained classifiers and the TF-IDF vectorizer
validation/        manual-labelling sample, labels, key, and the written criterion
demo/              index.html — the stage-by-stage walkthrough, plus its figure copies
archive/           superseded results, kept as before/after evidence
docs/              PROJECT_HANDOFF.md (full context), results_log.md (verbatim run output)
report/            (report deliverable)
```

`docs/results_log.md` holds the verbatim terminal output of a clean full run. **Every number
quoted in the report comes from there.**

## Data and ethics

The dataset is publicly released on Kaggle for research and education. Only review text, star
ratings, product names and prices are used — no personal identifiers. No data was collected from
Flipkart directly, so platform scraping terms do not apply to this work.

Live scraping was attempted and deliberately abandoned: Flipkart's API gateway returns HTTP 403
to non-browser clients, and working around that would have required a headful browser automation
stack. Using an approved public dataset was the lower-risk choice and is permitted by the course
guidelines.

## Validation and its limits

Component 3's flag was scored against 200 blind labels. **Those labels were produced by an LLM
(Claude) applying the written criterion in `validation/LABELLING_INSTRUCTIONS.md`. They are not
human annotation.**

The comparison is still meaningful, because the two systems use different signals: K-Means never
reads the words (it uses length, repetition frequency and star rating), while the labeller read
only the text. Agreement between them is therefore not circular.

But it is weaker than human annotation in a specific way: an LLM's notion of "low-information"
plausibly tracks brevity, and brevity is also most of what the clustering keys on. A shared bias
of that kind would inflate agreement. The 0.980 precision should be read with that in mind.

**Correct phrasing:** "validated against LLM-generated reference labels applying a pre-registered
criterion". **Incorrect:** "hand-labelled", "human judgement", "human validation".

## Known limitations

- `sentiment_label` is derived from the star rating, not from human annotation, so the
  classifiers predict ratings from text rather than human-judged sentiment.
- The neutral class is weak across all four models (best F1 0.314). Three-star reviews genuinely
  mix positive and negative language.
- Topic labels are human interpretations of top-word lists, not model output.
- **The validation labels are LLM-generated, not human annotation.** This is not human
  validation and must not be described as such.
- Only one labeller (the LLM), so inter-rater agreement (Cohen's kappa) is unmeasured.
- The labelling criterion was refined during labelling ("naming the product category counts as
  informative"); the refinement is disclosed in `validation/LABELLING_INSTRUCTIONS.md`.
- Deduplication is case-sensitive while the repetition count lowercases, so "Good" and "good"
  survive as separate rows. Defensible, but an internal inconsistency worth stating.
- The cluster-vs-sentiment crosstab is partly circular: `Rate` is both a clustering input and the
  source of `sentiment_label`.

See `docs/PROJECT_HANDOFF.md` for the full list and the methodology corrections made during
development.
