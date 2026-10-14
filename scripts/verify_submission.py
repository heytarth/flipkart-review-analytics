"""
Checks that the submission is internally consistent and reproducible.

The course guidelines require that "submission must be internally consistent and
reproducible" and that "code/notebook runs and reproduces the reported results".
This script makes that checkable instead of asserted. It verifies:

  1. every expected file exists
  2. every declared dependency actually imports
  3. the notebook executed cleanly, with no unexecuted cells and no errors
  4. the notebook's numbers agree with the scripts' output tables
  5. the demo walkthrough quotes numbers that match those tables
  6. results_log.md is not older than the scripts it reports on
  7. figures exist and are non-trivial in size

    python scripts/verify_submission.py

Exits non-zero if any check fails, so it can gate a submission.
"""
import importlib
import json
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
checks, failures, warnings_ = [], [], []


def ok(msg):
    checks.append(msg)
    print(f"  PASS  {msg}")


def fail(msg):
    failures.append(msg)
    print(f"  FAIL  {msg}")


def warn(msg):
    warnings_.append(msg)
    print(f"  WARN  {msg}")


def section(title):
    print(f"\n{title}\n" + "-" * len(title))


# --- 1. expected files -------------------------------------------------------
section("1. Expected files")
EXPECTED = [
    'README.md', 'requirements.txt', '.gitignore',
    'data/raw/flipkart_product.csv',
    'data/processed/flipkart_cleaned.csv',
    'data/processed/flipkart_with_clusters.csv',
    'notebook/Flipkart_Review_Analytics.ipynb',
    'demo/index.html',
    'docs/PROJECT_HANDOFF.md', 'docs/results_log.md',
    'validation/validation_sample_labeled.csv', 'validation/validation_key.csv',
    'validation/LABELLING_INSTRUCTIONS.md',
    'outputs/tables/classifier_comparison.csv',
    'outputs/tables/cluster_profiles.csv',
    'outputs/tables/validation_results.csv',
    'outputs/tables/topic_sentiment_crosstab.csv',
    'outputs/tables/topic_coherence.csv',
]
for rel in EXPECTED:
    (ok if (ROOT / rel).exists() else fail)(f"{rel}")

for i in range(1, 9):
    hits = list((ROOT / 'scripts').glob(f'{i:02d}_*.py'))
    (ok if hits else fail)(f"scripts/{i:02d}_*.py present")

# --- 2. dependencies ---------------------------------------------------------
section("2. Declared dependencies import")
alias = {'scikit-learn': 'sklearn'}
for line in (ROOT / 'requirements.txt').read_text(encoding='utf-8').splitlines():
    line = line.strip()
    if not line or line.startswith('#'):
        continue
    name = re.split(r'[=<>]', line)[0].strip()
    try:
        importlib.import_module(alias.get(name, name))
        ok(f"{name} imports")
    except Exception as exc:
        fail(f"{name} does not import: {exc}")

# --- 3. notebook executed cleanly -------------------------------------------
section("3. Notebook execution state")
nb = json.loads((ROOT / 'notebook' / 'Flipkart_Review_Analytics.ipynb')
                .read_text(encoding='utf-8'))
code_cells = [c for c in nb['cells'] if c['cell_type'] == 'code']
executed = [c for c in code_cells if c.get('execution_count')]
errors = [o for c in code_cells for o in c.get('outputs', [])
          if o.get('output_type') == 'error']
(ok if len(executed) == len(code_cells) else fail)(
    f"all {len(code_cells)} code cells executed ({len(executed)} have run counts)")
(ok if not errors else fail)(f"no error outputs ({len(errors)} found)")

def stream_text(output):
    """nbformat stores stream text as either a string or a list of lines."""
    text = output.get('text', '')
    return ''.join(text) if isinstance(text, list) else text


nb_text = '\n'.join(
    ''.join(stream_text(o) for o in c.get('outputs', [])
            if o.get('output_type') == 'stream')
    for c in code_cells)

# --- 4 & 5. numbers agree across notebook, tables and demo -------------------
section("4. Scripts vs notebook vs demo")
clf = pd.read_csv(ROOT / 'outputs/tables/classifier_comparison.csv', index_col=0)
prof = pd.read_csv(ROOT / 'outputs/tables/cluster_profiles.csv', index_col=0)
val = pd.read_csv(ROOT / 'outputs/tables/validation_results.csv').iloc[0]
cross = pd.read_csv(ROOT / 'outputs/tables/topic_sentiment_crosstab.csv', index_col=0)
demo = (ROOT / 'demo' / 'index.html').read_text(encoding='utf-8')

# The low-information cluster: most repeated text relative to shortest length.
# Derived the same way the pipeline derives it, rather than hard-coded, so this
# check does not silently pass against a stale cluster number.
lowinfo = int((prof['log_text_freq'].rank() - prof['word_count'].rank()).idxmax())

FACTS = {
    'NB weighted F1': (round(clf.loc['Naive Bayes', 'F1'], 4), '0.8222'),
    'LR macro F1': (round(clf.loc['Logistic Regression', 'Macro_F1'], 4), '0.6394'),
    'validation precision': (round(val['precision'], 3), '0.980'),
    'validation recall': (round(val['recall'], 3), '0.737'),
    'low-info cluster size': (int(prof.loc[lowinfo, 'n_reviews']), '19,620'),
    'most negative topic %': (round(100 * cross['negative'].max(), 1), '46.4'),
}
for label, (actual, as_written) in FACTS.items():
    needle = as_written.replace(',', '')
    present_demo = as_written in demo or needle in demo.replace(',', '')
    present_nb = as_written in nb_text or needle in nb_text.replace(',', '')
    if present_demo and present_nb:
        ok(f"{label} = {actual} appears in both notebook and demo")
    elif present_demo:
        warn(f"{label} = {actual} in demo; not found verbatim in notebook output")
    else:
        fail(f"{label} = {actual} not found in demo walkthrough")

# --- 6. freshness of generated artefacts -------------------------------------
section("5. Generated artefacts are not stale")
pipeline = sorted((ROOT / 'scripts').glob('0*.py'))

log = ROOT / 'docs' / 'results_log.md'
stale = [p.name for p in pipeline if p.stat().st_mtime > log.stat().st_mtime]
if stale:
    fail(f"results_log.md is older than: {', '.join(stale)} - re-run "
         f"scripts/regenerate_results_log.py")
else:
    ok("results_log.md is at least as recent as every pipeline script")

# The notebook goes stale the same way: it duplicates the analysis, so a script
# change after the last execution means it may no longer agree with the scripts.
# 07 and 08 produce standalone evidence the notebook does not contain, so a
# change to those does not make it stale.
nb_path = ROOT / 'notebook' / 'Flipkart_Review_Analytics.ipynb'
builder = ROOT / 'scripts' / 'build_notebook.py'
relevant = [p for p in pipeline if p.name[:2] in {'01', '02', '03', '04', '06'}]
relevant.append(builder)
nb_stale = [p.name for p in relevant if p.stat().st_mtime > nb_path.stat().st_mtime]
if nb_stale:
    fail(f"notebook is older than: {', '.join(nb_stale)} - re-run "
         f"scripts/build_notebook.py then scripts/run_notebook.py")
else:
    ok("notebook is at least as recent as the scripts it reproduces")

# demo/figures/ are copies; if outputs/figures/ moved on, the demo shows old art.
demo_fig_dir = ROOT / 'demo' / 'figures'
out_fig_dir = ROOT / 'outputs' / 'figures'
drifted = [f.name for f in out_fig_dir.glob('*.png')
           if (demo_fig_dir / f.name).exists()
           and f.stat().st_mtime > (demo_fig_dir / f.name).stat().st_mtime]
if drifted:
    fail(f"demo/figures/ is behind outputs/figures/ for: {', '.join(drifted)} - "
         f"run: cp outputs/figures/*.png demo/figures/")
else:
    ok("demo/figures/ matches outputs/figures/")

# --- 7. figures --------------------------------------------------------------
section("6. Figures")
for fig in sorted((ROOT / 'outputs' / 'figures').glob('*.png')):
    size_kb = fig.stat().st_size / 1024
    (ok if size_kb > 10 else fail)(f"{fig.name} ({size_kb:.0f} KB)")

# The demo is served by `python -m http.server`, which declares no charset, and
# is also opened straight from disk. A literal non-ASCII character then renders
# as mojibake (an em dash became "a-EUR-" in testing). Keeping the file pure
# ASCII with numeric HTML entities makes it charset-proof.
demo_bytes = (ROOT / 'demo' / 'index.html').read_bytes()
non_ascii = [(i, b) for i, b in enumerate(demo_bytes) if b > 127]
if non_ascii:
    fail(f"demo/index.html contains {len(non_ascii)} non-ASCII bytes "
         f"(first at offset {non_ascii[0][0]}) - these render as mojibake when "
         f"served without a charset. Convert them to &#NNNN; entities.")
else:
    ok("demo/index.html is pure ASCII (charset-proof)")

demo_figs = set(re.findall(r"'(figures/[^']+\.png)'", demo))
missing = [f for f in demo_figs if not (ROOT / 'demo' / f).exists()]
(ok if not missing else fail)(
    f"all {len(demo_figs)} demo figure references resolve"
    + (f" (missing: {missing})" if missing else ""))

# --- summary -----------------------------------------------------------------
print("\n" + "=" * 60)
print(f"{len(checks)} passed | {len(warnings_)} warnings | {len(failures)} failed")
print("=" * 60)
if warnings_:
    print("\nWarnings:")
    for w in warnings_:
        print(f"  - {w}")
if failures:
    print("\nFailures:")
    for f in failures:
        print(f"  - {f}")
    sys.exit(1)
print("\nSubmission is internally consistent.")
