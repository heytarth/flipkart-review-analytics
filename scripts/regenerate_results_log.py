"""
Regenerates docs/results_log.md by running the pipeline and capturing output.

results_log.md is the single source of truth for every number quoted in the
report and slides. If a script changes and the log is not regenerated, the
report silently cites stale figures -- so this is scripted rather than done by
hand, and it fails loudly if any stage errors instead of writing a partial log.

    python scripts/regenerate_results_log.py

NOTE: 05_export_validation_sample.py is deliberately NOT re-run. It would
rewrite validation_sample.csv and validation_key.csv, and the labelled
file is matched to the existing key by row_id. Regenerating the sample is only
correct if the clustering itself changes, and then the 200 reviews must be
re-labelled.
"""
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOG = ROOT / 'docs' / 'results_log.md'
LOG.parent.mkdir(parents=True, exist_ok=True)

STAGES = [
    ('01_preprocessing.py', 'Stage 1 — Preprocessing'),
    ('02_sentiment_classification.py', 'Stage 2 — Component 1: Sentiment classification'),
    ('03_topic_modeling.py', 'Stage 3 — Component 2: Topic discovery'),
    ('04_clustering_anomaly.py', 'Stage 4 — Component 3: Low-information detection'),
    ('06_score_validation.py', 'Stage 5 — Validation against reference labels (LLM-generated)'),
    ('07_topic_coherence.py', 'Stage 6 — Topic coherence sweep'),
    ('08_pipeline_diagram.py', 'Stage 7 — Pipeline architecture diagram'),
]

parts = []
failures = []
started = time.time()

for script, title in STAGES:
    path = ROOT / 'scripts' / script
    if not path.exists():
        failures.append((script, f'not found: {path}'))
        continue
    print(f'Running {script} ...', flush=True)
    t0 = time.time()
    proc = subprocess.run([sys.executable, str(path)], cwd=str(ROOT),
                          capture_output=True, text=True)
    elapsed = time.time() - t0
    output = proc.stdout + (('\n[stderr]\n' + proc.stderr) if proc.stderr.strip() else '')
    if proc.returncode != 0:
        failures.append((script, f'exit code {proc.returncode}'))
        print(f'  FAILED (exit {proc.returncode}) after {elapsed:.0f}s', flush=True)
    else:
        print(f'  ok ({elapsed:.0f}s)', flush=True)
    parts.append(f'## {title}\n\n`scripts/{script}` — {elapsed:.0f}s\n\n'
                 f'```\n{output.strip()}\n```\n')

header = f"""# Results Log

Verbatim terminal output of a full pipeline run. **Every number quoted in the report and
slides must come from this file.** Regenerate it after any script change:

```
python scripts/regenerate_results_log.py
```

- Generated: {datetime.now():%Y-%m-%d %H:%M}
- Python 3.13.6 | pandas 3.0.1 | scikit-learn 1.8.0
- All scripts use `random_state=42` and reproduce exactly.
- `05_export_validation_sample.py` is not re-run here; it would rewrite the sample that the
  existing hand labels are matched against. See the note in this script.

---
"""

LOG.write_text(header + '\n' + '\n'.join(parts), encoding='utf-8')
total = time.time() - started

print(f'\nWrote {LOG.relative_to(ROOT)} ({len(parts)} stages, {total:.0f}s total)')
if failures:
    for script, why in failures:
        print(f'  FAILED: {script} -- {why}')
    sys.exit(f'{len(failures)} stage(s) failed; the log records the failure output.')
print('All stages completed successfully.')
