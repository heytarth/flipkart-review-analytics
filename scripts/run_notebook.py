"""
Executes notebook/Flipkart_Review_Analytics.ipynb in place and saves the outputs.

Uses the nbclient Python API rather than the `jupyter nbconvert` CLI, because
the Jupyter console scripts are not on PATH in this environment -- the CLI
reports "command not found" and still exits 0, which silently leaves the
notebook unexecuted.

The kernel's working directory is set to the project root so the notebook's
own path detection resolves correctly.

    python scripts/run_notebook.py
"""
import sys
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError

ROOT = Path(__file__).resolve().parent.parent
NB_PATH = ROOT / 'notebook' / 'Flipkart_Review_Analytics.ipynb'

if not NB_PATH.exists():
    sys.exit(f"Notebook not found: {NB_PATH}\nRun scripts/build_notebook.py first.")

nb = nbformat.read(NB_PATH, as_version=4)
n_code = sum(c.cell_type == 'code' for c in nb.cells)
print(f"Executing {NB_PATH.relative_to(ROOT)} ({n_code} code cells)...")
print("This reruns the full analysis from the raw CSV and takes several minutes.")

client = NotebookClient(
    nb,
    timeout=2400,
    kernel_name='python3',
    allow_errors=False,
    resources={'metadata': {'path': str(ROOT)}},   # kernel cwd = project root
)

start = time.time()
try:
    client.execute()
except CellExecutionError as exc:
    nbformat.write(nb, NB_PATH)          # keep partial output for debugging
    print(f"\nFAILED after {time.time() - start:.0f}s")
    print("The partially executed notebook was saved so the failing cell can be inspected.")
    sys.exit(f"\n{exc}")

nbformat.write(nb, NB_PATH)
elapsed = time.time() - start

executed = sum(1 for c in nb.cells if c.cell_type == 'code' and c.execution_count)
errors = [o for c in nb.cells if c.cell_type == 'code'
          for o in c.get('outputs', []) if o.output_type == 'error']

print(f"\nCompleted in {elapsed:.0f}s")
print(f"Code cells executed: {executed}/{n_code}")
print(f"Errors: {len(errors)}")
if executed != n_code or errors:
    sys.exit("Notebook did not execute cleanly -- do not trust its numbers.")
print("Notebook executed cleanly and saved with outputs.")
