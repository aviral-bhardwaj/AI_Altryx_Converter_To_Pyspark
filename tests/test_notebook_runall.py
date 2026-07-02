"""
"Run All" test for the Skill Mode notebook: executes every code cell in order
(as a Databricks/Jupyter user would) with default settings and asserts the
converted notebook + summary land in the output directory.

Needs no Spark — the default configuration converts the bundled sample and
skips sample-data execution.
"""

import json
from pathlib import Path

import nbformat
import pytest

NOTEBOOK = Path(__file__).parent.parent / "notebooks" / "01_Skill_Mode_Alteryx_to_Databricks.ipynb"


def test_notebook_is_valid_nbformat():
    nb = nbformat.read(str(NOTEBOOK), as_version=4)
    nbformat.validate(nb)
    code_cells = [c for c in nb.cells if c.cell_type == "code"]
    md_cells = [c for c in nb.cells if c.cell_type == "markdown"]
    assert len(code_cells) >= 9, "expected one code cell per UX section"
    assert md_cells and "Skill Mode" in md_cells[0].source


def test_notebook_run_all(tmp_path, monkeypatch):
    monkeypatch.setenv("SKILL_OUTPUT_DIR", str(tmp_path))
    monkeypatch.delenv("DATABRICKS_RUNTIME_VERSION", raising=False)

    nb = json.loads(NOTEBOOK.read_text())
    namespace = {"__name__": "__main__"}
    executed = 0
    for index, cell in enumerate(nb["cells"]):
        if cell["cell_type"] != "code":
            continue
        source = cell["source"] if isinstance(cell["source"], str) else "".join(cell["source"])
        # %pip / %md magics are Databricks/IPython concerns, not plain Python.
        source = "\n".join(
            line for line in source.splitlines() if not line.strip().startswith("%"))
        if not source.strip():
            continue
        exec(compile(source, f"<cell {index}>", "exec"), namespace)
        executed += 1

    assert executed >= 9

    # Default widgets convert tests/sample_workflows/02_join_summarize.yxmd.
    exported_nb = tmp_path / "02_join_summarize.ipynb"
    exported_py = tmp_path / "02_join_summarize.py"
    summary_file = tmp_path / "conversion_summary.json"
    assert exported_nb.exists() and exported_py.exists() and summary_file.exists()

    summary = json.loads(summary_file.read_text())
    assert summary["02_join_summarize"]["status"] in ("PASS", "WARNING")
    assert summary["02_join_summarize"]["unsupported"] == []

    # The exported notebook is valid and leads with the validation report.
    exported = nbformat.read(str(exported_nb), as_version=4)
    nbformat.validate(exported)
    assert exported.cells[0].cell_type == "markdown"
    assert "Conversion Validation Report" in exported.cells[0].source
    assert exported.cells[0].source.startswith("## ✅") or "PASS" in exported.cells[0].source
