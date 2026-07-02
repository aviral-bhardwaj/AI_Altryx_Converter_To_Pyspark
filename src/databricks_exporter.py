"""
Databricks Exporter
====================
Turns generated Databricks source-format code into deployable artifacts:

- ``.ipynb``  — Jupyter notebook (via ``nbformat``) importable into any
  Databricks workspace or Git folder; ``%md`` documentation preserved as
  markdown cells and the validation report injected as the first cell.
- ``.py``     — Databricks "source format" notebook (round-trips with Repos).
- ``.dbc``    — Databricks archive (zip of NotebookV1 JSON) for legacy
  workspace imports.

Also provides ``batch_export``: point it at a folder of ``.yxmd`` files and it
runs the full self-correcting pipeline per workflow, producing a mirrored
Databricks folder structure of converted notebooks plus a batch summary.
"""

import io
import json
import logging
import re
import zipfile
from pathlib import Path
from typing import Optional

from .parser import AlteryxWorkflowParser
from .self_correction import SelfCorrectingConverter, SelfCorrectionResult

logger = logging.getLogger(__name__)

DATABRICKS_CELL_SEPARATOR = "# COMMAND ----------"


def safe_name(name: str) -> str:
    """Filesystem/notebook-safe version of a workflow name."""
    cleaned = name.lower().replace(" ", "_").replace("-", "_")
    return "".join(c for c in cleaned if c.isalnum() or c == "_") or "workflow"


def source_to_cells(code: str) -> list:
    """
    Split Databricks source-format code into (cell_type, source) tuples,
    where cell_type is "markdown" or "code".
    """
    body = code.replace("# Databricks notebook source", "", 1)
    cells = []
    for raw_cell in body.split(DATABRICKS_CELL_SEPARATOR):
        lines = [l for l in raw_cell.splitlines()]
        # Trim leading/trailing blank lines
        while lines and not lines[0].strip():
            lines.pop(0)
        while lines and not lines[-1].strip():
            lines.pop()
        if not lines:
            continue
        if all(l.strip().startswith("# MAGIC") or not l.strip() for l in lines):
            md_lines = []
            for l in lines:
                text = l.strip()
                text = re.sub(r"^# MAGIC ?", "", text)
                if text == "%md":
                    continue
                md_lines.append(text)
            cells.append(("markdown", "\n".join(md_lines)))
        else:
            cells.append(("code", "\n".join(lines)))
    return cells


def to_ipynb(code: str, report_markdown: Optional[str] = None) -> dict:
    """
    Build a Jupyter notebook dict from Databricks source-format code.
    The validation report (if given) becomes the FIRST cell of the notebook.
    Uses nbformat when available; the fallback emits the same v4 JSON schema.
    """
    cell_specs = source_to_cells(code)
    if report_markdown:
        cell_specs.insert(0, ("markdown", report_markdown))

    try:
        import nbformat

        nb = nbformat.v4.new_notebook()
        nb.metadata["language_info"] = {"name": "python"}
        nb.metadata["application/vnd.databricks.v1+notebook"] = {
            "language": "python",
            "notebookMetadata": {"pythonIndentUnit": 4},
        }
        for cell_type, source in cell_specs:
            if cell_type == "markdown":
                nb.cells.append(nbformat.v4.new_markdown_cell(source))
            else:
                nb.cells.append(nbformat.v4.new_code_cell(source))
        return json.loads(nbformat.writes(nb))
    except ImportError:
        logger.warning("nbformat not installed; using built-in .ipynb writer")
        cells = []
        for cell_type, source in cell_specs:
            cell = {
                "cell_type": cell_type,
                "metadata": {},
                "source": source.splitlines(keepends=True),
            }
            if cell_type == "code":
                cell["execution_count"] = None
                cell["outputs"] = []
            cells.append(cell)
        return {
            "cells": cells,
            "metadata": {"language_info": {"name": "python"}},
            "nbformat": 4,
            "nbformat_minor": 5,
        }


def to_dbc_bytes(code: str, notebook_name: str) -> bytes:
    """
    Build a minimal Databricks archive (.dbc): a zip containing a NotebookV1
    JSON document, importable via Workspace → Import.
    """
    commands = []
    position = 1.0
    for cell_type, source in source_to_cells(code):
        command_text = f"%md\n{source}" if cell_type == "markdown" else source
        commands.append({
            "version": "CommandV1",
            "position": position,
            "command": command_text,
            "commandTitle": "",
            "state": "finished",
        })
        position += 1.0

    notebook_json = {
        "version": "NotebookV1",
        "name": notebook_name,
        "language": "python",
        "commands": commands,
    }
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(f"{notebook_name}.python", json.dumps(notebook_json))
    return buffer.getvalue()


class DatabricksExporter:
    """Writes conversion results to disk in one or more Databricks formats."""

    SUPPORTED_FORMATS = ("ipynb", "py", "dbc")

    def __init__(self, output_dir: str = "./output"):
        self.output_dir = Path(output_dir)

    def export(
        self,
        result: SelfCorrectionResult,
        formats: tuple = ("ipynb", "py"),
        subdir: str = "",
    ) -> dict:
        """
        Export a self-corrected conversion. The validation report is embedded
        as the top %md cell in every format. Returns {format: written path}.
        """
        name = safe_name(result.conversion.workflow_name)
        target_dir = self.output_dir / subdir if subdir else self.output_dir
        target_dir.mkdir(parents=True, exist_ok=True)

        report_md = result.report_markdown()
        written = {}

        for fmt in formats:
            if fmt not in self.SUPPORTED_FORMATS:
                raise ValueError(f"Unsupported export format: {fmt!r} "
                                 f"(choose from {self.SUPPORTED_FORMATS})")
            path = target_dir / f"{name}.{fmt}"
            if fmt == "ipynb":
                path.write_text(json.dumps(to_ipynb(result.code, report_md), indent=1),
                                encoding="utf-8")
            elif fmt == "py":
                report_cell = "\n".join(
                    ["# COMMAND ----------", "", "# MAGIC %md"]
                    + [f"# MAGIC {line}" for line in report_md.splitlines()]
                    + ["", "# COMMAND ----------", ""]
                )
                code = result.code.replace(
                    "# Databricks notebook source\n",
                    f"# Databricks notebook source\n\n{report_cell}\n", 1)
                path.write_text(code, encoding="utf-8")
            elif fmt == "dbc":
                code_with_report = result.code + ""
                path.write_bytes(to_dbc_bytes(code_with_report, name))
            written[fmt] = str(path)
            logger.info("Exported %s -> %s", fmt, path)
        return written


def batch_export(
    input_dir: str,
    output_dir: str = "./output",
    formats: tuple = ("ipynb", "py"),
    converter: Optional[SelfCorrectingConverter] = None,
    spark=None,
    progress_callback=None,
) -> list:
    """
    Batch mode: convert every ``.yxmd`` in ``input_dir`` (recursively) through
    the full self-correcting pipeline, mirroring the folder structure under
    ``output_dir``. Returns a list of per-workflow summary dicts.
    """
    converter = converter or SelfCorrectingConverter()
    exporter = DatabricksExporter(output_dir)
    input_path = Path(input_dir)
    summaries = []

    yxmd_files = sorted(input_path.rglob("*.yxmd"))
    if not yxmd_files:
        logger.warning("No .yxmd files found under %s", input_dir)

    for yxmd in yxmd_files:
        rel_dir = str(yxmd.parent.relative_to(input_path))
        subdir = "" if rel_dir == "." else rel_dir
        entry = {"workflow": yxmd.name, "status": "", "files": {}, "iterations": 0}
        try:
            workflow = AlteryxWorkflowParser(str(yxmd)).parse()
            result = converter.convert(
                workflow, yxmd.stem, spark=spark, progress_callback=progress_callback)
            entry["status"] = result.status
            entry["iterations"] = len(result.iterations)
            entry["files"] = exporter.export(result, formats=formats, subdir=subdir)
        except Exception as exc:
            logger.exception("Batch conversion failed for %s", yxmd)
            entry["status"] = f"ERROR: {exc}"
        summaries.append(entry)

    # Write a machine-readable batch summary next to the notebooks.
    summary_path = Path(output_dir) / "batch_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    return summaries
