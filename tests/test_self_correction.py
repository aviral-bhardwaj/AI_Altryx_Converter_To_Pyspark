"""
Tests for the converter engine, structural validator, self-correction loop,
and Databricks exporter — end-to-end over the sample workflows.
"""

import json
from pathlib import Path

import pytest

from src.converter_engine import ConverterEngine, load_tool_mapping
from src.databricks_exporter import (
    DatabricksExporter,
    batch_export,
    source_to_cells,
    to_dbc_bytes,
    to_ipynb,
)
from src.parser import AlteryxWorkflowParser
from src.self_correction import SelfCorrectingConverter
from src.validators import StructuralValidator, ValidationStatus, strip_databricks_magics

SAMPLES_DIR = Path(__file__).parent / "sample_workflows"
SAMPLES = [
    "01_simple_filter_select.yxmd",
    "02_join_summarize.yxmd",
    "03_formula_union_sort.yxmd",
]


@pytest.fixture(params=SAMPLES)
def sample_workflow_file(request):
    return SAMPLES_DIR / request.param


@pytest.fixture
def parsed(sample_workflow_file):
    return AlteryxWorkflowParser(str(sample_workflow_file)).parse()


# ── ConverterEngine ────────────────────────────────────────────────

class TestConverterEngine:
    def test_tool_mapping_loads(self):
        mapping = load_tool_mapping()
        assert mapping.get("version") == 1
        assert "Filter" in mapping["tools"]

    def test_convert_produces_valid_python(self, parsed, sample_workflow_file):
        result = ConverterEngine().convert(parsed, sample_workflow_file.stem)
        compile(strip_databricks_magics(result.code), "<generated>", "exec")

    def test_tool_statuses_cover_all_tools(self, parsed, sample_workflow_file):
        result = ConverterEngine().convert(parsed, sample_workflow_file.stem)
        assert len(result.tool_statuses) == len(parsed.all_tools)

    def test_unsupported_tool_is_flagged_not_fatal(self, parsed):
        # Simulate an unknown tool by mutating a parsed tool type.
        tool = next(iter(parsed.all_tools.values()))
        tool.tool_type = "SomeExoticMacroTool"
        result = ConverterEngine().convert(parsed, "exotic")
        assert any(s.status == "unsupported" for s in result.tool_statuses)
        assert "UNSUPPORTED TOOL" in result.code
        compile(strip_databricks_magics(result.code), "<generated>", "exec")

    def test_strict_mode_recovers_configs(self, parsed):
        for tool in parsed.all_tools.values():
            tool.parsed_config = {}
        result = ConverterEngine().convert(parsed, "strict", strict=True)
        recovered = [t for t in parsed.all_tools.values() if t.parsed_config]
        assert recovered, "strict mode should re-extract configs from raw XML"
        assert result.strict


# ── StructuralValidator ────────────────────────────────────────────

class TestStructuralValidator:
    def test_samples_validate_clean(self, parsed, sample_workflow_file):
        result = ConverterEngine().convert(parsed, sample_workflow_file.stem)
        report = StructuralValidator().validate(result.code, parsed)
        assert report.status in (ValidationStatus.PASS, ValidationStatus.WARNING)
        assert not report.failures

    def test_syntax_error_fails(self, parsed):
        report = StructuralValidator().validate("df = spark.table(", parsed)
        assert not report.syntax_ok
        assert report.status == ValidationStatus.FAIL

    def test_missing_tool_operation_fails(self, parsed):
        # Code with none of the expected operations for the workflow's tools.
        report = StructuralValidator().validate("x = 1", parsed)
        assert report.failures

    def test_unresolved_reference_fails(self, parsed):
        code = "df_ok = spark.table('a.b.c')\ndf_bad = df_missing.filter(F.lit(True))"
        report = StructuralValidator().validate(code, parsed)
        assert any("df_missing" in i.message for i in report.failures)

    def test_lambda_params_are_not_unresolved(self, parsed):
        code = (
            "from functools import reduce\n"
            "df_a = spark.table('x.y.z')\n"
            "df_b = reduce(lambda a, b: a.unionByName(b), [df_a, df_a])\n"
        )
        report = StructuralValidator().validate(code, parsed)
        assert not any("Unresolved" in i.message and "'a'" in i.message
                       for i in report.failures)


# ── SelfCorrectingConverter ────────────────────────────────────────

class TestSelfCorrection:
    def test_samples_converge(self, parsed, sample_workflow_file):
        result = SelfCorrectingConverter().convert(parsed, sample_workflow_file.stem)
        assert result.status in ("PASS", "WARNING")
        assert 1 <= len(result.iterations) <= 3

    def test_report_markdown_contains_iterations(self, parsed, sample_workflow_file):
        result = SelfCorrectingConverter().convert(parsed, sample_workflow_file.stem)
        md = result.report_markdown()
        assert "Conversion Validation Report" in md
        assert "| Iteration |" in md

    def test_max_iterations_respected(self, parsed):
        # Force perpetual failure by renaming every tool to an unknown type
        # with no structural expectation satisfiable.
        for tool in parsed.all_tools.values():
            tool.tool_type = "Join"
            tool.parsed_config = {"left_keys": ["NoSuchKey"], "right_keys": ["NoSuchKey"]}
        converter = SelfCorrectingConverter(max_iterations=3)
        result = converter.convert(parsed, "doomed")
        assert len(result.iterations) <= 3

    def test_progress_callback_invoked(self, parsed, sample_workflow_file):
        events = []
        SelfCorrectingConverter().convert(
            parsed, sample_workflow_file.stem,
            progress_callback=lambda i, s, m: events.append((i, s, m)))
        assert events


# ── DatabricksExporter ─────────────────────────────────────────────

class TestExporter:
    def test_source_to_cells_roundtrip(self, parsed, sample_workflow_file):
        result = ConverterEngine().convert(parsed, sample_workflow_file.stem)
        cells = source_to_cells(result.code)
        assert any(t == "markdown" for t, _ in cells)
        assert any(t == "code" for t, _ in cells)

    def test_ipynb_report_is_first_cell(self, parsed, sample_workflow_file):
        result = SelfCorrectingConverter().convert(parsed, sample_workflow_file.stem)
        nb = to_ipynb(result.code, result.report_markdown())
        assert nb["cells"][0]["cell_type"] == "markdown"
        assert "Conversion Validation Report" in "".join(nb["cells"][0]["source"])

    def test_dbc_is_valid_zip(self, parsed, sample_workflow_file):
        import io
        import zipfile
        result = ConverterEngine().convert(parsed, sample_workflow_file.stem)
        blob = to_dbc_bytes(result.code, "nb")
        with zipfile.ZipFile(io.BytesIO(blob)) as zf:
            payload = json.loads(zf.read("nb.python"))
        assert payload["version"] == "NotebookV1"
        assert payload["language"] == "python"
        assert payload["commands"]

    def test_export_writes_all_formats(self, parsed, sample_workflow_file, tmp_path):
        result = SelfCorrectingConverter().convert(parsed, sample_workflow_file.stem)
        written = DatabricksExporter(str(tmp_path)).export(
            result, formats=("ipynb", "py", "dbc"))
        for path in written.values():
            assert Path(path).exists()
        nb = json.loads(Path(written["ipynb"]).read_text())
        assert nb["nbformat"] == 4

    def test_batch_export(self, tmp_path):
        summaries = batch_export(str(SAMPLES_DIR), str(tmp_path), formats=("ipynb",))
        assert len(summaries) >= 3
        assert (tmp_path / "batch_summary.json").exists()
        by_name = {s["workflow"]: s for s in summaries}
        for name in SAMPLES:
            assert by_name[name]["status"] in ("PASS", "WARNING")

    def test_dbc_contains_validation_report(self, parsed, sample_workflow_file, tmp_path):
        """Regression: the .dbc export must embed the report like ipynb/py do."""
        import io
        import zipfile
        result = SelfCorrectingConverter().convert(parsed, sample_workflow_file.stem)
        written = DatabricksExporter(str(tmp_path)).export(result, formats=("dbc",))
        with zipfile.ZipFile(io.BytesIO(Path(written["dbc"]).read_bytes())) as zf:
            payload = json.loads(zf.read(zf.namelist()[0]))
        first_command = payload["commands"][0]["command"]
        assert first_command.startswith("%md")
        assert "Conversion Validation Report" in first_command


# ── Copilot-review regression fixes ────────────────────────────────

class TestReviewRegressions:
    def test_report_markdown_uses_configured_max_iterations(self, parsed, sample_workflow_file):
        converter = SelfCorrectingConverter(max_iterations=2)
        result = converter.convert(parsed, sample_workflow_file.stem)
        assert "of 2" in result.report_markdown()

    def test_csv_to_delta_optimization_stays_parseable(self):
        """Regression: the Delta rewrite must not swallow the closing paren."""
        import ast
        from src.self_correction import SelfCorrectingConverter
        from src.validators import StructuralIssue, StructuralReport

        code = 'df_1 = spark.table("a.b.c")\ndf_1.write.csv("/tmp/out.csv", header=True)\n'
        report = StructuralReport()
        report.issues.append(StructuralIssue("WARNING", None, "OutputData",
                                             "consider Delta Lake format"))
        optimized = SelfCorrectingConverter().apply_optimizations(code, report, {"tools": []})
        ast.parse(optimized)
        assert '.write.format("delta")' in optimized

    def test_var_name_collision_dedupe(self):
        """Regression: identical annotations must not share one variable."""
        from src.models import Container, Tool, Workflow
        from src.tools import GeneratorContext

        t1 = Tool(1, "", "Formula", {}, "", "Clean Data")
        t2 = Tool(2, "", "Formula", {}, "", "Clean Data")
        wf = Workflow(containers=[], all_containers={}, all_tools={1: t1, 2: t2},
                      connections=[], text_inputs={})
        ctx = GeneratorContext(wf, [t1, t2], [])
        name1 = ctx.make_var_name(t1)
        name2 = ctx.make_var_name(t2)
        assert name1 == "df_clean_data"
        assert name2 != name1

    def test_output_temp_view_uses_file_stem(self):
        """Regression: 'output.csv' must give view 'output', not 'csv'."""
        from src.tools.builtin import _table_leaf_name
        assert _table_leaf_name("output.csv") == "output"
        assert _table_leaf_name("path/to/book.xlsx") == "book"
        assert _table_leaf_name("catalog.schema.table") == "table"
        assert _table_leaf_name("bare_table") == "bare_table"

    def test_output_table_qualified_with_target_catalog(self, parsed):
        """Bare output table names get catalog.schema qualification."""
        from src.models import Tool, Workflow
        from src.tools import GeneratorContext
        from src.tools.builtin import OutputDataConverter

        tool = Tool(9, "", "OutputData", {}, "", "", parsed_config={"table_name": "results"})
        wf = Workflow(containers=[], all_containers={}, all_tools={9: tool},
                      connections=[], text_inputs={})
        ctx = GeneratorContext(wf, [tool], [], target_catalog="main",
                               target_schema="alteryx_migrated")
        lines = OutputDataConverter().convert(tool, ctx)
        joined = "\n".join(lines)
        assert "main.alteryx_migrated.results" in joined
        # Already-qualified and file targets stay untouched.
        tool2 = Tool(10, "", "OutputData", {}, "", "",
                     parsed_config={"table_name": "cat.sch.tbl"})
        ctx2 = GeneratorContext(wf, [tool2], [], target_catalog="main",
                                target_schema="alteryx_migrated")
        assert "cat.sch.tbl" in "\n".join(OutputDataConverter().convert(tool2, ctx2))

    def test_summarize_first_last_flagged_order_dependent(self):
        from src.models import Tool, Workflow
        from src.tools import GeneratorContext
        from src.tools.builtin import SummarizeConverter

        tool = Tool(3, "", "Summarize", {}, "", "", parsed_config={
            "summarize_fields": [
                {"field": "Region", "action": "GroupBy", "rename": ""},
                {"field": "Amount", "action": "First", "rename": "first_amount"},
            ]})
        wf = Workflow(containers=[], all_containers={}, all_tools={3: tool},
                      connections=[], text_inputs={})
        lines = SummarizeConverter().convert(tool, GeneratorContext(wf, [tool], []))
        assert any("order-dependent" in line for line in lines)

    def test_html_report_escapes_untrusted_values(self):
        from src.validators import ReconciliationReporter, ValidationReport

        report = ValidationReport(workflow_name="<script>alert(1)</script>")
        report.recommendations = ["<img src=x onerror=alert(1)>"]
        html = ReconciliationReporter().to_html(report)
        assert "<script>alert(1)</script>" not in html
        assert "&lt;script&gt;" in html
        assert "<img src=x" not in html

    def test_batch_cli_honors_source_tables_config(self, tmp_path):
        """Regression: --batch must honor --source-tables-config + --self-correct."""
        import subprocess
        import sys
        cfg = tmp_path / "sources.json"
        cfg.write_text(json.dumps({"1": "main.mapped.transactions"}))
        out_dir = tmp_path / "out"
        src_dir = tmp_path / "flows"
        src_dir.mkdir()
        (src_dir / "flow.yxmd").write_bytes(
            (SAMPLES_DIR / "02_join_summarize.yxmd").read_bytes())
        subprocess.run(
            [sys.executable, "convert.py", str(src_dir), "--batch", "--self-correct",
             "--format", "py", "--output-dir", str(out_dir),
             "--source-tables-config", str(cfg)],
            check=True, cwd=Path(__file__).parent.parent, capture_output=True)
        generated = (out_dir / "flow.py").read_text()
        assert "main.mapped.transactions" in generated
