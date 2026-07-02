"""
Self-Correcting Validation Loop
================================
Wraps the converter engine in a generate → validate → correct cycle:

1. **Generate** a first-pass notebook with the standard engine.
2. **Validate** structurally (AST vs. the original .yxmd DAG: tool coverage,
   join keys, filter fields, unresolved references, syntax) and — when a live
   Spark session is provided — by executing the generated code on sample data
   and comparing row counts / schemas.
3. **Correct**:
   - FAILED  → regenerate in *strict mode* (granular re-parse of each tool's
     raw XML configuration, stricter mapping annotations).
   - WARNING → apply non-semantic optimization rules (broadcast small joins,
     prefer Delta writes, ZORDER hints) without changing core logic.
4. Repeat up to ``max_iterations`` (default 3). If still failing, the best
   attempt is returned with a detailed error report that the exporter places
   in a ``%md`` cell at the top of the notebook.
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

from .converter_engine import ConversionResult, ConverterEngine
from .models import Workflow
from .validators import (
    StructuralReport,
    StructuralValidator,
    ValidationStatus,
    run_generated_code,
)

logger = logging.getLogger(__name__)

MAX_ITERATIONS_DEFAULT = 3


@dataclass
class CorrectionIteration:
    """One pass through the generate/validate/correct cycle."""
    iteration: int
    strict: bool
    status: str                      # PASS | WARNING | FAIL
    failures: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    action: str = ""                 # what the loop did in response
    runtime_ok: Optional[bool] = None
    runtime_error: str = ""


@dataclass
class SelfCorrectionResult:
    """Final outcome of the self-correction loop."""
    status: str                      # PASS | WARNING | FAIL (best effort)
    conversion: ConversionResult
    structural_report: Optional[StructuralReport] = None
    iterations: list = field(default_factory=list)
    runtime_result: Optional[dict] = None

    @property
    def code(self) -> str:
        return self.conversion.code

    def report_markdown(self) -> str:
        """Validation report rendered as Markdown for the notebook's top %md cell."""
        icon = {"PASS": "✅", "WARNING": "⚠️", "FAIL": "❌"}.get(self.status, "❓")
        lines = [
            f"## {icon} Conversion Validation Report — {self.status}",
            "",
            f"- **Workflow:** {self.conversion.workflow_name}",
            f"- **Tools converted:** {self.conversion.num_tools} "
            f"({self.conversion.complexity} complexity, "
            f"{self.conversion.num_joins} joins, {self.conversion.num_formulas} formulas)",
            f"- **Self-correction iterations used:** {len(self.iterations)} "
            f"of {MAX_ITERATIONS_DEFAULT}",
            "",
            "| Iteration | Mode | Result | Failures | Warnings | Action taken |",
            "|---|---|---|---|---|---|",
        ]
        for it in self.iterations:
            mode = "strict" if it.strict else "standard"
            lines.append(
                f"| {it.iteration} | {mode} | {it.status} | "
                f"{len(it.failures)} | {len(it.warnings)} | {it.action} |"
            )

        report = self.structural_report
        if report and report.failures:
            lines += ["", "### ❌ Unresolved mismatches (best-effort output)"]
            for issue in report.failures:
                where = f"tool {issue.tool_id} ({issue.tool_type})" if issue.tool_id else "notebook"
                lines.append(f"- **{where}**: {issue.message}")
        if report and report.warnings:
            lines += ["", "### ⚠️ Review recommended"]
            for issue in report.warnings:
                lines.append(f"- {issue.message}")

        if self.runtime_result is not None:
            lines += ["", "### Sample-data execution"]
            if self.runtime_result.get("ok"):
                lines.append("- Generated code executed successfully against sample data.")
                for name, meta in sorted(self.runtime_result.get("dataframes", {}).items()):
                    lines.append(f"  - `{name}`: {meta['rows']} rows × {len(meta['columns'])} columns")
            else:
                lines.append(f"- ❌ Execution failed: {self.runtime_result.get('error')}")

        unsupported = self.conversion.unsupported_tools
        if unsupported:
            lines += ["", "### 🔧 Tools requiring manual conversion"]
            for s in unsupported:
                lines.append(f"- Tool {s.tool_id} ({s.tool_type}): {s.message}")
        return "\n".join(lines)


class SelfCorrectingConverter:
    """
    Orchestrates the conversion + validation + correction loop.

    Args:
        engine: a configured :class:`ConverterEngine` (a default one is built
            when omitted).
        max_iterations: maximum generate/validate cycles (default 3).
    """

    def __init__(
        self,
        engine: Optional[ConverterEngine] = None,
        max_iterations: int = MAX_ITERATIONS_DEFAULT,
    ):
        self.engine = engine or ConverterEngine()
        self.max_iterations = max_iterations
        self.validator = StructuralValidator()

    def convert(
        self,
        workflow: Workflow,
        workflow_name: str = "workflow",
        context: Optional[dict] = None,
        spark=None,
        progress_callback=None,
    ) -> SelfCorrectionResult:
        """
        Run the self-correcting conversion.

        Args:
            workflow: parsed Alteryx workflow.
            workflow_name: base name for the generated notebook.
            context: optional pre-built unified context.
            spark: optional live SparkSession — enables sample-data execution
                and row-count/schema validation on top of structural checks.
            progress_callback: optional ``fn(iteration:int, status:str, message:str)``
                used by the Skill Mode notebook to render live progress.
        """
        if context is None:
            context = workflow.get_unified_context()

        iterations: list = []
        best: Optional[tuple] = None  # (fail_count, result, report, runtime)
        strict = False
        runtime_result = None

        for i in range(1, self.max_iterations + 1):
            result = self.engine.convert(workflow, workflow_name, context, strict=strict)
            report = self.validator.validate(result.code, workflow, context)

            runtime_result = None
            if spark is not None and report.syntax_ok:
                runtime_result = run_generated_code(result.code, spark)
                if not runtime_result["ok"]:
                    report.status = ValidationStatus.FAIL
                    from .validators import StructuralIssue
                    report.issues.append(StructuralIssue(
                        "FAILED", None, "",
                        f"Sample-data execution failed: {runtime_result['error']}"))

            status = report.status.value
            it = CorrectionIteration(
                iteration=i,
                strict=strict,
                status=status,
                failures=[x.message for x in report.failures],
                warnings=[x.message for x in report.warnings],
                runtime_ok=None if runtime_result is None else runtime_result["ok"],
                runtime_error="" if runtime_result is None else runtime_result.get("error", ""),
            )

            fail_count = len(report.failures)
            if best is None or fail_count < best[0]:
                best = (fail_count, result, report, runtime_result)

            if status == "PASS":
                it.action = "validated clean — exported"
                iterations.append(it)
                self._notify(progress_callback, i, status, it.action)
                break

            if status == "WARNING":
                it.action = "applied optimization rules (broadcast/Delta/ZORDER)"
                iterations.append(it)
                self._notify(progress_callback, i, status, it.action)
                optimized_code = self.apply_optimizations(result.code, report, context)
                # Re-validate: optimizations must not break the structure.
                re_report = self.validator.validate(optimized_code, workflow, context)
                if not re_report.failures:
                    result.code = optimized_code
                    report = re_report
                best = (len(report.failures), result, report, runtime_result)
                break

            # FAILED — escalate to strict mode and regenerate.
            if i < self.max_iterations:
                it.action = "regenerating with strict granular re-parse"
                strict = True
            else:
                it.action = "max iterations reached — emitting best attempt with error report"
            iterations.append(it)
            self._notify(progress_callback, i, status, it.action)

        fail_count, result, report, runtime_result = best
        final_status = ("FAIL" if report.failures
                        else "WARNING" if report.warnings
                        else "PASS")
        logger.info("Self-correction finished: %s after %d iteration(s), %d failure(s)",
                    final_status, len(iterations), fail_count)

        return SelfCorrectionResult(
            status=final_status,
            conversion=result,
            structural_report=report,
            iterations=iterations,
            runtime_result=runtime_result,
        )

    # ── optimization rules (WARNING remediation) ──────────────────────

    def apply_optimizations(self, code: str, report: StructuralReport, context: dict) -> str:
        """
        Apply non-semantic Databricks optimizations flagged as WARNINGs:
        - broadcast joins when one side is a small inline TextInput
        - Delta Lake writes instead of CSV
        - ZORDER hint comments on Delta table writes
        Core transformation logic is never altered.
        """
        optimized = code
        messages = [w.message for w in report.warnings]

        if any("broadcast" in m.lower() for m in messages):
            text_input_vars = self._text_input_vars(context)
            for var in text_input_vars:
                optimized = optimized.replace(
                    f".join({var},", f".join(F.broadcast({var}),")

        if any("Delta" in m for m in messages):
            optimized = optimized.replace(
                ".write.csv(",
                '.write.format("delta").mode("overwrite").save(  # optimized from CSV: ',
            )

        if 'saveAsTable("' in optimized and "ZORDER" not in optimized:
            optimized += (
                "\n# OPTIMIZE: after the first production write, consider:\n"
                "#   OPTIMIZE <catalog>.<schema>.<table> ZORDER BY (<join/filter columns>)\n"
            )
        return optimized

    @staticmethod
    def _text_input_vars(context: dict) -> list:
        """Variable names generated for TextInput tools (broadcast candidates)."""
        variables = []
        for tool in context.get("tools", []):
            if tool.tool_type == "TextInput":
                ann = (tool.annotation or "").strip()
                # Mirror GeneratorContext.make_var_name for TextInput tools.
                if ann and len(ann) > 2:
                    import re
                    cleaned = re.sub(r"_+", "_", re.sub(r"[^a-z0-9_]", "_", ann.lower())).strip("_")
                    if cleaned and len(cleaned) <= 40:
                        variables.append(f"df_{cleaned}")
                        continue
                variables.append(f"df_text_input_{tool.tool_id}")
        return variables

    @staticmethod
    def _notify(callback, iteration: int, status: str, message: str):
        if callback is not None:
            try:
                callback(iteration, status, message)
            except Exception:
                logger.debug("progress callback raised", exc_info=True)
