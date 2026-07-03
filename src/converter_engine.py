"""
Converter Engine
=================
Core Alteryx → PySpark conversion engine.

Responsibilities:
- Load the YAML-driven tool mapping (``config/tool_mapping.yaml``) and build
  the converter map from the plugin registry (``src/tools``).
- Topologically sort the workflow DAG and convert each tool via its plugin.
- Track per-tool conversion status (converted / partial / unsupported / error)
  so the self-correction loop and the Skill Mode notebook can report progress.
- Assemble a Databricks-source-format notebook string with ``%md``
  documentation cells, sequential DataFrame tracking, and graceful fallbacks
  for unsupported tools.

Strict mode (used by the self-correction loop on a FAILED validation) re-parses
each tool's raw ``<Configuration>`` XML to recover configurations that the
first-pass parse may have missed, and annotates any still-unsupported tool
with its full configuration for manual follow-up.
"""

import datetime
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .models import Workflow, Tool
from .parser import ET as _XML, SAFE_XML_PARSER, _extract_structured_config
from .tools import GeneratorContext, build_converter_map, get_converter, topological_sort

logger = logging.getLogger(__name__)

DEFAULT_TOOL_MAPPING_PATH = Path(__file__).resolve().parent.parent / "config" / "tool_mapping.yaml"

# Tool types that never produce runnable transformations.
NON_CODE_TOOLS = ("Container", "Comment")
INPUT_TOOLS = ("InputData", "LockInStreamIn", "DynamicInput", "TextInput")
OUTPUT_TOOLS = ("OutputData", "LockInStreamOut")


def load_tool_mapping(path: Optional[str] = None) -> dict:
    """Load config/tool_mapping.yaml (returns {} when missing or PyYAML absent)."""
    mapping_path = Path(path) if path else DEFAULT_TOOL_MAPPING_PATH
    if not mapping_path.exists():
        return {}
    try:
        import yaml
    except ImportError:
        logger.warning("PyYAML not installed; using built-in tool registry only")
        return {}
    with open(mapping_path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


@dataclass
class ToolStatus:
    """Conversion outcome for a single Alteryx tool."""
    tool_id: int
    tool_type: str
    annotation: str = ""
    status: str = "converted"  # converted | partial | unsupported | error | skipped
    message: str = ""


@dataclass
class ConversionResult:
    """Everything the exporter / validator needs about one conversion pass."""
    workflow_name: str
    code: str = ""                       # Databricks source-format notebook
    tool_statuses: list = field(default_factory=list)
    strict: bool = False
    num_tools: int = 0
    num_joins: int = 0
    num_formulas: int = 0
    complexity: str = "Low"

    @property
    def unsupported_tools(self) -> list:
        return [s for s in self.tool_statuses if s.status in ("unsupported", "error")]

    @property
    def partial_tools(self) -> list:
        return [s for s in self.tool_statuses if s.status == "partial"]

    def summary(self) -> dict:
        counts: dict = {}
        for s in self.tool_statuses:
            counts[s.status] = counts.get(s.status, 0) + 1
        return {
            "workflow": self.workflow_name,
            "tools": self.num_tools,
            "complexity": self.complexity,
            "strict_mode": self.strict,
            **counts,
        }


class ConverterEngine:
    """
    YAML-driven, plugin-based Alteryx → PySpark conversion engine.

    Args:
        tool_mapping_path: path to tool_mapping.yaml (defaults to config/).
        source_tables_config: optional {alteryx input id/name: catalog.schema.table}.
        target_catalog / target_schema: Unity Catalog defaults injected into
            generated output-table references when the Alteryx output has no
            three-level name.
    """

    def __init__(
        self,
        tool_mapping_path: Optional[str] = None,
        source_tables_config: Optional[dict] = None,
        target_catalog: str = "",
        target_schema: str = "",
    ):
        self.tool_mapping = load_tool_mapping(tool_mapping_path)
        self.converter_map = build_converter_map(self.tool_mapping)
        self.source_tables_config = source_tables_config or {}
        self.target_catalog = target_catalog
        self.target_schema = target_schema

    # ── public API ────────────────────────────────────────────────────

    def convert(
        self,
        workflow: Workflow,
        workflow_name: str = "workflow",
        context: Optional[dict] = None,
        strict: bool = False,
    ) -> ConversionResult:
        """Convert a parsed workflow into a Databricks notebook (source format)."""
        if context is None:
            context = workflow.get_unified_context()

        tools = context.get("tools", [])
        connections = context.get("internal_connections", [])

        if strict:
            self._reextract_configs(tools)

        ctx = GeneratorContext(
            workflow=workflow,
            tools=tools,
            connections=connections,
            source_tables_config=self.source_tables_config,
            target_catalog=self.target_catalog,
            target_schema=self.target_schema,
        )

        ordered_ids = topological_sort(tools, connections)
        tool_map = {t.tool_id: t for t in tools}

        code_sections = {"sources": [], "transformations": [], "outputs": []}
        statuses: list = []

        for tid in ordered_ids:
            tool = tool_map.get(tid)
            if tool is None:
                continue
            if tool.tool_type in NON_CODE_TOOLS:
                statuses.append(ToolStatus(tid, tool.tool_type, tool.annotation, "skipped",
                                           "annotation/container only"))
                continue

            lines, status = self._convert_tool(tool, ctx, strict)
            statuses.append(status)

            if tool.tool_type in INPUT_TOOLS:
                code_sections["sources"].extend(lines)
            elif tool.tool_type in OUTPUT_TOOLS:
                code_sections["outputs"].extend(lines)
            else:
                code_sections["transformations"].extend(lines)

        num_tools = len([t for t in tools if t.tool_type not in NON_CODE_TOOLS])
        num_joins = len([t for t in tools if t.tool_type in ("Join", "LockInJoin")])
        num_formulas = len([t for t in tools if t.tool_type in ("Formula", "LockInFormula")])
        complexity = "Low" if num_tools < 10 else ("Medium" if num_tools < 30 else "High")

        code = self._assemble_notebook(
            workflow_name=workflow_name,
            code_sections=code_sections,
            complexity=complexity,
            num_tools=num_tools,
            num_joins=num_joins,
            num_formulas=num_formulas,
            statuses=statuses,
            strict=strict,
        )

        return ConversionResult(
            workflow_name=workflow_name,
            code=code,
            tool_statuses=statuses,
            strict=strict,
            num_tools=num_tools,
            num_joins=num_joins,
            num_formulas=num_formulas,
            complexity=complexity,
        )

    # ── internals ─────────────────────────────────────────────────────

    def _convert_tool(self, tool: Tool, ctx: GeneratorContext, strict: bool):
        """Convert one tool, returning (code_lines, ToolStatus)."""
        converter = get_converter(tool.tool_type, self.converter_map)
        is_registered = tool.tool_type in self.converter_map

        try:
            lines = converter.convert(tool, ctx)
        except Exception as exc:  # graceful fallback — never abort the whole notebook
            logger.exception("Converter for tool %s (%s) raised", tool.tool_id, tool.tool_type)
            input_var = ctx.get_input_var(tool.tool_id)
            var = f"df_{tool.tool_id}"
            ctx.set_output_var(tool.tool_id, var)
            lines = [
                f"# Tool {tool.tool_id}: {tool.tool_type} — CONVERSION ERROR: {exc}",
                f"{var} = {input_var}  # passthrough fallback; review manually",
            ]
            return lines, ToolStatus(tool.tool_id, tool.tool_type, tool.annotation,
                                     "error", str(exc))

        joined = "\n".join(lines)
        if not is_registered:
            msg = (f"No converter registered for Alteryx tool '{tool.tool_type}'. "
                   f"Passed through unchanged — add a plugin in src/tools/ or map it "
                   f"in config/tool_mapping.yaml.")
            lines = [f"# ⚠️ UNSUPPORTED TOOL — {msg}"] + lines
            if strict and tool.configuration_xml:
                lines.append("# Original Alteryx configuration for manual conversion:")
                for xml_line in tool.configuration_xml.splitlines()[:30]:
                    lines.append(f"#   {xml_line}")
            return lines, ToolStatus(tool.tool_id, tool.tool_type, tool.annotation,
                                     "unsupported", msg)

        if "TODO" in joined:
            return lines, ToolStatus(tool.tool_id, tool.tool_type, tool.annotation,
                                     "partial", "generated code contains TODO placeholders")

        return lines, ToolStatus(tool.tool_id, tool.tool_type, tool.annotation, "converted")

    def _reextract_configs(self, tools: list):
        """
        Strict-mode granular re-parse: for tools whose structured config is
        empty, re-parse the raw <Configuration> XML captured by the parser.
        """
        for tool in tools:
            if tool.parsed_config or not tool.configuration_xml:
                continue
            try:
                config_el = _XML.fromstring(tool.configuration_xml.encode("utf-8"),
                                            SAFE_XML_PARSER)
                reparsed = _extract_structured_config(config_el, tool.tool_type)
                if reparsed:
                    tool.parsed_config = reparsed
                    logger.info("Strict mode recovered config for tool %s (%s)",
                                tool.tool_id, tool.tool_type)
            except Exception as exc:
                logger.warning("Strict re-parse failed for tool %s: %s", tool.tool_id, exc)

    def _assemble_notebook(
        self,
        workflow_name: str,
        code_sections: dict,
        complexity: str,
        num_tools: int,
        num_joins: int,
        num_formulas: int,
        statuses: Optional[list] = None,
        strict: bool = False,
    ) -> str:
        """Assemble code sections into a Databricks source-format notebook."""
        parts = []
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        mode = "strict" if strict else "standard"

        parts.append("# Databricks notebook source")
        parts.append("")
        parts.append("# COMMAND ----------")
        parts.append("")
        parts.append("# MAGIC %md")
        parts.append(f"# MAGIC # {workflow_name}")
        parts.append("# MAGIC Auto-generated by the Alteryx → Databricks Converter Engine")
        parts.append(f"# MAGIC Source: {workflow_name}.yxmd | Mode: {mode}")
        parts.append(f"# MAGIC Generated: {timestamp}")
        parts.append(f"# MAGIC Complexity: {complexity} ({num_tools} tools, {num_joins} joins, {num_formulas} formulas)")

        unsupported = [s for s in (statuses or []) if s.status in ("unsupported", "error")]
        if unsupported:
            parts.append("# MAGIC")
            parts.append(f"# MAGIC **⚠️ {len(unsupported)} tool(s) need manual attention:**")
            for s in unsupported:
                parts.append(f"# MAGIC - Tool {s.tool_id} ({s.tool_type}): {s.message}")

        parts.append("")
        parts.append("# COMMAND ----------")
        parts.append("")
        parts.append("from pyspark.sql import functions as F")
        parts.append("from pyspark.sql.types import *")
        parts.append("from pyspark.sql.window import Window")

        section_titles = {
            "sources": "Input Sources",
            "transformations": "Transformations",
            "outputs": "Output",
        }
        for key in ("sources", "transformations", "outputs"):
            if not code_sections[key]:
                continue
            parts.append("")
            parts.append("# COMMAND ----------")
            parts.append("")
            parts.append("# MAGIC %md")
            parts.append(f"# MAGIC ## {section_titles[key]}")
            parts.append("")
            parts.append("# COMMAND ----------")
            parts.append("")
            parts.extend(code_sections[key])

        parts.append("")
        parts.append("# COMMAND ----------")
        parts.append("")
        parts.append("# MAGIC %md")
        parts.append("# MAGIC ## Validation")
        parts.append("")
        parts.append("# COMMAND ----------")
        parts.append("")
        parts.append("# Row count sanity check")
        parts.append("# Uncomment and set df_final to your final output DataFrame")
        parts.append("# row_count = df_final.count()")
        parts.append('# print(f"Final row count: {row_count:,}")')
        parts.append('# assert row_count > 0, "Output DataFrame is empty!"')
        parts.append("# df_final.limit(5).display()")
        parts.append("")

        return "\n".join(parts)


class PySparkCodeGenerator:
    """
    Backwards-compatible facade over :class:`ConverterEngine`.

    Preserves the original ``generate(workflow, workflow_name, context) -> str``
    API used by earlier releases and the existing test suite.
    """

    def __init__(self, source_tables_config: Optional[dict] = None):
        self.source_tables_config = source_tables_config or {}
        self._engine = ConverterEngine(source_tables_config=source_tables_config)

    def generate(
        self,
        workflow: Workflow,
        workflow_name: str = "workflow",
        context: Optional[dict] = None,
    ) -> str:
        return self._engine.convert(workflow, workflow_name, context).code
