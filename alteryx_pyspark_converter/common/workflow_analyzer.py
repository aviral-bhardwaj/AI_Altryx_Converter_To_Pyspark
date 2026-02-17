"""Autonomous workflow analyzer.

Analyzes an Alteryx workflow to automatically:
- Detect all data sources and suggest Databricks table mappings
- Identify transformation patterns and complexity
- Build a recommended execution plan
- Infer column mappings from tool configurations
- Generate a config YAML template automatically
"""

import logging
import re
import yaml
from collections import Counter, defaultdict
from typing import Optional

from ..phase1_parser.models.workflow import Workflow
from ..phase1_parser.models.tool import Tool
from ..phase2_generator.generator.flow_analyzer import FlowAnalyzer
from ..phase2_generator.generator.dependency_resolver import DependencyResolver

logger = logging.getLogger(__name__)


class WorkflowAnalyzer:
    """Autonomously analyze an Alteryx workflow and generate configuration."""

    def __init__(self, workflow: Workflow):
        self.workflow = workflow
        self.flow_analyzer = FlowAnalyzer(workflow)

    def analyze(self) -> dict:
        """
        Perform a full analysis of the workflow.

        Returns a dictionary with:
        - summary: Overall workflow summary
        - containers: List of container analyses
        - data_sources: Detected data sources
        - complexity: Complexity metrics
        - suggested_config: Auto-generated YAML config
        - warnings: Potential issues detected
        """
        result = {
            "summary": self._build_summary(),
            "containers": [],
            "data_sources": self._detect_data_sources(),
            "complexity": self._assess_complexity(),
            "column_inventory": self._inventory_columns(),
            "suggested_config": {},
            "warnings": [],
        }

        # Analyze each container
        for container in self.workflow.containers:
            container_analysis = self._analyze_container(container)
            result["containers"].append(container_analysis)

            # Generate suggested config for each container
            result["suggested_config"][container.name] = (
                self._generate_config(container)
            )

        # Detect warnings
        result["warnings"] = self._detect_warnings()

        return result

    def print_report(self) -> str:
        """Generate a human-readable analysis report."""
        analysis = self.analyze()
        lines = []

        lines.append("=" * 70)
        lines.append("  ALTERYX WORKFLOW ANALYSIS REPORT")
        lines.append("=" * 70)
        lines.append("")

        # Summary
        summary = analysis["summary"]
        lines.append(f"Workflow: {summary['name']}")
        lines.append(f"Total Tools: {summary['total_tools']}")
        lines.append(f"Total Containers: {summary['total_containers']}")
        lines.append(f"Total Connections: {summary['total_connections']}")
        lines.append("")

        # Tool type breakdown
        lines.append("Tool Type Distribution:")
        for tool_type, count in sorted(
            summary["tool_types"].items(), key=lambda x: -x[1]
        ):
            lines.append(f"  {tool_type}: {count}")
        lines.append("")

        # Data sources
        sources = analysis["data_sources"]
        if sources:
            lines.append("Data Sources Detected:")
            for src in sources:
                lines.append(f"  Tool {src['tool_id']} ({src['type']})")
                if src.get("table_name"):
                    lines.append(f"    Table: {src['table_name']}")
                if src.get("annotation"):
                    lines.append(f"    Label: {src['annotation']}")
                if src.get("suggested_databricks_table"):
                    lines.append(
                        f"    Suggested Databricks: {src['suggested_databricks_table']}"
                    )
            lines.append("")

        # Complexity
        complexity = analysis["complexity"]
        lines.append(f"Complexity Score: {complexity['score']}/10 ({complexity['level']})")
        lines.append(f"  Branches (joins/unions): {complexity['branches']}")
        lines.append(f"  Filter chains: {complexity['filter_chains']}")
        lines.append(f"  Expression complexity: {complexity['expression_complexity']}")
        lines.append("")

        # Container details
        for container_info in analysis["containers"]:
            lines.append("-" * 50)
            lines.append(f"Container: {container_info['name']}")
            lines.append(f"  Tools: {container_info['tool_count']}")
            lines.append(f"  External inputs: {container_info['external_inputs']}")
            lines.append(f"  Execution order: {container_info['execution_order']}")
            lines.append(
                f"  Pipeline description: {container_info['description']}"
            )
            lines.append("")

        # Warnings
        warnings = analysis["warnings"]
        if warnings:
            lines.append("WARNINGS:")
            for w in warnings:
                lines.append(f"  - {w}")
            lines.append("")

        # Suggested config
        for container_name, config in analysis["suggested_config"].items():
            lines.append(f"Suggested Config for '{container_name}':")
            lines.append(yaml.dump(config, default_flow_style=False, indent=2))
            lines.append("")

        return "\n".join(lines)

    def generate_config_yaml(self, container_name: str = "") -> str:
        """Generate a YAML configuration file for the given container.

        If container_name is empty, generates for the first container.
        """
        target = None
        if container_name:
            for c in self.workflow.containers:
                if c.name.lower() == container_name.lower() or \
                   container_name.lower() in c.name.lower():
                    target = c
                    break
        if target is None and self.workflow.containers:
            target = self.workflow.containers[0]

        if target is None:
            return "# No containers found in workflow\n"

        config = self._generate_config(target)
        return yaml.dump(config, default_flow_style=False, indent=2)

    # ------------------------------------------------------------------
    # Internal analysis methods
    # ------------------------------------------------------------------

    def _get_all_tools(self):
        """Get all tools as an iterable, supporting both Workflow models."""
        all_tools = self.workflow.all_tools
        if isinstance(all_tools, dict):
            return list(all_tools.values())
        return list(all_tools) if hasattr(all_tools, '__iter__') else []

    @staticmethod
    def _get_tool_config(tool) -> dict:
        """Get tool configuration dict, supporting both Tool models.

        src.models.Tool uses ``parsed_config``; phase1 Tool uses ``configuration``.
        """
        cfg = getattr(tool, 'configuration', None)
        if cfg is not None and isinstance(cfg, dict):
            return cfg
        cfg = getattr(tool, 'parsed_config', None)
        if cfg is not None and isinstance(cfg, dict):
            return cfg
        return {}

    def _build_summary(self) -> dict:
        """Build an overall workflow summary."""
        tool_types = Counter()
        tools = list(self._get_all_tools())
        for tool in tools:
            tool_types[tool.tool_type] += 1

        name = getattr(self.workflow, 'name', '') or 'Workflow'

        return {
            "name": name,
            "total_tools": len(tools),
            "total_containers": len(self.workflow.containers),
            "total_connections": len(self.workflow.connections),
            "tool_types": dict(tool_types),
        }

    def _detect_data_sources(self) -> list[dict]:
        """Detect all data source tools and suggest Databricks mappings."""
        sources = []
        input_types = {
            "DbInput", "LockInInput", "InputData", "LockInStreamIn",
            "TextInput",
        }

        for tool in self._get_all_tools():
            if tool.tool_type in input_types:
                src = {
                    "tool_id": tool.tool_id,
                    "type": tool.tool_type,
                    "annotation": tool.annotation,
                    "table_name": "",
                    "suggested_databricks_table": "",
                    "suggested_key": "",
                }

                cfg = self._get_tool_config(tool)
                table_name = cfg.get("table_name", "") or cfg.get("query", "")
                if table_name:
                    src["table_name"] = table_name
                    # Suggest a Databricks table name
                    src["suggested_databricks_table"] = (
                        self._suggest_databricks_table(table_name)
                    )
                    # Suggest a config key
                    src["suggested_key"] = self._suggest_config_key(
                        table_name, tool.annotation
                    )

                if tool.tool_type == "TextInput":
                    data = cfg.get("data", [])
                    src["inline_data_rows"] = len(data)

                sources.append(src)

        return sources

    def _assess_complexity(self) -> dict:
        """Assess the complexity of the workflow."""
        tool_types = Counter()
        for tool in self._get_all_tools():
            tool_types[tool.tool_type] += 1

        branches = tool_types.get("Join", 0) + tool_types.get("Union", 0) + \
                   tool_types.get("LockInJoin", 0) + tool_types.get("LockInUnion", 0)
        filter_chains = tool_types.get("Filter", 0) + tool_types.get("LockInFilter", 0)

        # Count expression complexity
        expression_complexity = 0
        for tool in self._get_all_tools():
            cfg = self._get_tool_config(tool)
            if "formulas" in cfg:
                for f in cfg["formulas"]:
                    expr = f.get("expression", "")
                    if "IF" in expr.upper():
                        expression_complexity += 2
                    elif len(expr) > 50:
                        expression_complexity += 1
            if "expression" in cfg:
                expr = cfg["expression"]
                if "IF" in str(expr).upper():
                    expression_complexity += 2

        total_tools = len(list(self._get_all_tools()))
        score = min(10, (
            total_tools // 5 +
            branches * 2 +
            filter_chains +
            expression_complexity // 3
        ))

        level = "Simple"
        if score >= 7:
            level = "Complex"
        elif score >= 4:
            level = "Moderate"

        return {
            "score": score,
            "level": level,
            "branches": branches,
            "filter_chains": filter_chains,
            "expression_complexity": expression_complexity,
            "total_tools": total_tools,
        }

    def _inventory_columns(self) -> dict:
        """Inventory all column names referenced across the workflow."""
        columns = defaultdict(set)

        for tool in self._get_all_tools():
            cfg = self._get_tool_config(tool)

            # Formula fields
            for f in cfg.get("formulas", []):
                field = f.get("field", "")
                if field:
                    columns[field].add(f"Formula:{tool.tool_id}")
                # Extract referenced columns from expression
                expr = f.get("expression", "")
                for col in re.findall(r"\[([^\]]+)\]", expr):
                    columns[col].add(f"FormulaRef:{tool.tool_id}")

            # Join keys
            for key in cfg.get("left_keys", []):
                columns[key].add(f"JoinLeft:{tool.tool_id}")
            for key in cfg.get("right_keys", []):
                columns[key].add(f"JoinRight:{tool.tool_id}")

            # Select fields
            for sf in cfg.get("fields", []):
                name = sf.get("name", sf.get("field", ""))
                if name and not name.startswith("*"):
                    columns[name].add(f"Select:{tool.tool_id}")

            # Filter expression columns
            expr = cfg.get("expression", "")
            if isinstance(expr, str):
                for col in re.findall(r"\[([^\]]+)\]", expr):
                    columns[col].add(f"Filter:{tool.tool_id}")

            # Sort fields
            for sf in cfg.get("fields", []):
                field = sf.get("field", "")
                if field:
                    columns[field].add(f"Sort:{tool.tool_id}")

            # Unique fields
            for uf in cfg.get("unique_fields", []):
                columns[uf].add(f"Unique:{tool.tool_id}")

            # Summarize fields
            for sf in cfg.get("fields", []):
                field = sf.get("field", "")
                if field:
                    columns[field].add(f"Summarize:{tool.tool_id}")

        return {col: sorted(refs) for col, refs in sorted(columns.items())}

    def _analyze_container(self, container) -> dict:
        """Analyze a single container."""
        child_ids = getattr(container, 'child_tool_ids', [])
        if not child_ids:
            child_ids = getattr(container, 'child_tools', [])
            if child_ids and hasattr(child_ids[0], 'tool_id'):
                child_ids = [t.tool_id for t in child_ids]

        try:
            dep_resolver = DependencyResolver(self.workflow)
            exec_order = dep_resolver.resolve_execution_order(container.tool_id)
        except (ValueError, AttributeError):
            exec_order = list(child_ids)

        # Count external inputs
        ext_inputs = []
        if hasattr(self.workflow, 'get_container_external_inputs'):
            try:
                ext_inputs = self.workflow.get_container_external_inputs(container.tool_id)
            except (AttributeError, TypeError):
                ext_inputs = []
        elif hasattr(self.workflow, 'get_container_context'):
            try:
                ctx = self.workflow.get_container_context(container.tool_id)
                ext_inputs = ctx.get('external_inputs', [])
            except (AttributeError, TypeError):
                ext_inputs = []

        # Build a description of what this container does
        description = self._describe_container_pipeline(container, exec_order)

        return {
            "name": container.name,
            "tool_id": container.tool_id,
            "tool_count": len(container.child_tool_ids),
            "external_inputs": len(ext_inputs) if ext_inputs else 0,
            "execution_order": exec_order,
            "description": description,
        }

    def _describe_container_pipeline(
        self, container, exec_order: list[int]
    ) -> str:
        """Generate a natural language description of the pipeline."""
        steps = []
        for tid in exec_order:
            tool = self.workflow.get_tool(tid)
            if tool is None:
                continue
            tt = tool.tool_type
            ann = tool.annotation

            if tt in ("Browse", "Comment", "ExplorerBox", "Container"):
                continue

            if ann:
                steps.append(f"{tt}({ann})")
            else:
                cfg = self._get_tool_config(tool)
                if tt == "Filter":
                    expr = cfg.get("expression", "?")
                    steps.append(f"Filter({expr[:40]})")
                elif tt == "Join":
                    keys = cfg.get("left_keys", [])
                    steps.append(f"Join(on {','.join(keys[:2])})")
                elif tt == "Formula":
                    fields = [f.get("field", "") for f in cfg.get("formulas", [])[:2]]
                    steps.append(f"Formula({','.join(fields)})")
                elif tt == "Summarize":
                    groups = [
                        f.get("field", "")
                        for f in cfg.get("fields", [])
                        if f.get("action") == "GroupBy"
                    ]
                    steps.append(f"Summarize(by {','.join(groups[:2])})")
                else:
                    steps.append(tt)

        return " -> ".join(steps) if steps else "(empty)"

    def _generate_config(self, container) -> dict:
        """Auto-generate a YAML config for the given container."""
        config = {
            "container_name": container.name,
            "input_tables": {},
            "column_mappings": {},
            "output": {
                "table_name": f"TODO_catalog.schema.{container.name.lower().replace(' ', '_')}",
                "add_validation_cell": True,
                "add_schema_print": True,
            },
            "notebook": {
                "add_validation_cell": True,
                "add_schema_print": True,
            },
        }

        # Detect input tables for this container
        sources = self._detect_data_sources()
        for src in sources:
            key = src.get("suggested_key") or f"input_{src['tool_id']}"
            table = src.get("suggested_databricks_table") or "TODO_catalog.schema.table"
            config["input_tables"][key] = {
                "databricks_table": table,
                "maps_to_tool_id": src["tool_id"],
            }

        return config

    def _detect_warnings(self) -> list[str]:
        """Detect potential issues in the workflow."""
        warnings = []

        # Check for unsupported tool types
        supported = {
            "Filter", "Join", "Formula", "Select", "CrossTab",
            "Summarize", "Union", "Sort", "Unique", "Sample",
            "TextInput", "RecordID", "Transpose", "MultiRowFormula",
            "RegEx", "AppendFields", "FindReplace", "DynamicRename",
            "Browse", "Comment", "ExplorerBox", "Container",
            "DbInput", "DbOutput", "InputData", "OutputData",
            "LockInInput", "LockInWrite", "LockInFilter", "LockInJoin",
            "LockInFormula", "LockInSelect", "LockInSummarize",
            "LockInUnion", "LockInSort", "LockInUnique",
            "LockInSample", "LockInCrossTab",
            "AlteryxSelect", "MacroInput", "MacroOutput",
            "LockInStreamIn", "LockInStreamOut",
        }
        for tool in self._get_all_tools():
            if tool.tool_type not in supported:
                warnings.append(
                    f"Unsupported tool type '{tool.tool_type}' "
                    f"(Tool {tool.tool_id}). Manual conversion may be needed."
                )

        # Check for disconnected tools
        for container in self.workflow.containers:
            flow = self.flow_analyzer.get_container_flow(container.tool_id)
            for tid in container.child_tool_ids:
                tool = self.workflow.get_tool(tid)
                if tool is None:
                    continue
                if tool.tool_type in ("Browse", "Comment", "ExplorerBox"):
                    continue
                if tid in flow:
                    inputs = flow[tid]["inputs"]
                    outputs = flow[tid]["outputs"]
                    if not inputs and not outputs:
                        warnings.append(
                            f"Tool {tid} ({tool.tool_type}) in container "
                            f"'{container.name}' is disconnected."
                        )

        # Check for empty formulas
        for tool in self._get_all_tools():
            if tool.tool_type in ("Formula", "LockInFormula"):
                formulas = self._get_tool_config(tool).get("formulas", [])
                if not formulas:
                    warnings.append(
                        f"Tool {tool.tool_id} (Formula) has no formula fields."
                    )

        return warnings

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _suggest_databricks_table(self, table_name: str) -> str:
        """Suggest a Databricks table path from an Alteryx table reference."""
        # Clean up common patterns
        name = table_name.strip()
        # Remove file extensions
        name = re.sub(r'\.(csv|xlsx|yxdb|accdb|mdb)$', '', name, flags=re.IGNORECASE)
        # Extract just the table/file name
        name = name.replace("\\", "/").split("/")[-1]
        # Clean for Databricks
        name = re.sub(r'[^a-zA-Z0-9_]', '_', name).lower().strip('_')
        return f"TODO_catalog.TODO_schema.{name}" if name else "TODO_catalog.TODO_schema.TODO_table"

    def _suggest_config_key(self, table_name: str, annotation: str) -> str:
        """Suggest a short config key for a data source."""
        # Prefer annotation
        if annotation:
            key = annotation.lower()
            key = re.sub(r'[^a-z0-9]+', '_', key).strip('_')
            if key:
                return key[:30]

        # Fall back to table name
        name = table_name.replace("\\", "/").split("/")[-1]
        name = re.sub(r'\.(csv|xlsx|yxdb|accdb|mdb)$', '', name, flags=re.IGNORECASE)
        key = re.sub(r'[^a-z0-9]+', '_', name.lower()).strip('_')
        return key[:30] if key else "input"
