"""Generate PySpark code for individual tools."""

import logging
from typing import Optional

from ...phase1_parser.models.workflow import Workflow
from ...phase1_parser.models.tool import Tool
from ..config.column_mapping import ColumnMapper
from ..expression_converter.alteryx_to_pyspark import AlteryxExpressionConverter
from ..tool_converters import CONVERTER_MAP, BaseToolConverter
from .flow_analyzer import FlowAnalyzer
from .semantic_namer import SemanticNamer

logger = logging.getLogger(__name__)


class CodeGenerator:
    """
    Generate PySpark code for tools using the appropriate converter.

    Manages DataFrame naming and coordinates between tool converters.
    Uses SemanticNamer for intelligent, context-aware variable names
    instead of generic names like df_join_42.
    """

    def __init__(
        self,
        workflow: Workflow,
        column_mapper: ColumnMapper,
        expression_converter: AlteryxExpressionConverter,
        input_table_map: dict[int, str] = None,
        pre_filters: dict[str, str] = None,
    ):
        """
        Args:
            workflow: Parsed workflow.
            column_mapper: Column name mapper.
            expression_converter: Expression converter.
            input_table_map: Maps external input tool_id -> Databricks table name.
            pre_filters: Maps table config key -> SQL pre-filter expression.
        """
        self.workflow = workflow
        self.column_mapper = column_mapper
        self.expr_converter = expression_converter
        self.input_table_map = input_table_map or {}
        self.pre_filters = pre_filters or {}
        self.flow_analyzer = FlowAnalyzer(workflow)

        # Track DataFrame names: tool_id -> {connection_name: df_var_name}
        self.df_names: dict[int, dict[str, str]] = {}

        # Semantic namer for intelligent variable names
        self.namer = SemanticNamer(workflow, self.flow_analyzer)

        # Track the last generated DataFrame name (for final output)
        self._last_output_df: str | None = None

    def generate_tool_code(
        self,
        tool: Tool,
        container_id: int,
    ) -> str:
        """
        Generate PySpark code for a single tool.

        Args:
            tool: The tool to generate code for.
            container_id: The container context.

        Returns:
            PySpark code string.
        """
        tool_type = tool.tool_type

        # Skip non-functional tools
        if tool_type in ("Browse", "Comment", "ExplorerBox", "Container"):
            self._register_passthrough(tool, container_id)
            return f"# Skipping {tool_type} tool {tool.tool_id}\n"

        # Get input DataFrame names
        input_dfs = self._resolve_input_dfs(tool, container_id)

        # Get intelligent output DataFrame name via semantic namer
        output_prefix = self.namer.resolve_name(
            tool=tool,
            container_id=container_id,
            upstream_names=input_dfs,
        )

        # Get converter
        converter_class = CONVERTER_MAP.get(tool_type)

        if converter_class is None:
            logger.warning(
                "No converter for tool type: %s (tool %d)",
                tool_type, tool.tool_id,
            )
            code = (
                f"# WARNING: No converter for {tool_type} "
                f"(Tool ID: {tool.tool_id})\n"
            )
            if tool.annotation:
                code += f"# Annotation: {tool.annotation}\n"
            # Pass through
            main_input = input_dfs.get("Input", input_dfs.get("Left", ""))
            if main_input:
                code += f"{output_prefix} = {main_input}\n"
                self._register_output(tool, output_prefix)
            return code

        # Create converter instance
        converter = converter_class(
            tool_config=tool.to_dict(),
            column_mapper=self.column_mapper,
            expression_converter=self.expr_converter,
        )

        # Generate code based on tool type
        code = self._dispatch_converter(
            converter, tool, input_dfs, output_prefix, container_id
        )

        return code

    def generate_external_input_code(
        self,
        tool_id: int,
        table_key: str,
        table_name: str,
    ) -> str:
        """
        Generate code to load an external input table.

        Args:
            tool_id: The tool ID receiving the external input.
            table_key: Config key for the table.
            table_name: Databricks table name.

        Returns:
            PySpark code to load the table.
        """
        df_name = self.namer.resolve_input_source_name(
            table_key=table_key,
            table_name=table_name,
            tool_id=tool_id,
        )

        code = f'# Load input: {table_key}\n'
        code += f'{df_name} = spark.table("{table_name}")\n'

        # Apply pre-filter if specified
        pre_filter = self.pre_filters.get(table_key, "")
        if pre_filter:
            code += f'# Pre-filter\n'
            code += f'{df_name} = {df_name}.filter("""{pre_filter}""")\n'

        # Register the DataFrame name for downstream tools
        self.df_names[tool_id] = {"Output": df_name, "Input": df_name}

        return code

    def get_df_name(self, tool_id: int, connection: str = "Output") -> Optional[str]:
        """Get the DataFrame variable name for a tool's output."""
        if tool_id in self.df_names:
            return self.df_names[tool_id].get(
                connection,
                self.df_names[tool_id].get("Output"),
            )
        return None

    def get_last_output_df(self) -> str:
        """Return the last DataFrame variable that was generated.

        This is used by the notebook generator to write the final output
        instead of assuming a hardcoded ``df_result``.
        """
        return self._last_output_df or "df_result"

    def _resolve_input_dfs(
        self, tool: Tool, container_id: int
    ) -> dict[str, str]:
        """Resolve input DataFrame names for a tool."""
        result: dict[str, str] = {}

        flow = self.flow_analyzer.get_container_flow(container_id)
        if tool.tool_id not in flow:
            return result

        for edge in flow[tool.tool_id]["inputs"]:
            from_id = edge.from_tool_id
            from_conn = edge.from_connection
            to_conn = edge.to_connection

            if from_id in self.df_names:
                # Map output connection names to DataFrame names
                conn_map = {
                    "Output": "Output",
                    "True": "True",
                    "False": "False",
                    "Join": "Join",
                    "Left": "Left",
                    "Right": "Right",
                    "Unique": "Unique",
                    "Dupes": "Dupes",
                }
                source_key = conn_map.get(from_conn, "Output")

                if source_key in self.df_names[from_id]:
                    result[to_conn] = self.df_names[from_id][source_key]
                elif "Output" in self.df_names[from_id]:
                    result[to_conn] = self.df_names[from_id]["Output"]

        return result

    def _dispatch_converter(
        self,
        converter: BaseToolConverter,
        tool: Tool,
        input_dfs: dict[str, str],
        output_prefix: str,
        container_id: int,
    ) -> str:
        """Dispatch to the appropriate converter method."""
        tool_type = tool.tool_type

        # Get primary input
        primary_input = (
            input_dfs.get("Input")
            or input_dfs.get("Left")
            or input_dfs.get("Output")
            or f"df_unknown_{tool.tool_id}"
        )

        if tool_type in ("Join", "LockInJoin"):
            right_input = input_dfs.get("Right", f"df_unknown_right_{tool.tool_id}")
            code = converter.generate_code(
                input_df_name=primary_input,
                output_df_name=output_prefix,
                right_df_name=right_input,
            )
            # Register all three outputs with descriptive suffixes
            self.df_names[tool.tool_id] = {
                "Join": f"{output_prefix}_matched",
                "Left": f"{output_prefix}_left_unmatched",
                "Right": f"{output_prefix}_right_unmatched",
                "Output": f"{output_prefix}_matched",  # Default to inner join
            }
            self._last_output_df = f"{output_prefix}_matched"

        elif tool_type in ("Filter", "LockInFilter"):
            code = converter.generate_code(
                input_df_name=primary_input,
                output_df_name=output_prefix,
            )
            self.df_names[tool.tool_id] = {
                "True": f"{output_prefix}_true",
                "False": f"{output_prefix}_false",
                "Output": f"{output_prefix}_true",  # Default to true branch
            }
            self._last_output_df = f"{output_prefix}_true"

        elif tool_type in ("Unique", "LockInUnique"):
            code = converter.generate_code(
                input_df_name=primary_input,
                output_df_name=output_prefix,
            )
            self.df_names[tool.tool_id] = {
                "Unique": f"{output_prefix}_unique",
                "Dupes": f"{output_prefix}_duplicates",
                "Output": f"{output_prefix}_unique",
            }
            self._last_output_df = f"{output_prefix}_unique"

        elif tool_type in ("Union", "LockInUnion"):
            # Gather all input DataFrames
            additional = [
                v for k, v in sorted(input_dfs.items())
                if k != "Input" and k != next(iter(input_dfs), None)
            ]
            if not additional and len(input_dfs) > 1:
                all_inputs = list(input_dfs.values())
                primary_input = all_inputs[0]
                additional = all_inputs[1:]

            code = converter.generate_code(
                input_df_name=primary_input,
                output_df_name=output_prefix,
                additional_inputs=additional,
            )
            self._register_output(tool, output_prefix)

        elif tool_type == "TextInput":
            code = converter.generate_code(
                input_df_name="",
                output_df_name=output_prefix,
            )
            self._register_output(tool, output_prefix)

        elif tool_type == "LockInInput":
            table_name = self.input_table_map.get(tool.tool_id, "")
            code = converter.generate_code(
                input_df_name="",
                output_df_name=output_prefix,
                table_name=table_name,
            )
            self._register_output(tool, output_prefix)

        elif tool_type == "LockInWrite":
            code = converter.generate_code(
                input_df_name=primary_input,
                output_df_name=output_prefix,
            )
            self._register_output(tool, output_prefix)

        else:
            # Standard single-input, single-output tool
            code = converter.generate_code(
                input_df_name=primary_input,
                output_df_name=output_prefix,
            )
            self._register_output(tool, output_prefix)

        return code

    def _register_output(self, tool: Tool, df_name: str) -> None:
        """Register a simple tool output DataFrame name."""
        self.df_names[tool.tool_id] = {"Output": df_name}
        self._last_output_df = df_name

    def _register_passthrough(self, tool: Tool, container_id: int) -> None:
        """For skip tools, pass through the input DataFrame name."""
        input_dfs = self._resolve_input_dfs(tool, container_id)
        primary = (
            input_dfs.get("Input")
            or input_dfs.get("Left")
            or next(iter(input_dfs.values()), None)
        )
        if primary:
            self.df_names[tool.tool_id] = {"Output": primary}
