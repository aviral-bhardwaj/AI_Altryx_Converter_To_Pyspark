"""Main notebook generator - orchestrates Phase 2 code generation."""

import os
import logging
from datetime import datetime, timezone
from typing import Optional

from ...phase1_parser.models.workflow import Workflow
from ...phase1_parser.output.json_writer import JSONWriter
from ..config.output_config import GeneratorConfig
from ..config.column_mapping import ColumnMapper
from ..config.schema_config import SchemaConfig
from ..expression_converter.alteryx_to_pyspark import AlteryxExpressionConverter
from .flow_analyzer import FlowAnalyzer
from .dependency_resolver import DependencyResolver
from .code_generator import CodeGenerator

logger = logging.getLogger(__name__)

# Databricks notebook cell separator
CELL_SEP = "\n# COMMAND ----------\n\n"


class NotebookGenerator:
    """
    Generate Databricks notebooks from intermediate JSON and user config.

    This is the main orchestrator for Phase 2.
    """

    VERSION = "1.0.0"

    def __init__(
        self,
        intermediate_json_path: str,
        config_path: str = "",
        config_dict: dict = None,
    ):
        """
        Args:
            intermediate_json_path: Path to the intermediate JSON from Phase 1.
            config_path: Path to the user configuration YAML file.
            config_dict: Alternative: config as a dictionary.
        """
        # Load intermediate representation
        self.workflow = JSONWriter.load(intermediate_json_path)

        # Load configuration
        if config_path:
            self.config = GeneratorConfig(config_path=config_path)
        elif config_dict:
            self.config = GeneratorConfig(config_dict=config_dict)
        else:
            raise ValueError("Either config_path or config_dict must be provided")

        # Initialize components
        self.column_mapper = ColumnMapper(self.config.column_mappings)
        self.schema_config = SchemaConfig()
        self.expr_converter = AlteryxExpressionConverter(self.column_mapper)
        self.flow_analyzer = FlowAnalyzer(self.workflow)
        self.dep_resolver = DependencyResolver(self.workflow)

    def generate(self) -> str:
        """
        Generate the complete Databricks notebook.

        Steps:
        1. Find the target container
        2. Resolve tool execution order
        3. Generate code for each tool in order
        4. Assemble into notebook format

        Returns:
            Complete notebook content as a string.
        """
        # Find the target container
        container = self._find_container()
        if not container:
            raise ValueError(
                f"Container '{self.config.container_name}' not found in workflow"
            )

        logger.info(
            "Generating notebook for container: %s (ID: %d)",
            container.name, container.tool_id,
        )

        # Resolve execution order
        execution_order = self.dep_resolver.resolve_execution_order(
            container.tool_id
        )
        logger.info("Execution order: %s", execution_order)

        # Set up code generator
        tool_id_map = self.config.get_tool_id_mapping()
        table_names = self.config.get_input_table_names()
        # Build input_table_map: tool_id -> table_name
        input_table_map = {}
        for tool_id, config_key in tool_id_map.items():
            if config_key in table_names:
                input_table_map[tool_id] = table_names[config_key]

        code_gen = CodeGenerator(
            workflow=self.workflow,
            column_mapper=self.column_mapper,
            expression_converter=self.expr_converter,
            input_table_map=input_table_map,
            pre_filters=self.config.get_pre_filters(),
        )

        # Generate cells
        cells = []

        # 1. Header cell (markdown)
        cells.append(self._generate_header_cell(container))

        # 2. Configuration cell (markdown)
        cells.append(self._generate_config_markdown_cell())

        # 3. Configuration cell (code)
        cells.append(self._generate_config_cell())

        # 4. Imports cell
        cells.append(self._generate_imports_cell())

        # 5. Load external inputs
        ext_input_code = self._generate_external_inputs(
            code_gen, container.tool_id
        )
        if ext_input_code:
            cells.append(ext_input_code)

        # 6. Main transformation code
        main_code = self._generate_main_flow(
            code_gen, execution_order, container.tool_id
        )
        cells.append(main_code)

        # 7. Write output cell
        if self.config.output.table_name:
            cells.append(self._generate_write_output_cell(code_gen))

        # 8. Validation cell (optional)
        if self.config.output.add_validation_cell:
            cells.append(self._generate_validation_cell())

        # Assemble notebook
        notebook = self._assemble_notebook(cells)

        logger.info("Generated notebook: %d cells", len(cells))
        return notebook

    def generate_and_save(self, output_path: str = "") -> str:
        """
        Generate notebook and save to file.

        Args:
            output_path: Override output path (uses config if empty).

        Returns:
            Path to the saved notebook.
        """
        notebook_content = self.generate()

        path = output_path or self.config.output.notebook_path
        if not path:
            path = f"./output/{self.config.container_name}.py"

        # Ensure directory exists
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

        with open(path, "w", encoding="utf-8") as f:
            f.write(notebook_content)

        logger.info("Saved notebook to: %s", path)
        return path

    def _find_container(self):
        """Find the target container by name or ID."""
        name = self.config.container_name

        # Try by name first
        container = self.workflow.get_container_by_name(name)
        if container:
            return container

        # Try by ID
        try:
            container_id = int(name)
            return self.workflow.get_container(container_id)
        except (ValueError, TypeError):
            pass

        # Fuzzy match
        for c in self.workflow.containers:
            if name.lower() in c.name.lower():
                logger.info(
                    "Fuzzy matched container '%s' to '%s'", name, c.name
                )
                return c

        return None

    def _generate_header_cell(self, container) -> str:
        """Generate the markdown header cell."""
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

        # Build input table rows
        table_rows = ""
        for key, cfg in self.config.input_tables.items():
            table_name = cfg.get("databricks_table", key)
            table_rows += f"# MAGIC | {key} | {table_name} |\n"

        return (
            f"# Databricks notebook source\n"
            f"# MAGIC %md\n"
            f"# MAGIC # {container.name} - Auto-Generated from Alteryx\n"
            f"# MAGIC \n"
            f"# MAGIC **Source Workflow:** {self.workflow.name}.yxmd\n"
            f"# MAGIC **Container:** {container.name}\n"
            f"# MAGIC **Generated:** {now}\n"
            f"# MAGIC **Generator Version:** {self.VERSION}\n"
            f"# MAGIC \n"
            f"# MAGIC ---\n"
            f"# MAGIC \n"
            f"# MAGIC ## Data Sources\n"
            f"# MAGIC | Input | Databricks Table |\n"
            f"# MAGIC |-------|------------------|\n"
            f"{table_rows}"
        )

    def _generate_config_markdown_cell(self) -> str:
        """Generate configuration markdown cell."""
        return (
            f"# MAGIC %md\n"
            f"# MAGIC ## Configuration\n"
        )

    def _generate_config_cell(self) -> str:
        """Generate the configuration code cell."""
        lines = [
            "# =============================================================================",
            "# AUTO-GENERATED CONFIGURATION - DO NOT EDIT ABOVE THIS LINE",
            "# =============================================================================",
            "",
        ]

        # Source tables
        lines.append("# Source Tables")
        for key, cfg in self.config.input_tables.items():
            table_name = cfg.get("databricks_table", key)
            var_name = key.upper() + "_TABLE"
            lines.append(f'{var_name} = "{table_name}"')

        lines.append("")

        # Output table
        if self.config.output.table_name:
            lines.append("# Output Table")
            lines.append(f'OUTPUT_TABLE = "{self.config.output.table_name}"')
            lines.append("")

        # Expected row count
        if self.config.output.expected_row_count:
            lines.append("# Expected row count for validation")
            lines.append(
                f"EXPECTED_ROW_COUNT = {self.config.output.expected_row_count}"
            )
            lines.append("")

        return "\n".join(lines)

    def _generate_imports_cell(self) -> str:
        """Generate the imports cell."""
        return (
            "from pyspark.sql import DataFrame\n"
            "from pyspark.sql import functions as F\n"
            "from pyspark.sql.types import (\n"
            "    StructType, StructField, StringType, IntegerType,\n"
            "    LongType, DoubleType, FloatType, BooleanType,\n"
            "    DateType, TimestampType, DecimalType, ShortType, ByteType\n"
            ")\n"
            "from pyspark.sql.window import Window\n"
        )

    def _generate_external_inputs(
        self, code_gen: CodeGenerator, container_id: int
    ) -> str:
        """Generate code to load external input tables."""
        ext_inputs = self.workflow.get_container_external_inputs(container_id)
        tool_id_map = self.config.get_tool_id_mapping()
        table_names = self.config.get_input_table_names()

        if not ext_inputs:
            return ""

        lines = [
            "# =============================================================================",
            "# LOAD INPUT DATA",
            "# =============================================================================",
            "",
        ]

        # Track which tools have already been loaded
        loaded_tools: set[int] = set()

        for conn in ext_inputs:
            dest_tool_id = conn.destination_tool_id

            # Find the config key for this external input
            # The origin tool is what maps to a table
            origin_id = conn.origin_tool_id
            config_key = tool_id_map.get(origin_id)

            if config_key and config_key in table_names and origin_id not in loaded_tools:
                table_name = table_names[config_key]
                input_code = code_gen.generate_external_input_code(
                    tool_id=origin_id,
                    table_key=config_key,
                    table_name=table_name,
                )
                lines.append(input_code)
                loaded_tools.add(origin_id)

                # Also register the destination tool with the same df
                df_name = code_gen.get_df_name(origin_id, "Output")
                if df_name and dest_tool_id not in code_gen.df_names:
                    code_gen.df_names[dest_tool_id] = {"Input": df_name, "Output": df_name}

        return "\n".join(lines)

    def _generate_main_flow(
        self,
        code_gen: CodeGenerator,
        execution_order: list[int],
        container_id: int,
    ) -> str:
        """Generate the main transformation code."""
        lines = [
            "# =============================================================================",
            "# TRANSFORMATIONS",
            "# =============================================================================",
            "",
        ]

        for tool_id in execution_order:
            tool = self.workflow.get_tool(tool_id)
            if tool is None:
                lines.append(f"# WARNING: Tool {tool_id} not found\n")
                continue

            code = code_gen.generate_tool_code(tool, container_id)
            if code:
                lines.append(code)

        return "\n".join(lines)

    def _generate_write_output_cell(self, code_gen: CodeGenerator) -> str:
        """Generate code to write the final output."""
        table_name = self.config.output.table_name
        output_cols = self.config.output.columns

        lines = [
            "# =============================================================================",
            "# WRITE OUTPUT",
            "# =============================================================================",
            "",
        ]

        # Use the last tool's actual output DataFrame name
        final_df = code_gen.get_last_output_df()
        lines.append(f"# Final output from: {final_df}")

        if output_cols:
            cols_str = ", ".join(f'"{c}"' for c in output_cols)
            lines.append(f"df_output = {final_df}.select({cols_str})")
        else:
            lines.append(f"df_output = {final_df}")

        lines.append("")
        lines.append(f'df_output.write.mode("overwrite").saveAsTable("{table_name}")')

        return "\n".join(lines)

    def _generate_validation_cell(self) -> str:
        """Generate validation code cell."""
        lines = [
            "# =============================================================================",
            "# VALIDATION",
            "# =============================================================================",
            "",
        ]

        if self.config.output.table_name:
            lines.append(
                f'df_validate = spark.table("{self.config.output.table_name}")'
            )
            lines.append('print(f"Output row count: {df_validate.count()}")')

            if self.config.output.expected_row_count:
                lines.append(
                    f'expected = {self.config.output.expected_row_count}'
                )
                lines.append('actual = df_validate.count()')
                lines.append(
                    'assert actual == expected, '
                    'f"Row count mismatch: expected {expected}, got {actual}"'
                )
                lines.append('print(f"Row count validation passed: {actual}")')

        if self.config.output.add_schema_print:
            lines.append("")
            lines.append("# Print schema")
            lines.append("df_validate.printSchema()")
            lines.append("")
            lines.append("# Show sample")
            lines.append("df_validate.show(5, truncate=False)")

        return "\n".join(lines)

    def _assemble_notebook(self, cells: list[str]) -> str:
        """Assemble cells into a complete Databricks notebook."""
        return CELL_SEP.join(cells) + "\n"
