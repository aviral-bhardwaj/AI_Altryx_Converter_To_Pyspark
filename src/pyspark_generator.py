"""
Deterministic PySpark Code Generator
======================================
Given a parsed Alteryx workflow (WorkflowModel), generates optimized PySpark code
using topological sort and per-tool converters — no AI required.

Each Alteryx tool type has a dedicated converter class that emits correct PySpark
DataFrame transformations.

Output: A complete PySpark notebook (.py file) ready to run on Databricks.
"""

import datetime
import re
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from typing import Optional

from .models import Workflow, Tool, Connection, Container
from .expression_parser import convert_expression, convert_filter_expression


# ═══════════════════════════════════════════════════════════════════
# Topological Sort
# ═══════════════════════════════════════════════════════════════════

def topological_sort(tools: list, connections: list) -> list:
    """
    Return tool IDs in execution order using Kahn's algorithm.
    Tools with no dependencies come first.
    """
    tool_ids = {t.tool_id for t in tools}
    in_degree = defaultdict(int)
    adjacency = defaultdict(list)

    for conn in connections:
        if conn.origin_tool_id in tool_ids and conn.dest_tool_id in tool_ids:
            adjacency[conn.origin_tool_id].append(conn.dest_tool_id)
            in_degree[conn.dest_tool_id] += 1

    # Initialize queue with tools that have no incoming edges
    queue = deque()
    for t in tools:
        if in_degree[t.tool_id] == 0:
            queue.append(t.tool_id)

    ordered = []
    while queue:
        tid = queue.popleft()
        ordered.append(tid)
        for neighbor in adjacency[tid]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    # If some tools weren't reached (cycle), append them at the end
    visited = set(ordered)
    for t in tools:
        if t.tool_id not in visited:
            ordered.append(t.tool_id)

    return ordered


# ═══════════════════════════════════════════════════════════════════
# Tool Converter Base Class
# ═══════════════════════════════════════════════════════════════════

class ToolConverter(ABC):
    """Base class for all tool converters."""

    @abstractmethod
    def convert(self, tool: Tool, ctx: "GeneratorContext") -> list:
        """
        Convert a tool into PySpark code lines.

        Args:
            tool: The Alteryx tool to convert.
            ctx: Generator context with variable mappings and connections.

        Returns:
            List of code line strings.
        """
        ...


class GeneratorContext:
    """
    Holds state during code generation:
    - Variable names for each tool's output DataFrame(s)
    - Connection graph
    - Source table configuration
    """

    def __init__(
        self,
        workflow: Workflow,
        tools: list,
        connections: list,
        source_tables_config: Optional[dict] = None,
    ):
        self.workflow = workflow
        self.tools = {t.tool_id: t for t in tools}
        self.connections = connections
        self.source_tables_config = source_tables_config or {}

        # df_vars: tool_id -> { port_name: variable_name }
        # Default port is "Output"
        self.df_vars: dict = {}

        # Build connection lookups
        self._incoming: dict = defaultdict(list)
        self._outgoing: dict = defaultdict(list)
        for conn in connections:
            self._incoming[conn.dest_tool_id].append(conn)
            self._outgoing[conn.origin_tool_id].append(conn)

    def get_input_var(self, tool_id: int, port: str = "Input") -> str:
        """Get the DataFrame variable name feeding into a tool's input port."""
        for conn in self._incoming[tool_id]:
            if conn.dest_connection == port or port == "Input":
                origin_port = conn.origin_connection
                vars_map = self.df_vars.get(conn.origin_tool_id, {})
                # Try exact port, then "Output" default
                if origin_port in vars_map:
                    return vars_map[origin_port]
                if "Output" in vars_map:
                    return vars_map["Output"]
        return f"df_{tool_id}_input"

    def get_input_var_for_port(self, tool_id: int, port: str) -> str:
        """Get input variable for a specific named port (e.g., 'Left', 'Right')."""
        for conn in self._incoming[tool_id]:
            if conn.dest_connection == port:
                origin_port = conn.origin_connection
                vars_map = self.df_vars.get(conn.origin_tool_id, {})
                if origin_port in vars_map:
                    return vars_map[origin_port]
                if "Output" in vars_map:
                    return vars_map["Output"]
        return f"df_{tool_id}_{port.lower()}_input"

    def set_output_var(self, tool_id: int, var_name: str, port: str = "Output"):
        """Set the output DataFrame variable name for a tool's port."""
        if tool_id not in self.df_vars:
            self.df_vars[tool_id] = {}
        self.df_vars[tool_id][port] = var_name

    def get_output_var(self, tool_id: int, port: str = "Output") -> str:
        """Get the output variable name for a tool."""
        vars_map = self.df_vars.get(tool_id, {})
        return vars_map.get(port, vars_map.get("Output", f"df_{tool_id}"))

    def is_port_connected(self, tool_id: int, port: str) -> bool:
        """Check if a specific output port is connected downstream."""
        for conn in self._outgoing[tool_id]:
            if conn.origin_connection == port:
                return True
        return False

    def get_outgoing(self, tool_id: int) -> list:
        """Get all outgoing connections from a tool."""
        return self._outgoing[tool_id]

    def get_incoming(self, tool_id: int) -> list:
        """Get all incoming connections to a tool."""
        return self._incoming[tool_id]

    def make_var_name(self, tool: Tool) -> str:
        """Generate a meaningful variable name for a tool's output."""
        ann = (tool.annotation or "").strip()
        if ann and len(ann) > 2:
            cleaned = ann.lower()
            cleaned = re.sub(r"[^a-z0-9_]", "_", cleaned)
            cleaned = re.sub(r"_+", "_", cleaned).strip("_")
            if cleaned and len(cleaned) <= 40:
                return f"df_{cleaned}"

        pc = tool.parsed_config or {}
        tt = tool.tool_type

        if tt in ("InputData", "LockInStreamIn"):
            table = pc.get("table_name", "")
            if table:
                name = table.rsplit(".", 1)[-1].rsplit("/", 1)[-1]
                name = re.sub(r"[^a-z0-9_]", "_", name.lower()).strip("_")
                if name:
                    return f"df_{name}"

        if tt == "TextInput":
            return f"df_text_input_{tool.tool_id}"

        return f"df_{tool.tool_id}"


# ═══════════════════════════════════════════════════════════════════
# Tool Converters
# ═══════════════════════════════════════════════════════════════════

class InputDataConverter(ToolConverter):
    """Convert InputData tools to spark.read / spark.table calls."""

    def convert(self, tool: Tool, ctx: GeneratorContext) -> list:
        pc = tool.parsed_config or {}
        table_name = pc.get("table_name", "")
        var = ctx.make_var_name(tool)
        ctx.set_output_var(tool.tool_id, var)

        # Check source tables config for a mapping
        for key, value in ctx.source_tables_config.items():
            if (str(tool.tool_id) == str(key)
                    or (tool.annotation and key.lower() in tool.annotation.lower())
                    or (table_name and key.lower() in table_name.lower())):
                return [f'{var} = spark.table("{value}")  # Tool {tool.tool_id}: InputData']

        if not table_name:
            return [f'{var} = spark.table("TODO.unknown_table")  # Tool {tool.tool_id}: InputData - NEEDS CONFIG']

        # Detect file type from extension
        lower_table = table_name.lower()
        if lower_table.endswith(".csv"):
            return [f'{var} = spark.read.csv("{table_name}", header=True, inferSchema=True)  # Tool {tool.tool_id}']
        elif lower_table.endswith((".xlsx", ".xls")):
            return [
                f'{var} = spark.read.format("com.crealytics.spark.excel")'
                f'.option("header", "true").option("inferSchema", "true")'
                f'.load("{table_name}")  # Tool {tool.tool_id}'
            ]
        elif lower_table.endswith(".parquet"):
            return [f'{var} = spark.read.parquet("{table_name}")  # Tool {tool.tool_id}']
        elif "." in table_name and not table_name.startswith("/") and not table_name.startswith("\\"):
            # Looks like a catalog.schema.table reference
            return [f'{var} = spark.table("{table_name}")  # Tool {tool.tool_id}']
        else:
            # JDBC / other
            return [f'{var} = spark.table("TODO.{table_name}")  # Tool {tool.tool_id}: InputData']


class TextInputConverter(ToolConverter):
    """Convert TextInput tools to spark.createDataFrame()."""

    def convert(self, tool: Tool, ctx: GeneratorContext) -> list:
        var = ctx.make_var_name(tool)
        ctx.set_output_var(tool.tool_id, var)

        text_data = ctx.workflow.text_inputs.get(tool.tool_id, [])
        if not text_data:
            return [f'{var} = spark.createDataFrame([], schema="")  # Tool {tool.tool_id}: TextInput (empty)']

        headers = list(text_data[0].keys())
        lines = []
        lines.append(f"# Tool {tool.tool_id}: TextInput")
        lines.append(f"_data_{tool.tool_id} = [")
        for row in text_data:
            values = ", ".join(repr(row.get(h, "")) for h in headers)
            lines.append(f"    ({values}),")
        lines.append("]")
        col_list = ", ".join(repr(h) for h in headers)
        lines.append(f"{var} = spark.createDataFrame(_data_{tool.tool_id}, [{col_list}])")
        return lines


class OutputDataConverter(ToolConverter):
    """Convert OutputData tools to df.write statements."""

    def convert(self, tool: Tool, ctx: GeneratorContext) -> list:
        pc = tool.parsed_config or {}
        table_name = pc.get("table_name", "output_table")
        input_var = ctx.get_input_var(tool.tool_id)
        var = f"df_{tool.tool_id}"
        ctx.set_output_var(tool.tool_id, var)

        lines = [f"# Tool {tool.tool_id}: OutputData -> {table_name}"]
        lines.append(f"{var} = {input_var}")

        lower_table = table_name.lower()
        if lower_table.endswith(".csv"):
            lines.append(f'# {var}.write.csv("{table_name}", header=True, mode="overwrite")')
        else:
            lines.append(f'# {var}.write.format("delta").mode("overwrite").saveAsTable("{table_name}")')
        lines.append(f'{var}.createOrReplaceTempView("{table_name.rsplit(".", 1)[-1]}")')
        return lines


class FilterConverter(ToolConverter):
    """Convert Filter tools to df.filter()."""

    def convert(self, tool: Tool, ctx: GeneratorContext) -> list:
        pc = tool.parsed_config or {}
        expression = pc.get("expression", "")
        input_var = ctx.get_input_var(tool.tool_id)

        lines = []
        pyspark_expr = convert_filter_expression(expression) if expression else "F.lit(True)"

        # True output
        true_connected = ctx.is_port_connected(tool.tool_id, "True")
        false_connected = ctx.is_port_connected(tool.tool_id, "False")
        output_connected = ctx.is_port_connected(tool.tool_id, "Output")

        if true_connected or output_connected:
            true_var = f"df_{tool.tool_id}_true"
            lines.append(f"{true_var} = {input_var}.filter({pyspark_expr})  # Tool {tool.tool_id}: Filter (True)")
            ctx.set_output_var(tool.tool_id, true_var, "True")
            ctx.set_output_var(tool.tool_id, true_var, "Output")

        if false_connected:
            false_var = f"df_{tool.tool_id}_false"
            lines.append(f"{false_var} = {input_var}.filter(~({pyspark_expr}))  # Tool {tool.tool_id}: Filter (False)")
            ctx.set_output_var(tool.tool_id, false_var, "False")

        if not lines:
            # Neither port connected - emit True output by default
            var = f"df_{tool.tool_id}"
            lines.append(f"{var} = {input_var}.filter({pyspark_expr})  # Tool {tool.tool_id}: Filter")
            ctx.set_output_var(tool.tool_id, var)

        return lines


class FormulaConverter(ToolConverter):
    """Convert Formula tools to df.withColumn() chains."""

    def convert(self, tool: Tool, ctx: GeneratorContext) -> list:
        pc = tool.parsed_config or {}
        formulas = pc.get("formulas", [])
        input_var = ctx.get_input_var(tool.tool_id)
        var = ctx.make_var_name(tool)
        ctx.set_output_var(tool.tool_id, var)

        if not formulas:
            return [f"{var} = {input_var}  # Tool {tool.tool_id}: Formula (no formulas parsed)"]

        lines = [f"# Tool {tool.tool_id}: Formula"]
        lines.append(f"{var} = {input_var}")
        for f in formulas:
            field = f.get("field", "unknown")
            expr = f.get("expression", "")
            ftype = f.get("type", "")
            try:
                pyspark_expr = convert_expression(expr) if expr else 'F.lit(None)'
            except Exception:
                pyspark_expr = f'F.expr("{expr}")  # TODO: manual conversion needed'

            lines.append(f'{var} = {var}.withColumn("{field}", {pyspark_expr})')
            if ftype:
                type_map = {
                    "Int16": "short", "Int32": "int", "Int64": "long",
                    "Byte": "byte", "Float": "float", "Double": "double",
                    "FixedDecimal": "decimal(18,2)", "String": "string",
                    "V_String": "string", "V_WString": "string", "WString": "string",
                    "Bool": "boolean", "Date": "date", "DateTime": "timestamp",
                }
                spark_type = type_map.get(ftype, "")
                if spark_type:
                    lines.append(f'{var} = {var}.withColumn("{field}", F.col("{field}").cast("{spark_type}"))')
        return lines


class SelectConverter(ToolConverter):
    """Convert Select tools to df.select/drop/rename/cast operations."""

    def convert(self, tool: Tool, ctx: GeneratorContext) -> list:
        pc = tool.parsed_config or {}
        fields = pc.get("select_fields", [])
        input_var = ctx.get_input_var(tool.tool_id)
        var = ctx.make_var_name(tool)
        ctx.set_output_var(tool.tool_id, var)

        if not fields:
            return [f"{var} = {input_var}  # Tool {tool.tool_id}: Select (passthrough)"]

        lines = [f"# Tool {tool.tool_id}: Select"]
        lines.append(f"{var} = {input_var}")

        type_map = {
            "Int16": "short", "Int32": "int", "Int64": "long",
            "Byte": "byte", "Float": "float", "Double": "double",
            "FixedDecimal": "decimal(18,2)", "String": "string",
            "V_String": "string", "V_WString": "string", "WString": "string",
            "Bool": "boolean", "Date": "date", "DateTime": "timestamp",
        }

        drops = []
        for sf in fields:
            field = sf.get("field", "")
            selected = sf.get("selected", "True")
            rename = sf.get("rename", "")
            ftype = sf.get("type", "")

            if field.startswith("*"):
                continue  # Wildcard, skip

            if selected == "False":
                drops.append(field)
                continue

            if rename:
                lines.append(f'{var} = {var}.withColumnRenamed("{field}", "{rename}")')

            if ftype and ftype in type_map:
                col_name = rename or field
                lines.append(f'{var} = {var}.withColumn("{col_name}", F.col("{col_name}").cast("{type_map[ftype]}"))')

        if drops:
            drop_str = ", ".join(f'"{d}"' for d in drops)
            lines.append(f"{var} = {var}.drop({drop_str})")

        return lines


class JoinConverter(ToolConverter):
    """Convert Join tools to df.join()."""

    def convert(self, tool: Tool, ctx: GeneratorContext) -> list:
        pc = tool.parsed_config or {}
        left_keys = pc.get("left_keys", [])
        right_keys = pc.get("right_keys", [])
        input_left = ctx.get_input_var_for_port(tool.tool_id, "Left")
        input_right = ctx.get_input_var_for_port(tool.tool_id, "Right")

        lines = [f"# Tool {tool.tool_id}: Join"]

        # Determine which output ports are connected
        join_connected = ctx.is_port_connected(tool.tool_id, "Join")
        left_anti_connected = ctx.is_port_connected(tool.tool_id, "Left")
        right_anti_connected = ctx.is_port_connected(tool.tool_id, "Right")

        # Build join condition
        if left_keys == right_keys:
            # Same key names - use on=[] syntax to auto-deduplicate
            key_list = ", ".join(f'"{k}"' for k in left_keys)
            join_cond = f"[{key_list}]"
        else:
            # Different key names - use explicit condition
            conditions = []
            for lk, rk in zip(left_keys, right_keys):
                conditions.append(f'{input_left}["{lk}"] == {input_right}["{rk}"]')
            join_cond = " & ".join(f"({c})" for c in conditions) if len(conditions) > 1 else conditions[0] if conditions else "F.lit(True)"

        if join_connected:
            join_var = f"df_{tool.tool_id}_joined"
            lines.append(f'{join_var} = {input_left}.join({input_right}, {join_cond}, "inner")  # Join output')

            # Post-join column handling
            select_config = pc.get("select_config", [])
            for sf in select_config:
                field = sf.get("field", "")
                selected = sf.get("selected", "True")
                rename = sf.get("rename", "")
                if selected == "False" and field:
                    lines.append(f'{join_var} = {join_var}.drop("{field}")')
                elif rename and field:
                    lines.append(f'{join_var} = {join_var}.withColumnRenamed("{field}", "{rename}")')

            ctx.set_output_var(tool.tool_id, join_var, "Join")
            ctx.set_output_var(tool.tool_id, join_var, "Output")

        if left_anti_connected:
            left_var = f"df_{tool.tool_id}_left_unmatched"
            lines.append(f'{left_var} = {input_left}.join({input_right}, {join_cond}, "left_anti")  # Left unmatched')
            ctx.set_output_var(tool.tool_id, left_var, "Left")

        if right_anti_connected:
            right_var = f"df_{tool.tool_id}_right_unmatched"
            lines.append(f'{right_var} = {input_right}.join({input_left}, {join_cond}, "left_anti")  # Right unmatched')
            ctx.set_output_var(tool.tool_id, right_var, "Right")

        if not lines[1:]:
            # No ports connected - default to inner join
            join_var = f"df_{tool.tool_id}"
            lines.append(f'{join_var} = {input_left}.join({input_right}, {join_cond}, "inner")')
            ctx.set_output_var(tool.tool_id, join_var)

        return lines


class UnionConverter(ToolConverter):
    """Convert Union tools to unionByName."""

    def convert(self, tool: Tool, ctx: GeneratorContext) -> list:
        var = ctx.make_var_name(tool)
        ctx.set_output_var(tool.tool_id, var)

        # Collect all input variables
        incoming = ctx.get_incoming(tool.tool_id)
        input_vars = []
        for conn in incoming:
            origin_port = conn.origin_connection
            vars_map = ctx.df_vars.get(conn.origin_tool_id, {})
            v = vars_map.get(origin_port, vars_map.get("Output", f"df_{conn.origin_tool_id}"))
            input_vars.append(v)

        if len(input_vars) < 2:
            if input_vars:
                return [f"{var} = {input_vars[0]}  # Tool {tool.tool_id}: Union (single input)"]
            return [f"{var} = spark.createDataFrame([], schema='')  # Tool {tool.tool_id}: Union (no inputs)"]

        lines = [f"# Tool {tool.tool_id}: Union"]
        stream_list = ", ".join(input_vars)
        lines.append(f"_union_streams_{tool.tool_id} = [{stream_list}]")
        lines.append(f"from functools import reduce")
        lines.append(
            f"{var} = reduce(lambda a, b: a.unionByName(b, allowMissingColumns=True), "
            f"_union_streams_{tool.tool_id})"
        )
        return lines


class SummarizeConverter(ToolConverter):
    """Convert Summarize tools to groupBy().agg()."""

    def convert(self, tool: Tool, ctx: GeneratorContext) -> list:
        pc = tool.parsed_config or {}
        fields = pc.get("summarize_fields", [])
        input_var = ctx.get_input_var(tool.tool_id)
        var = ctx.make_var_name(tool)
        ctx.set_output_var(tool.tool_id, var)

        if not fields:
            return [f"{var} = {input_var}  # Tool {tool.tool_id}: Summarize (no config)"]

        group_cols = []
        agg_exprs = []

        action_map = {
            "GroupBy": None,
            "Sum": "F.sum",
            "Count": "F.count",
            "CountDistinct": "F.countDistinct",
            "Avg": "F.avg",
            "Min": "F.min",
            "Max": "F.max",
            "First": "F.first",
            "Last": "F.last",
            "CountNonNull": "F.count",
        }

        for sf in fields:
            field = sf.get("field", "")
            action = sf.get("action", "")
            rename = sf.get("rename", "")
            alias = rename or f"{action.lower()}_{field}"

            if action == "GroupBy":
                group_cols.append(f'"{field}"')
            elif action == "Concatenate":
                agg_exprs.append(f'F.concat_ws(", ", F.collect_list("{field}")).alias("{alias}")')
            elif action in action_map and action_map[action]:
                func = action_map[action]
                agg_exprs.append(f'{func}("{field}").alias("{alias}")')
            else:
                agg_exprs.append(f'F.first("{field}").alias("{alias}")')

        lines = [f"# Tool {tool.tool_id}: Summarize"]
        if group_cols:
            group_str = ", ".join(group_cols)
            agg_str = ",\n    ".join(agg_exprs)
            lines.append(f"{var} = {input_var}.groupBy({group_str}).agg(")
            lines.append(f"    {agg_str}")
            lines.append(")")
        elif agg_exprs:
            agg_str = ",\n    ".join(agg_exprs)
            lines.append(f"{var} = {input_var}.agg(")
            lines.append(f"    {agg_str}")
            lines.append(")")
        else:
            lines.append(f"{var} = {input_var}")

        return lines


class SortConverter(ToolConverter):
    """Convert Sort tools to df.orderBy()."""

    def convert(self, tool: Tool, ctx: GeneratorContext) -> list:
        pc = tool.parsed_config or {}
        sort_fields = pc.get("sort_fields", [])
        input_var = ctx.get_input_var(tool.tool_id)
        var = ctx.make_var_name(tool)
        ctx.set_output_var(tool.tool_id, var)

        if not sort_fields:
            return [f"{var} = {input_var}  # Tool {tool.tool_id}: Sort (no sort config)"]

        order_exprs = []
        for sf in sort_fields:
            field = sf.get("field", "")
            order = sf.get("order", "Ascending")
            if order.lower() == "descending":
                order_exprs.append(f'F.col("{field}").desc()')
            else:
                order_exprs.append(f'F.col("{field}").asc()')

        order_str = ", ".join(order_exprs)
        return [f"{var} = {input_var}.orderBy({order_str})  # Tool {tool.tool_id}: Sort"]


class UniqueConverter(ToolConverter):
    """Convert Unique tools to df.dropDuplicates()."""

    def convert(self, tool: Tool, ctx: GeneratorContext) -> list:
        pc = tool.parsed_config or {}
        unique_fields = pc.get("unique_fields", [])
        input_var = ctx.get_input_var(tool.tool_id)

        lines = [f"# Tool {tool.tool_id}: Unique"]

        unique_connected = ctx.is_port_connected(tool.tool_id, "Unique") or ctx.is_port_connected(tool.tool_id, "Output")
        dupes_connected = ctx.is_port_connected(tool.tool_id, "Dupes")

        if unique_fields:
            field_list = ", ".join(f'"{f}"' for f in unique_fields)
            subset = f"[{field_list}]"
        else:
            subset = ""

        if unique_connected:
            unique_var = f"df_{tool.tool_id}_unique"
            if subset:
                lines.append(f"{unique_var} = {input_var}.dropDuplicates({subset})")
            else:
                lines.append(f"{unique_var} = {input_var}.dropDuplicates()")
            ctx.set_output_var(tool.tool_id, unique_var, "Unique")
            ctx.set_output_var(tool.tool_id, unique_var, "Output")

        if dupes_connected:
            dupes_var = f"df_{tool.tool_id}_dupes"
            unique_ref = f"df_{tool.tool_id}_unique"
            lines.append(f"{dupes_var} = {input_var}.exceptAll({unique_ref})")
            ctx.set_output_var(tool.tool_id, dupes_var, "Dupes")

        if not unique_connected and not dupes_connected:
            var = f"df_{tool.tool_id}"
            if subset:
                lines.append(f"{var} = {input_var}.dropDuplicates({subset})")
            else:
                lines.append(f"{var} = {input_var}.dropDuplicates()")
            ctx.set_output_var(tool.tool_id, var)

        return lines


class SampleConverter(ToolConverter):
    """Convert Sample tools to df.limit()."""

    def convert(self, tool: Tool, ctx: GeneratorContext) -> list:
        pc = tool.parsed_config or {}
        n_records = pc.get("n_records", "100")
        input_var = ctx.get_input_var(tool.tool_id)
        var = ctx.make_var_name(tool)
        ctx.set_output_var(tool.tool_id, var)

        try:
            n = int(n_records)
        except (ValueError, TypeError):
            n = 100

        return [f"{var} = {input_var}.limit({n})  # Tool {tool.tool_id}: Sample"]


class CrossTabConverter(ToolConverter):
    """Convert CrossTab tools to groupBy().pivot().agg()."""

    def convert(self, tool: Tool, ctx: GeneratorContext) -> list:
        pc = tool.parsed_config or {}
        group_fields = pc.get("group_fields", "")
        header_field = pc.get("header_field", "")
        data_field = pc.get("data_field", "")
        method = pc.get("method", "Sum")
        input_var = ctx.get_input_var(tool.tool_id)
        var = ctx.make_var_name(tool)
        ctx.set_output_var(tool.tool_id, var)

        method_map = {
            "Sum": "F.sum", "Count": "F.count", "Avg": "F.avg",
            "Min": "F.min", "Max": "F.max",
        }
        agg_func = method_map.get(method, "F.sum")

        group_list = ", ".join(f'"{g.strip()}"' for g in group_fields.split(",") if g.strip())

        lines = [f"# Tool {tool.tool_id}: CrossTab"]
        lines.append(
            f'{var} = {input_var}.groupBy({group_list})'
            f'.pivot("{header_field}")'
            f'.agg({agg_func}("{data_field}"))'
        )
        return lines


class TransposeConverter(ToolConverter):
    """Convert Transpose tools to stack() / unpivot."""

    def convert(self, tool: Tool, ctx: GeneratorContext) -> list:
        pc = tool.parsed_config or {}
        key_fields = pc.get("key_fields", [])
        data_fields = pc.get("data_fields", [])
        input_var = ctx.get_input_var(tool.tool_id)
        var = ctx.make_var_name(tool)
        ctx.set_output_var(tool.tool_id, var)

        if not data_fields:
            return [f"{var} = {input_var}  # Tool {tool.tool_id}: Transpose (no data fields)"]

        key_cols = ", ".join(f'"{k}"' for k in key_fields)
        n = len(data_fields)
        col_pairs = ", ".join(
            f"'{f}', cast(`{f}` as string)" for f in data_fields
        )
        stack_expr = f"stack({n}, {col_pairs}) as (Name, Value)"

        lines = [f"# Tool {tool.tool_id}: Transpose"]
        lines.append(f'{var} = {input_var}.select({key_cols}, F.expr("{stack_expr}"))')
        return lines


class MultiRowFormulaConverter(ToolConverter):
    """Convert MultiRowFormula tools to Window functions."""

    def convert(self, tool: Tool, ctx: GeneratorContext) -> list:
        pc = tool.parsed_config or {}
        formulas = pc.get("formulas", [])
        num_rows = pc.get("num_rows", "1")
        input_var = ctx.get_input_var(tool.tool_id)
        var = ctx.make_var_name(tool)
        ctx.set_output_var(tool.tool_id, var)

        lines = [f"# Tool {tool.tool_id}: MultiRowFormula"]
        lines.append(f"_w_{tool.tool_id} = Window.orderBy(F.monotonically_increasing_id())")
        lines.append(f"{var} = {input_var}")

        try:
            offset = int(num_rows)
        except (ValueError, TypeError):
            offset = 1

        for f in formulas:
            field = f.get("field", "unknown")
            expr = f.get("expression", "")
            # Detect common patterns
            if "Row-1" in expr or "row-1" in expr.lower():
                lines.append(f'{var} = {var}.withColumn("{field}", F.lag("{field}", {offset}).over(_w_{tool.tool_id}))')
            elif "Row+1" in expr or "row+1" in expr.lower():
                lines.append(f'{var} = {var}.withColumn("{field}", F.lead("{field}", {offset}).over(_w_{tool.tool_id}))')
            else:
                try:
                    pyspark_expr = convert_expression(expr) if expr else 'F.lit(None)'
                except Exception:
                    pyspark_expr = f'F.lit(None)  # TODO: "{expr}"'
                lines.append(f'{var} = {var}.withColumn("{field}", {pyspark_expr})')

        return lines


class RegExConverter(ToolConverter):
    """Convert RegEx tools to regexp_extract/replace/rlike."""

    def convert(self, tool: Tool, ctx: GeneratorContext) -> list:
        pc = tool.parsed_config or {}
        expression = pc.get("expression", "")
        field = pc.get("field", "")
        output_method = pc.get("output_method", "")
        input_var = ctx.get_input_var(tool.tool_id)
        var = ctx.make_var_name(tool)
        ctx.set_output_var(tool.tool_id, var)

        lines = [f"# Tool {tool.tool_id}: RegEx"]

        if "replace" in output_method.lower():
            lines.append(f'{var} = {input_var}.withColumn("{field}", F.regexp_replace(F.col("{field}"), r"{expression}", ""))')
        elif "match" in output_method.lower():
            lines.append(f'{var} = {input_var}.filter(F.col("{field}").rlike(r"{expression}"))')
        elif "parse" in output_method.lower() or "tokenize" in output_method.lower():
            lines.append(f'{var} = {input_var}.withColumn("{field}_parsed", F.regexp_extract(F.col("{field}"), r"{expression}", 1))')
        else:
            lines.append(f'{var} = {input_var}.withColumn("{field}_regex", F.regexp_extract(F.col("{field}"), r"{expression}", 0))')

        return lines


class RecordIDConverter(ToolConverter):
    """Convert RecordID tools to monotonically_increasing_id()."""

    def convert(self, tool: Tool, ctx: GeneratorContext) -> list:
        pc = tool.parsed_config or {}
        field_name = pc.get("field_name", "RecordID")
        start_value = pc.get("start_value", "1")
        input_var = ctx.get_input_var(tool.tool_id)
        var = ctx.make_var_name(tool)
        ctx.set_output_var(tool.tool_id, var)

        try:
            start = int(start_value)
        except (ValueError, TypeError):
            start = 1

        return [
            f'{var} = {input_var}.withColumn("{field_name}", '
            f'F.monotonically_increasing_id() + {start})  # Tool {tool.tool_id}: RecordID'
        ]


class AppendFieldsConverter(ToolConverter):
    """Convert AppendFields tools to crossJoin."""

    def convert(self, tool: Tool, ctx: GeneratorContext) -> list:
        input_target = ctx.get_input_var_for_port(tool.tool_id, "Target")
        input_source = ctx.get_input_var_for_port(tool.tool_id, "Source")

        # If we can't distinguish Target/Source, use Left/Right or generic
        if input_target == f"df_{tool.tool_id}_target_input":
            input_target = ctx.get_input_var_for_port(tool.tool_id, "Input")
        if input_source == f"df_{tool.tool_id}_source_input":
            # Get second incoming connection
            incoming = ctx.get_incoming(tool.tool_id)
            if len(incoming) >= 2:
                origin_id = incoming[1].origin_tool_id
                vars_map = ctx.df_vars.get(origin_id, {})
                input_source = vars_map.get("Output", f"df_{origin_id}")

        var = ctx.make_var_name(tool)
        ctx.set_output_var(tool.tool_id, var)

        return [
            f"# Tool {tool.tool_id}: AppendFields (CrossJoin)",
            f"{var} = {input_target}.crossJoin({input_source})"
        ]


class RunningTotalConverter(ToolConverter):
    """Convert RunningTotal tools to window-based running sum."""

    def convert(self, tool: Tool, ctx: GeneratorContext) -> list:
        pc = tool.parsed_config or {}
        input_var = ctx.get_input_var(tool.tool_id)
        var = ctx.make_var_name(tool)
        ctx.set_output_var(tool.tool_id, var)

        lines = [f"# Tool {tool.tool_id}: RunningTotal"]
        lines.append(f"_w_rt_{tool.tool_id} = Window.orderBy(F.monotonically_increasing_id()).rowsBetween(Window.unboundedPreceding, Window.currentRow)")
        lines.append(f"{var} = {input_var}")
        lines.append(f'# TODO: Add running total columns: {var} = {var}.withColumn("running_total", F.sum("value_col").over(_w_rt_{tool.tool_id}))')
        return lines


class BrowseConverter(ToolConverter):
    """Browse tools are inspection-only; pass through."""

    def convert(self, tool: Tool, ctx: GeneratorContext) -> list:
        input_var = ctx.get_input_var(tool.tool_id)
        var = f"df_{tool.tool_id}"
        ctx.set_output_var(tool.tool_id, var)
        return [f"{var} = {input_var}  # Tool {tool.tool_id}: Browse (passthrough)"]


class TextToColumnsConverter(ToolConverter):
    """Convert TextToColumns to F.split / explode."""

    def convert(self, tool: Tool, ctx: GeneratorContext) -> list:
        pc = tool.parsed_config or {}
        input_var = ctx.get_input_var(tool.tool_id)
        var = ctx.make_var_name(tool)
        ctx.set_output_var(tool.tool_id, var)

        lines = [f"# Tool {tool.tool_id}: TextToColumns"]
        lines.append(f'{var} = {input_var}  # TODO: configure split field and delimiter')
        return lines


class DateTimeConverter(ToolConverter):
    """Convert DateTime tools."""

    def convert(self, tool: Tool, ctx: GeneratorContext) -> list:
        input_var = ctx.get_input_var(tool.tool_id)
        var = ctx.make_var_name(tool)
        ctx.set_output_var(tool.tool_id, var)
        return [f"{var} = {input_var}  # Tool {tool.tool_id}: DateTime - TODO: configure format conversion"]


class DynamicRenameConverter(ToolConverter):
    """Convert DynamicRename tools."""

    def convert(self, tool: Tool, ctx: GeneratorContext) -> list:
        input_var = ctx.get_input_var(tool.tool_id)
        var = ctx.make_var_name(tool)
        ctx.set_output_var(tool.tool_id, var)

        lines = [f"# Tool {tool.tool_id}: DynamicRename"]
        lines.append(f"{var} = {input_var}")
        lines.append(f"# TODO: Apply dynamic rename rules from configuration")
        return lines


class GenerateRowsConverter(ToolConverter):
    """Convert GenerateRows tools."""

    def convert(self, tool: Tool, ctx: GeneratorContext) -> list:
        var = ctx.make_var_name(tool)
        ctx.set_output_var(tool.tool_id, var)
        return [
            f"# Tool {tool.tool_id}: GenerateRows",
            f"# TODO: configure row generation logic",
            f'{var} = spark.range(0, 100).toDF("RowCount")  # placeholder',
        ]


class MultiFieldFormulaConverter(ToolConverter):
    """Convert MultiFieldFormula tools."""

    def convert(self, tool: Tool, ctx: GeneratorContext) -> list:
        input_var = ctx.get_input_var(tool.tool_id)
        var = ctx.make_var_name(tool)
        ctx.set_output_var(tool.tool_id, var)
        return [
            f"# Tool {tool.tool_id}: MultiFieldFormula",
            f"{var} = {input_var}  # TODO: apply formula across multiple fields",
        ]


class FindReplaceConverter(ToolConverter):
    """Convert FindReplace tools."""

    def convert(self, tool: Tool, ctx: GeneratorContext) -> list:
        input_var = ctx.get_input_var(tool.tool_id)
        var = ctx.make_var_name(tool)
        ctx.set_output_var(tool.tool_id, var)
        return [
            f"# Tool {tool.tool_id}: FindReplace",
            f"{var} = {input_var}  # TODO: configure find/replace pairs",
        ]


class PassthroughConverter(ToolConverter):
    """Fallback converter for unknown or no-op tools."""

    def convert(self, tool: Tool, ctx: GeneratorContext) -> list:
        input_var = ctx.get_input_var(tool.tool_id)
        var = f"df_{tool.tool_id}"
        ctx.set_output_var(tool.tool_id, var)
        return [f"{var} = {input_var}  # Tool {tool.tool_id}: {tool.tool_type} (passthrough)"]


# ═══════════════════════════════════════════════════════════════════
# Converter Registry
# ═══════════════════════════════════════════════════════════════════

CONVERTER_REGISTRY: dict = {
    "InputData": InputDataConverter(),
    "LockInStreamIn": InputDataConverter(),
    "DynamicInput": InputDataConverter(),
    "TextInput": TextInputConverter(),
    "OutputData": OutputDataConverter(),
    "LockInStreamOut": OutputDataConverter(),
    "Filter": FilterConverter(),
    "LockInFilter": FilterConverter(),
    "Formula": FormulaConverter(),
    "LockInFormula": FormulaConverter(),
    "Select": SelectConverter(),
    "LockInSelect": SelectConverter(),
    "Join": JoinConverter(),
    "LockInJoin": JoinConverter(),
    "Union": UnionConverter(),
    "Summarize": SummarizeConverter(),
    "Sort": SortConverter(),
    "Unique": UniqueConverter(),
    "Sample": SampleConverter(),
    "CrossTab": CrossTabConverter(),
    "Transpose": TransposeConverter(),
    "MultiRowFormula": MultiRowFormulaConverter(),
    "RegEx": RegExConverter(),
    "RecordID": RecordIDConverter(),
    "AppendFields": AppendFieldsConverter(),
    "TextToColumns": TextToColumnsConverter(),
    "DateTime": DateTimeConverter(),
    "DynamicRename": DynamicRenameConverter(),
    "GenerateRows": GenerateRowsConverter(),
    "MultiFieldFormula": MultiFieldFormulaConverter(),
    "FindReplace": FindReplaceConverter(),
    "Browse": BrowseConverter(),
    "BrowseV2": BrowseConverter(),
    "Comment": PassthroughConverter(),
    "BlockUntilDone": PassthroughConverter(),
    "RunCommand": PassthroughConverter(),
}


def get_converter(tool_type: str) -> ToolConverter:
    """Get the converter for a tool type, falling back to passthrough."""
    return CONVERTER_REGISTRY.get(tool_type, PassthroughConverter())


# ═══════════════════════════════════════════════════════════════════
# Main Code Generator
# ═══════════════════════════════════════════════════════════════════

class PySparkCodeGenerator:
    """
    Deterministic PySpark code generator.

    Takes a parsed Alteryx workflow and generates a complete PySpark
    notebook (.py file) in Databricks format.
    """

    def __init__(self, source_tables_config: Optional[dict] = None):
        self.source_tables_config = source_tables_config or {}

    def generate(
        self,
        workflow: Workflow,
        workflow_name: str = "workflow",
        context: Optional[dict] = None,
    ) -> str:
        """
        Generate a complete PySpark Databricks notebook.

        Args:
            workflow: Parsed Alteryx Workflow object.
            workflow_name: Name for the output notebook.
            context: Optional pre-built context dict (from workflow.get_unified_context()).

        Returns:
            Complete PySpark notebook as a string.
        """
        if context is None:
            context = workflow.get_unified_context()

        tools = context.get("tools", [])
        connections = context.get("internal_connections", [])

        # Build generator context
        ctx = GeneratorContext(
            workflow=workflow,
            tools=tools,
            connections=connections,
            source_tables_config=self.source_tables_config,
        )

        # Topological sort for execution order
        ordered_ids = topological_sort(tools, connections)
        tool_map = {t.tool_id: t for t in tools}

        # Classify tools
        code_sections = {
            "imports": [],
            "sources": [],
            "transformations": [],
            "outputs": [],
        }

        for tid in ordered_ids:
            tool = tool_map.get(tid)
            if tool is None:
                continue

            # Skip container and comment tools
            if tool.tool_type in ("Container", "Comment"):
                continue

            converter = get_converter(tool.tool_type)
            lines = converter.convert(tool, ctx)

            if tool.tool_type in ("InputData", "LockInStreamIn", "DynamicInput", "TextInput"):
                code_sections["sources"].extend(lines)
            elif tool.tool_type in ("OutputData", "LockInStreamOut"):
                code_sections["outputs"].extend(lines)
            else:
                code_sections["transformations"].extend(lines)

        # Count complexity metrics
        num_tools = len([t for t in tools if t.tool_type not in ("Container", "Comment")])
        num_joins = len([t for t in tools if t.tool_type in ("Join", "LockInJoin")])
        num_formulas = len([t for t in tools if t.tool_type in ("Formula", "LockInFormula")])
        complexity = "Low" if num_tools < 10 else ("Medium" if num_tools < 30 else "High")

        # Build the notebook
        return self._assemble_notebook(
            workflow_name=workflow_name,
            code_sections=code_sections,
            complexity=complexity,
            num_tools=num_tools,
            num_joins=num_joins,
            num_formulas=num_formulas,
        )

    def _assemble_notebook(
        self,
        workflow_name: str,
        code_sections: dict,
        complexity: str,
        num_tools: int,
        num_joins: int,
        num_formulas: int,
    ) -> str:
        """Assemble code sections into a Databricks notebook."""
        parts = []
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Header
        parts.append("# Databricks notebook source")
        parts.append("")
        parts.append("# COMMAND ----------")
        parts.append("")
        parts.append("# MAGIC %md")
        parts.append(f"# MAGIC # {workflow_name}")
        parts.append(f"# MAGIC Auto-generated by PySpark Code Generator (deterministic mode)")
        parts.append(f"# MAGIC Source: {workflow_name}.yxmd")
        parts.append(f"# MAGIC Generated: {timestamp}")
        parts.append(f"# MAGIC Complexity: {complexity} ({num_tools} tools, {num_joins} joins, {num_formulas} formulas)")

        # Imports
        parts.append("")
        parts.append("# COMMAND ----------")
        parts.append("")
        parts.append("from pyspark.sql import functions as F")
        parts.append("from pyspark.sql.types import *")
        parts.append("from pyspark.sql.window import Window")

        # Source tables
        if code_sections["sources"]:
            parts.append("")
            parts.append("# COMMAND ----------")
            parts.append("")
            parts.append("# MAGIC %md")
            parts.append("# MAGIC ## Input Sources")
            parts.append("")
            parts.append("# COMMAND ----------")
            parts.append("")
            parts.extend(code_sections["sources"])

        # Transformations
        if code_sections["transformations"]:
            parts.append("")
            parts.append("# COMMAND ----------")
            parts.append("")
            parts.append("# MAGIC %md")
            parts.append("# MAGIC ## Transformations")
            parts.append("")
            parts.append("# COMMAND ----------")
            parts.append("")
            parts.extend(code_sections["transformations"])

        # Outputs
        if code_sections["outputs"]:
            parts.append("")
            parts.append("# COMMAND ----------")
            parts.append("")
            parts.append("# MAGIC %md")
            parts.append("# MAGIC ## Output")
            parts.append("")
            parts.append("# COMMAND ----------")
            parts.append("")
            parts.extend(code_sections["outputs"])

        # Validation section
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
