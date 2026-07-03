"""
Built-in Tool Converters
=========================
One plugin class per Alteryx tool family. Each converter emits PySpark
DataFrame code lines for its tool and registers itself with the tool
registry via ``@register("ToolType", ...)``.

To add support for a new Alteryx tool:

1. Subclass ``ToolConverter`` here (or in a new module under ``src/tools/``).
2. Decorate it with ``@register("YourToolType")``.
3. Optionally declare it in ``config/tool_mapping.yaml`` to override or
   extend the built-in mapping without touching this file.
"""

import re

from ..expression_parser import convert_expression, convert_filter_expression
from ..models import Tool
from .base import GeneratorContext, ToolConverter
from .registry import register


def pystr(value: str) -> str:
    """Emit a safe double-quoted Python string literal (escapes \\ and \")."""
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _identifier(value: str) -> str:
    """Sanitize an arbitrary string into a snake_case identifier fragment."""
    cleaned = re.sub(r"[^a-z0-9_]+", "_", value.lower()).strip("_")
    return cleaned or "unknown"


FILE_EXTENSIONS = {"csv", "xlsx", "xls", "parquet", "yxdb", "txt", "json", "avro", "orc"}


def _table_leaf_name(table_name: str) -> str:
    """
    Extract the logical table/stem name from an output target:
    ``catalog.schema.table`` → ``table``; ``path/output.csv`` → ``output``.
    """
    leaf = table_name.replace("\\", "/").rsplit("/", 1)[-1]
    if "." in leaf:
        stem, ext = leaf.rsplit(".", 1)
        if ext.lower() in FILE_EXTENSIONS:
            # File target: use the stem's last dotted segment (drop the extension).
            return stem.rsplit(".", 1)[-1]
        # Catalog reference: the last dotted segment IS the table name.
        return ext
    return leaf

# Alteryx field type -> Spark SQL type used for casts.
SPARK_TYPE_MAP = {
    "Int16": "short", "Int32": "int", "Int64": "long",
    "Byte": "byte", "Float": "float", "Double": "double",
    "FixedDecimal": "decimal(18,2)", "String": "string",
    "V_String": "string", "V_WString": "string", "WString": "string",
    "Bool": "boolean", "Date": "date", "DateTime": "timestamp",
}


@register("InputData", "LockInStreamIn", "DynamicInput")
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

        # Alteryx sheet suffix, e.g. "book.xlsx|||`Sheet1$`"
        sheet = ""
        if "|||" in table_name:
            table_name, sheet = table_name.split("|||", 1)

        lower_table = table_name.lower()
        fwd_path = table_name.replace("\\", "/")
        is_local_path = bool(re.match(r"^[a-z]:[\\/]", lower_table)) or "\\" in table_name

        # Local Alteryx sources (.yxdb or Windows paths without a readable
        # format) cannot be read by Spark — emit a mapping TODO instead of
        # broken code, preserving the original path for the migration team.
        if lower_table.endswith(".yxdb") or (
            is_local_path and not lower_table.endswith((".csv", ".xlsx", ".xls", ".parquet"))
        ):
            stem = _identifier(fwd_path.rsplit("/", 1)[-1].rsplit(".", 1)[0])
            return [
                f"# Tool {tool.tool_id}: InputData — local Alteryx source, needs a Unity Catalog mapping",
                f"# TODO: map via source-tables config. Original source: {fwd_path}",
                f'{var} = spark.table("TODO.{stem}")',
            ]

        if lower_table.endswith(".csv"):
            return [f'{var} = spark.read.csv({pystr(fwd_path)}, header=True, inferSchema=True)  # Tool {tool.tool_id}']
        elif lower_table.endswith((".xlsx", ".xls")):
            sheet_comment = f"  # sheet: {sheet}" if sheet else ""
            return [
                f'{var} = spark.read.format("com.crealytics.spark.excel")'
                f'.option("header", "true").option("inferSchema", "true")'
                f'.load({pystr(fwd_path)})  # Tool {tool.tool_id}{sheet_comment}'
            ]
        elif lower_table.endswith(".parquet"):
            return [f'{var} = spark.read.parquet({pystr(fwd_path)})  # Tool {tool.tool_id}']
        elif "." in table_name and not table_name.startswith("/") and not is_local_path:
            # Looks like a catalog.schema.table reference
            return [f'{var} = spark.table("{table_name}")  # Tool {tool.tool_id}']
        else:
            # JDBC / other
            return [f'{var} = spark.table("TODO.{_identifier(fwd_path)}")  # Tool {tool.tool_id}: InputData']


@register("TextInput")
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


@register("OutputData", "LockInStreamOut")
class OutputDataConverter(ToolConverter):
    """Convert OutputData tools to df.write statements."""

    def convert(self, tool: Tool, ctx: GeneratorContext) -> list:
        pc = tool.parsed_config or {}
        table_name = pc.get("table_name", "output_table")
        input_var = ctx.get_input_var(tool.tool_id)
        var = f"df_{tool.tool_id}"
        ctx.set_output_var(tool.tool_id, var)

        lower_table = table_name.lower()
        is_file_target = ("/" in table_name or "\\" in table_name
                          or lower_table.rsplit(".", 1)[-1] in FILE_EXTENSIONS)

        # Qualify bare table names with the configured Unity Catalog target
        # (catalog.schema.table) so generated writes are workspace-portable.
        if (not is_file_target and table_name.count(".") < 2
                and ctx.target_catalog and ctx.target_schema):
            table_name = (f"{ctx.target_catalog}.{ctx.target_schema}."
                          f"{_identifier(table_name)}")

        lines = [f"# Tool {tool.tool_id}: OutputData -> {table_name}"]
        lines.append(f"{var} = {input_var}")

        if lower_table.endswith(".csv"):
            lines.append(f'# {var}.write.csv({pystr(table_name)}, header=True, mode="overwrite")')
        else:
            lines.append(f'# {var}.write.format("delta").mode("overwrite").saveAsTable({pystr(table_name)})')
        view_name = _identifier(_table_leaf_name(table_name))
        lines.append(f'{var}.createOrReplaceTempView("{view_name}")')
        return lines


@register("Filter", "LockInFilter")
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


@register("Formula", "LockInFormula")
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
                spark_type = SPARK_TYPE_MAP.get(ftype, "")
                if spark_type:
                    lines.append(f'{var} = {var}.withColumn("{field}", F.col("{field}").cast("{spark_type}"))')
        return lines


@register("Select", "LockInSelect")
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

            if ftype and ftype in SPARK_TYPE_MAP:
                col_name = rename or field
                lines.append(f'{var} = {var}.withColumn("{col_name}", F.col("{col_name}").cast("{SPARK_TYPE_MAP[ftype]}"))')

        if drops:
            drop_str = ", ".join(f'"{d}"' for d in drops)
            lines.append(f"{var} = {var}.drop({drop_str})")

        return lines


@register("Join", "LockInJoin")
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


@register("Union")
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
        lines.append("from functools import reduce")
        lines.append(
            f"{var} = reduce(lambda a, b: a.unionByName(b, allowMissingColumns=True), "
            f"_union_streams_{tool.tool_id})"
        )
        return lines


@register("Summarize")
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

        order_sensitive = False
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
                if action in ("First", "Last"):
                    order_sensitive = True
            else:
                agg_exprs.append(f'F.first("{field}").alias("{alias}")')
                order_sensitive = True

        lines = [f"# Tool {tool.tool_id}: Summarize"]
        if order_sensitive:
            # Unlike Alteryx's in-order desktop engine, Spark gives no row-order
            # guarantee after a shuffle, so First/Last can vary between runs.
            lines.append("# TODO: First/Last aggregations are order-dependent — Spark does not")
            lines.append("#       guarantee row order after a shuffle. Add an explicit ordering")
            lines.append("#       (e.g. Window + row_number over a sort key) for deterministic results.")
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


@register("Sort")
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


@register("Unique")
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


@register("Sample")
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


@register("CrossTab")
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


@register("Transpose")
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


@register("MultiRowFormula")
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
        lines.append("# TODO: monotonically_increasing_id() only approximates the original row")
        lines.append("#       order and is unreliable after shuffles — replace with a real sort")
        lines.append("#       key (e.g. an upstream RecordID column) for order-dependent logic.")
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


@register("RegEx")
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


@register("RecordID")
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


@register("AppendFields")
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


@register("RunningTotal")
class RunningTotalConverter(ToolConverter):
    """Convert RunningTotal tools to window-based running sum."""

    def convert(self, tool: Tool, ctx: GeneratorContext) -> list:
        input_var = ctx.get_input_var(tool.tool_id)
        var = ctx.make_var_name(tool)
        ctx.set_output_var(tool.tool_id, var)

        lines = [f"# Tool {tool.tool_id}: RunningTotal"]
        lines.append(f"_w_rt_{tool.tool_id} = Window.orderBy(F.monotonically_increasing_id()).rowsBetween(Window.unboundedPreceding, Window.currentRow)")
        lines.append(f"{var} = {input_var}")
        lines.append(f'# TODO: Add running total columns: {var} = {var}.withColumn("running_total", F.sum("value_col").over(_w_rt_{tool.tool_id}))')
        return lines


@register("Browse", "BrowseV2")
class BrowseConverter(ToolConverter):
    """Browse tools are inspection-only; pass through."""

    def convert(self, tool: Tool, ctx: GeneratorContext) -> list:
        input_var = ctx.get_input_var(tool.tool_id)
        var = f"df_{tool.tool_id}"
        ctx.set_output_var(tool.tool_id, var)
        return [f"{var} = {input_var}  # Tool {tool.tool_id}: Browse (passthrough)"]


@register("TextToColumns")
class TextToColumnsConverter(ToolConverter):
    """Convert TextToColumns to F.split / explode."""

    def convert(self, tool: Tool, ctx: GeneratorContext) -> list:
        input_var = ctx.get_input_var(tool.tool_id)
        var = ctx.make_var_name(tool)
        ctx.set_output_var(tool.tool_id, var)

        lines = [f"# Tool {tool.tool_id}: TextToColumns"]
        lines.append(f'{var} = {input_var}  # TODO: configure split field and delimiter')
        return lines


@register("DateTime")
class DateTimeConverter(ToolConverter):
    """Convert DateTime tools."""

    def convert(self, tool: Tool, ctx: GeneratorContext) -> list:
        input_var = ctx.get_input_var(tool.tool_id)
        var = ctx.make_var_name(tool)
        ctx.set_output_var(tool.tool_id, var)
        return [f"{var} = {input_var}  # Tool {tool.tool_id}: DateTime - TODO: configure format conversion"]


@register("DynamicRename")
class DynamicRenameConverter(ToolConverter):
    """Convert DynamicRename tools."""

    def convert(self, tool: Tool, ctx: GeneratorContext) -> list:
        input_var = ctx.get_input_var(tool.tool_id)
        var = ctx.make_var_name(tool)
        ctx.set_output_var(tool.tool_id, var)

        lines = [f"# Tool {tool.tool_id}: DynamicRename"]
        lines.append(f"{var} = {input_var}")
        lines.append("# TODO: Apply dynamic rename rules from configuration")
        return lines


@register("GenerateRows")
class GenerateRowsConverter(ToolConverter):
    """Convert GenerateRows tools."""

    def convert(self, tool: Tool, ctx: GeneratorContext) -> list:
        var = ctx.make_var_name(tool)
        ctx.set_output_var(tool.tool_id, var)
        return [
            f"# Tool {tool.tool_id}: GenerateRows",
            "# TODO: configure row generation logic",
            f'{var} = spark.range(0, 100).toDF("RowCount")  # placeholder',
        ]


@register("MultiFieldFormula")
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


@register("FindReplace")
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


@register("Comment", "BlockUntilDone", "RunCommand")
class PassthroughConverter(ToolConverter):
    """Fallback converter for unknown or no-op tools."""

    def convert(self, tool: Tool, ctx: GeneratorContext) -> list:
        input_var = ctx.get_input_var(tool.tool_id)
        var = f"df_{tool.tool_id}"
        ctx.set_output_var(tool.tool_id, var)
        return [f"{var} = {input_var}  # Tool {tool.tool_id}: {tool.tool_type} (passthrough)"]
