"""
Context Builder
================
Transforms parsed Alteryx workflow data into a structured prompt context
that Claude AI can use to generate accurate PySpark code.
"""

from typing import Optional
from .models import Workflow, Container, Tool, Connection


def _format_tool_config(tool: Tool) -> str:
    """Format a tool's parsed configuration as clear, structured text."""
    pc = tool.parsed_config
    if not pc:
        return "    (no parsed configuration)"

    lines = []
    tool_type = tool.tool_type

    if tool_type in ("Filter", "LockInFilter"):
        lines.append(f"    Filter Expression: {pc.get('expression', '(none)')}")
        lines.append(f"    Mode: {pc.get('mode', 'Custom')}")
        lines.append("    Outputs: True (rows matching condition), False (rows NOT matching)")

    elif tool_type in ("Formula", "LockInFormula"):
        formulas = pc.get("formulas", [])
        if formulas:
            lines.append("    Formula Fields:")
            for f in formulas:
                field = f.get("field", "?")
                expr = f.get("expression", "?")
                ftype = f.get("type", "")
                type_str = f" (type={ftype})" if ftype else ""
                lines.append(f"      [{field}]{type_str} = {expr}")

    elif tool_type in ("Join", "LockInJoin"):
        left_keys = pc.get("left_keys", [])
        right_keys = pc.get("right_keys", [])
        lines.append("    Join Keys:")
        for i in range(max(len(left_keys), len(right_keys))):
            lk = left_keys[i] if i < len(left_keys) else "?"
            rk = right_keys[i] if i < len(right_keys) else "?"
            lines.append(f"      Left.[{lk}] = Right.[{rk}]")
        lines.append("    Outputs:")
        lines.append("      'Join' output = INNER join result (matched rows)")
        lines.append("      'Left' output = LEFT-ONLY unmatched rows")
        lines.append("      'Right' output = RIGHT-ONLY unmatched rows")
        select_config = pc.get("select_config", [])
        if select_config:
            lines.append("    Post-Join Column Handling:")
            for sf in select_config:
                field = sf.get("field", "")
                selected = sf.get("selected", "True")
                rename = sf.get("rename", "")
                if selected == "False":
                    lines.append(f"      DROP: {field}")
                elif rename:
                    lines.append(f"      RENAME: {field} -> {rename}")
                else:
                    lines.append(f"      KEEP: {field}")

    elif tool_type in ("Select", "LockInSelect"):
        fields = pc.get("select_fields", [])
        if fields:
            lines.append("    Column Configuration:")
            for sf in fields:
                field = sf.get("field", "")
                selected = sf.get("selected", "True")
                rename = sf.get("rename", "")
                if field.startswith("*"):
                    lines.append(f"      {field}: selected={selected}")
                elif selected == "False":
                    lines.append(f"      DROP: {field}")
                elif rename:
                    lines.append(f"      RENAME: {field} -> {rename}")
                else:
                    lines.append(f"      KEEP: {field}")

    elif tool_type == "Summarize":
        fields = pc.get("summarize_fields", [])
        if fields:
            lines.append("    Aggregation Fields:")
            for sf in fields:
                field = sf.get("field", "")
                action = sf.get("action", "")
                rename = sf.get("rename", "")
                rename_str = f" AS {rename}" if rename else ""
                lines.append(f"      {action}({field}){rename_str}")

    elif tool_type == "CrossTab":
        lines.append(f"    Group By: {pc.get('group_fields', '')}")
        lines.append(f"    Pivot Header: {pc.get('header_field', '')}")
        lines.append(f"    Value Field: {pc.get('data_field', '')}")
        lines.append(f"    Aggregation: {pc.get('method', '')}")

    elif tool_type == "Sort":
        fields = pc.get("sort_fields", [])
        if fields:
            lines.append("    Sort Order:")
            for sf in fields:
                lines.append(f"      {sf.get('field', '')} {sf.get('order', 'Ascending')}")

    elif tool_type == "Unique":
        fields = pc.get("unique_fields", [])
        if fields:
            lines.append(f"    Deduplicate On: {', '.join(fields)}")
            lines.append("    Outputs: 'Unique' (first occurrence), 'Dupes' (duplicates)")

    elif tool_type == "Union":
        lines.append(f"    Mode: {pc.get('mode', 'Auto')}")
        lines.append(f"    By Name: {pc.get('by_name', 'True')}")

    elif tool_type == "Sample":
        lines.append(f"    N Records: {pc.get('n_records', '')}")
        lines.append(f"    Mode: {pc.get('mode', '')}")
        if pc.get("group_fields"):
            lines.append(f"    Group Fields: {pc.get('group_fields')}")

    elif tool_type in ("InputData", "LockInStreamIn"):
        table = pc.get("table_name", "")
        if table:
            lines.append(f"    Source Table/File: {table}")
        fields = pc.get("fields", [])
        if fields:
            lines.append("    Columns:")
            for f in fields:
                lines.append(f"      {f.get('name', '')} ({f.get('type', '')})")

    elif tool_type in ("OutputData", "LockInStreamOut"):
        table = pc.get("table_name", "")
        if table:
            lines.append(f"    Target Table/File: {table}")

    elif tool_type == "MultiRowFormula":
        formulas = pc.get("formulas", [])
        if formulas:
            lines.append("    Formula Fields:")
            for f in formulas:
                lines.append(f"      [{f.get('field', '')}] = {f.get('expression', '')}")
        lines.append(f"    Num Rows: {pc.get('num_rows', '')}")

    elif tool_type == "RegEx":
        lines.append(f"    Field: {pc.get('field', '')}")
        lines.append(f"    Expression: {pc.get('expression', '')}")
        lines.append(f"    Output Method: {pc.get('output_method', '')}")

    elif tool_type == "Transpose":
        lines.append(f"    Key Fields: {', '.join(pc.get('key_fields', []))}")
        lines.append(f"    Data Fields: {', '.join(pc.get('data_fields', []))}")

    elif tool_type == "RecordID":
        lines.append(f"    Field Name: {pc.get('field_name', '')}")
        lines.append(f"    Start Value: {pc.get('start_value', '1')}")

    # Fallback for unknown types - show raw config keys
    if not lines and pc:
        for k, v in pc.items():
            lines.append(f"    {k}: {v}")

    return "\n".join(lines) if lines else "    (no configuration details)"


def build_container_prompt(
    container: Container,
    context: dict,
    workflow: Workflow,
    source_tables_config: Optional[dict] = None,
) -> str:
    """
    Build a detailed prompt describing a container's full logic for Claude AI.

    Returns a structured text description including:
    - Container name and purpose
    - All source tables with full details
    - All tools with structured configurations
    - Data flow (connections) in execution order
    - External inputs and outputs
    - Inline data (TextInput tools)
    """
    parts = []

    # ── Header ───────────────────────────────────────────────────────
    parts.append(f"# Container: {container.name}")
    parts.append(f"# Container ToolID: {container.tool_id}")
    parts.append("")

    # ── Source tables config (user-provided mapping) ──────────────────
    if source_tables_config:
        parts.append("## Source Table Mapping (Alteryx Input -> Databricks Table):")
        for key, value in source_tables_config.items():
            parts.append(f"  '{key}' -> spark.table(\"{value}\")")
        parts.append("")

    # ── External inputs with FULL details ─────────────────────────────
    external_inputs = context.get("external_inputs", [])
    source_tools = context.get("source_tools", {})
    if external_inputs:
        parts.append("## SOURCE DATA (external inputs flowing INTO this container):")
        parts.append("These are DataFrames that this container receives from outside.")
        parts.append("You MUST define these as spark.table() reads at the top of your code.")
        parts.append("")
        for conn in external_inputs:
            src_info = source_tools.get(conn.origin_tool_id, {})
            src_type = src_info.get("type", "Unknown")
            src_annotation = src_info.get("annotation", "")
            src_container = src_info.get("container", "Root")
            src_parsed = src_info.get("parsed_config", {})

            parts.append(f"  ### External Source: Tool {conn.origin_tool_id} ({src_type})")
            if src_annotation:
                parts.append(f"      Annotation: {src_annotation}")
            parts.append(f"      From container: '{src_container}'")
            parts.append(f"      Connects to: Tool {conn.dest_tool_id} via [{conn.dest_connection}] input")

            # Show table name if available
            if src_parsed:
                table_name = src_parsed.get("table_name", "")
                if table_name:
                    parts.append(f"      Original Table/File: {table_name}")
                # Show source table fields
                fields = src_parsed.get("fields", [])
                if fields:
                    parts.append(f"      Columns ({len(fields)}):")
                    for f in fields:
                        parts.append(f"        - {f.get('name', '')} ({f.get('type', '')})")

            # Check source_tables_config for a mapping
            if source_tables_config:
                for key, value in source_tables_config.items():
                    if (str(conn.origin_tool_id) == str(key)
                            or (src_annotation and key.lower() in src_annotation.lower())
                            or (src_parsed.get("table_name", "") and key.lower() in src_parsed.get("table_name", "").lower())):
                        parts.append(f"      -> USE: spark.table(\"{value}\")")
                        break

            # Show TextInput data if the source is a TextInput
            text_data = src_info.get("text_input_data")
            if text_data:
                parts.append(f"      Inline Data ({len(text_data)} rows):")
                if text_data:
                    headers = list(text_data[0].keys())
                    parts.append(f"        Columns: {headers}")
                    for row in text_data[:10]:
                        parts.append(f"        {row}")
                    if len(text_data) > 10:
                        parts.append(f"        ... ({len(text_data) - 10} more rows)")

            parts.append("")
        parts.append("")

    # ── Sub-containers ────────────────────────────────────────────────
    sub_containers = context.get("sub_containers", [])
    if sub_containers:
        parts.append("## Sub-Containers (nested groups within this container):")
        for sc in sub_containers:
            parts.append(f"  - '{sc.name}' (ToolID={sc.tool_id})")
        parts.append("")

    # ── Data Flow (connections) ───────────────────────────────────────
    internal_connections = context.get("internal_connections", [])
    if internal_connections:
        parts.append("## DATA FLOW (how data moves between tools inside this container):")
        parts.append("Read these connections carefully to determine the execution order.")
        parts.append("The origin_connection name tells you WHICH output of a tool to use.")
        parts.append("")
        for conn in internal_connections:
            origin_tool = workflow.get_tool(conn.origin_tool_id)
            dest_tool = workflow.get_tool(conn.dest_tool_id)
            origin_desc = f"{origin_tool.tool_type}" if origin_tool else "?"
            dest_desc = f"{dest_tool.tool_type}" if dest_tool else "?"
            parts.append(
                f"  Tool {conn.origin_tool_id} ({origin_desc})"
                f" --[{conn.origin_connection}]--> "
                f"Tool {conn.dest_tool_id} ({dest_desc}) [{conn.dest_connection}]"
            )
        parts.append("")

    # ── External outputs ──────────────────────────────────────────────
    external_outputs = context.get("external_outputs", [])
    if external_outputs:
        parts.append("## OUTPUTS (data flowing OUT of this container to downstream):")
        for conn in external_outputs:
            dest_tool = workflow.get_tool(conn.dest_tool_id)
            dest_desc = ""
            if dest_tool:
                dest_container = workflow._find_tool_container_name(dest_tool.tool_id)
                dest_desc = (
                    f"Tool {conn.dest_tool_id} ({dest_tool.tool_type}, "
                    f"'{dest_tool.annotation}', container='{dest_container}')"
                )
            else:
                dest_desc = f"Tool {conn.dest_tool_id}"
            origin_tool = workflow.get_tool(conn.origin_tool_id)
            origin_desc = f"{origin_tool.tool_type}" if origin_tool else "?"
            parts.append(
                f"  Tool {conn.origin_tool_id} ({origin_desc})"
                f" --[{conn.origin_connection}]--> {dest_desc}"
            )
        parts.append("  Create a temp view or write statement for each final output DataFrame.")
        parts.append("")

    # ── Tools detail (structured, not raw XML) ────────────────────────
    tools = context.get("tools", [])
    tools_sorted = sorted(tools, key=lambda t: (t.position.get("x", 0), t.position.get("y", 0)))

    parts.append("## TOOLS (detailed configuration of each tool in this container):")
    parts.append("")

    for tool in tools_sorted:
        parts.append(f"### Tool ID={tool.tool_id} | Type={tool.tool_type}")
        if tool.annotation:
            parts.append(f"    Annotation: {tool.annotation}")
        if tool.container_id and tool.container_id != container.tool_id:
            sub = workflow.get_container(tool.container_id)
            if sub:
                parts.append(f"    Inside sub-container: '{sub.name}'")

        # Structured configuration
        config_text = _format_tool_config(tool)
        parts.append(config_text)

        # Also show raw XML as fallback for complex configs
        if tool.configuration_xml and not tool.parsed_config:
            parts.append(f"    Raw Configuration XML:")
            for line in tool.configuration_xml.split("\n"):
                parts.append(f"      {line}")

        parts.append("")

    # ── Text Input inline data ────────────────────────────────────────
    text_input_data = context.get("text_input_data", {})
    if text_input_data:
        parts.append("## INLINE DATA (from TextInput tools - use spark.createDataFrame()):")
        for tid, rows in text_input_data.items():
            parts.append(f"  TextInput Tool {tid}:")
            if rows:
                headers = list(rows[0].keys())
                parts.append(f"    Columns: {headers}")
                parts.append(f"    Data ({len(rows)} rows):")
                for row in rows:
                    parts.append(f"      {row}")
            parts.append("")

    return "\n".join(parts)


def build_system_prompt() -> str:
    """Build the system prompt for Claude AI code generation."""
    return """You are an expert data engineer converting Alteryx workflows to production-ready PySpark code for Databricks.

## Your Task
Given a detailed description of an Alteryx container (a group of connected tools), generate a COMPLETE, CORRECT PySpark script that replicates the EXACT same data transformation logic. Every single tool must be converted.

## CRITICAL RULES - READ CAREFULLY

### 1. Source Table Definitions (MANDATORY)
- Every external input listed in "SOURCE DATA" MUST have a corresponding `spark.table()` call
- If a Source Table Mapping is provided, use the mapped Databricks table name
- If no mapping is provided, use a TODO placeholder: `spark.table("TODO_catalog.schema.table_name")`
- Place ALL source table reads at the TOP of the script, right after imports
- Example:
  ```python
  # Source tables
  df_fact_nps = spark.table("gold.insurance.st_fact_nps")  # Tool 3111: InputData
  df_dim_provider = spark.table("gold.insurance.dim_provider")  # Tool 2367: InputData
  ```

### 2. Data Flow - Follow Connections EXACTLY
- The DATA FLOW section shows exactly how tools connect
- Process tools in dependency order (upstream before downstream)
- Each tool reads from its input connection and writes to a named output DataFrame
- Use the Tool ID in variable names for traceability: `df_filter_42`, `df_join_15`

### 3. Filter Tool (True/False Outputs)
- A Filter creates TWO output DataFrames:
  - `df_filter_XX_true = input_df.filter(condition)` for the True output
  - `df_filter_XX_false = input_df.filter(~(condition))` for the False output
- ONLY generate the outputs that are actually connected downstream
- Check the DATA FLOW to see which outputs are used (True, False, or both)

### 4. Join Tool (Join/Left/Right Outputs)
- A Join takes TWO inputs (Left and Right) and creates up to THREE outputs
- The `dest_connection` in the DATA FLOW tells you which INPUT port a DataFrame connects to:
  - `[Left]` = this DataFrame is the LEFT table of the join
  - `[Right]` = this DataFrame is the RIGHT table of the join
- The `origin_connection` tells you which OUTPUT is used downstream:
  - `[Join]` = INNER join result (rows that matched on BOTH sides)
  - `[Left]` = LEFT-ONLY rows (unmatched from left table)
  - `[Right]` = RIGHT-ONLY rows (unmatched from right table)
- Generate the join using the specified join keys
- Apply the Post-Join Column Handling: DROP columns marked False, RENAME as specified
- Example:
  ```python
  # Join Tool 15 - Join on matching keys
  df_join_15 = df_left.join(df_right, df_left["key1"] == df_right["key2"], "inner")
  # Drop duplicate/unwanted columns per SelectConfiguration
  df_join_15 = df_join_15.drop("Right_key2", "Right_unwanted_col")
  ```

### 5. Select Tool (Column Management)
- SELECT means: keep only the columns marked as selected, drop the rest
- Apply renames: `.withColumnRenamed("old", "new")`
- Respect the field order as specified

### 6. Formula Tool (Column Calculations)
- Each FormulaField creates or updates a column with `.withColumn()`
- Convert Alteryx expressions to PySpark:
  - `[ColumnName]` -> `F.col("ColumnName")`
  - `IF...THEN...ELSEIF...ELSE...ENDIF` -> `F.when().when().otherwise()`
  - `Contains([field], "text")` -> `F.col("field").contains("text")`
  - `!Contains(...)` -> `~F.col("field").contains("text")`
  - `IFNULL([x], default)` -> `F.coalesce(F.col("x"), F.lit(default))`
  - `Null()` -> `F.lit(None)`
  - `ToString([x], "fmt")` -> `F.col("x").cast("string")`
  - `ToNumber([x])` -> `F.col("x").cast("double")`
  - String `+` concatenation -> `F.concat()`
  - `Upper/Lower/Trim/Left/Right/PadLeft/PadRight` -> `F.upper/F.lower/F.trim/F.substring/F.lpad/F.rpad`
  - `DateTimeParse/DateTimeFormat` -> `F.to_date/F.date_format`
  - `GetWord([field], n)` -> `F.split(F.col("field"), " ")[n]`

### 7. Summarize Tool (Group By + Aggregation)
- Translate each SummarizeField action:
  - `GroupBy` -> group by column
  - `Sum/Min/Max/Avg/Count/CountDistinct/First/Last` -> PySpark aggregation
  - Respect the `rename` field for output column names

### 8. Union Tool
- `.unionByName(allowMissingColumns=True)` for multiple inputs

### 9. Sort Tool
- `.orderBy()` with ascending/descending as specified

### 10. Unique Tool
- `.dropDuplicates(subset=[...])` on specified fields

### 11. CrossTab / Pivot
- `.groupBy().pivot().agg()` pattern

### 12. TextInput Tool
- `spark.createDataFrame(data, schema)` using the inline data provided

## OUTPUT FORMAT
Generate a SINGLE, COMPLETE Databricks notebook-style Python file:

```python
# Databricks notebook source

# COMMAND ----------

# Module docstring explaining container purpose

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.types import *

# COMMAND ----------

# ============================================================
# SOURCE TABLES
# ============================================================
df_source_1 = spark.table("catalog.schema.table1")  # Tool ID XX
df_source_2 = spark.table("catalog.schema.table2")  # Tool ID YY

# COMMAND ----------

# ============================================================
# STEP 1: Filter (Tool ID XX) - description
# ============================================================
df_filter_XX_true = df_source_1.filter(...)

# COMMAND ----------

# ============================================================
# STEP 2: Join (Tool ID YY) - description
# ============================================================
df_join_YY = df_filter_XX_true.join(df_source_2, ...)

# COMMAND ----------

# ... more steps following the data flow ...

# COMMAND ----------

# ============================================================
# FINAL OUTPUT
# ============================================================
df_result.createOrReplaceTempView("output_view_name")
# OR: df_result.write.mode("overwrite").saveAsTable("catalog.schema.table")
```

## IMPORTANT
- Use `from pyspark.sql import functions as F` consistently
- NEVER use pandas
- Handle null values with `F.coalesce()` or `F.when().isNull()`
- Reference Tool IDs in comments for traceability
- Add `# COMMAND ----------` separators between logical sections
- EVERY tool in the TOOLS section must appear in your generated code
- EVERY connection in DATA FLOW must be represented
"""
