# Databricks notebook source
# MAGIC %md
# MAGIC # Context Builder
# MAGIC Transforms parsed Alteryx workflow data into a structured prompt context
# MAGIC that Claude AI can use to generate accurate PySpark code.
# MAGIC
# MAGIC **Usage:** This notebook is imported by other notebooks via `%run ./context_builder`
# MAGIC
# MAGIC **Requires:** `%run ./models` must be executed first (handled by the orchestrator).

# COMMAND ----------

# MAGIC %run ./models

# COMMAND ----------

from typing import Optional

# COMMAND ----------

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

    if not lines and pc:
        for k, v in pc.items():
            lines.append(f"    {k}: {v}")

    return "\n".join(lines) if lines else "    (no configuration details)"

# COMMAND ----------

def _identify_source_tools(context):
    """
    Identify source tools within a module: tools that have no incoming
    internal connections (InputData, TextInput, etc.).
    """
    tools = context.get("tools", [])
    internal_connections = context.get("internal_connections", [])
    tools_with_input = {c.dest_tool_id for c in internal_connections}
    source_types = {"InputData", "TextInput", "LockInStreamIn", "DynamicInput"}
    return [t for t in tools if t.tool_id not in tools_with_input and t.tool_type in source_types]

# COMMAND ----------

def build_container_prompt(
    container: Container,
    context: dict,
    workflow: Workflow,
    source_tables_config: Optional[dict] = None,
) -> str:
    """Build a detailed prompt describing a module's full logic for Claude AI.
    Handles both real containers and root-level virtual containers."""
    parts = []
    is_root = (container.tool_id == -1)

    # Header
    if is_root:
        parts.append(f"# Module: {container.name} (Root-Level Workflow)")
        parts.append(f"# This module contains all {len(context.get('tools', []))} tools at the "
                      "root level of the workflow (outside any container).")
    else:
        parts.append(f"# Container: {container.name}")
        parts.append(f"# Container ToolID: {container.tool_id}")
    parts.append("")

    # Source tables config
    if source_tables_config:
        parts.append("## Source Table Mapping (Alteryx Input -> Databricks Table):")
        for key, value in source_tables_config.items():
            parts.append(f"  '{key}' -> spark.table(\"{value}\")")
        parts.append("")

    # Internal source tools (InputData/TextInput within this module)
    internal_sources = _identify_source_tools(context)
    if internal_sources:
        parts.append("## SOURCE DATA (data source tools within this module):")
        parts.append("These tools read data from external sources. Each MUST become a")
        parts.append("spark.table() call (or spark.createDataFrame() for TextInput) at the top of your code.")
        parts.append("Use a MEANINGFUL variable name derived from the table name or annotation.")
        parts.append("")
        for tool in internal_sources:
            pc = tool.parsed_config or {}
            parts.append(f"  ### Source Tool {tool.tool_id} ({tool.tool_type})")
            if tool.annotation:
                parts.append(f"      Annotation: {tool.annotation}")
            table_name = pc.get("table_name", "")
            if table_name:
                parts.append(f"      Table/File: {table_name}")
            fields = pc.get("fields", [])
            if fields:
                parts.append(f"      Columns ({len(fields)}):")
                for f in fields:
                    parts.append(f"        - {f.get('name', '')} ({f.get('type', '')})")
            if source_tables_config:
                for key, value in source_tables_config.items():
                    if (str(tool.tool_id) == str(key)
                            or (tool.annotation and key.lower() in tool.annotation.lower())
                            or (table_name and key.lower() in table_name.lower())):
                        parts.append(f"      -> USE: spark.table(\"{value}\")")
                        break
            parts.append("")
        parts.append("")

    # External inputs with FULL details
    external_inputs = context.get("external_inputs", [])
    source_tools = context.get("source_tools", {})
    if external_inputs:
        parts.append("## EXTERNAL INPUTS (data flowing INTO this module from outside):")
        parts.append("These are DataFrames that this module receives from other containers.")
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

            if src_parsed:
                table_name = src_parsed.get("table_name", "")
                if table_name:
                    parts.append(f"      Original Table/File: {table_name}")
                fields = src_parsed.get("fields", [])
                if fields:
                    parts.append(f"      Columns ({len(fields)}):")
                    for f in fields:
                        parts.append(f"        - {f.get('name', '')} ({f.get('type', '')})")

            if source_tables_config:
                for key, value in source_tables_config.items():
                    if (str(conn.origin_tool_id) == str(key)
                            or (src_annotation and key.lower() in src_annotation.lower())
                            or (src_parsed.get("table_name", "") and key.lower() in src_parsed.get("table_name", "").lower())):
                        parts.append(f"      -> USE: spark.table(\"{value}\")")
                        break

            text_data = src_info.get("text_input_data")
            if text_data:
                parts.append(f"      Inline Data ({len(text_data)} rows):")
                if text_data:
                    headers = list(text_data[0].keys())
                    parts.append(f"        Columns: {headers}")
                    for row in text_data[:10]:
                        parts.append(f"        {row}")

            parts.append("")
        parts.append("")

    # Sub-containers
    sub_containers = context.get("sub_containers", [])
    if sub_containers:
        parts.append("## Sub-Containers (nested groups within this container):")
        for sc in sub_containers:
            parts.append(f"  - '{sc.name}' (ToolID={sc.tool_id})")
        parts.append("")

    # Data Flow
    internal_connections = context.get("internal_connections", [])
    if internal_connections:
        module_label = "root-level workflow" if is_root else "this container"
        parts.append(f"## DATA FLOW (how data moves between tools inside {module_label}):")
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

    # External outputs
    external_outputs = context.get("external_outputs", [])
    if external_outputs:
        parts.append("## OUTPUTS (data flowing OUT of this module to downstream):")
        for conn in external_outputs:
            dest_tool = workflow.get_tool(conn.dest_tool_id)
            if dest_tool:
                dest_container = workflow._find_tool_container_name(dest_tool.tool_id)
                dest_desc = (f"Tool {conn.dest_tool_id} ({dest_tool.tool_type}, "
                             f"'{dest_tool.annotation}', container='{dest_container}')")
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

    # Tools detail (structured)
    tools = context.get("tools", [])
    tools_sorted = sorted(tools, key=lambda t: (t.position.get("x", 0), t.position.get("y", 0)))

    module_label = "root-level workflow" if is_root else "this container"
    parts.append(f"## TOOLS (detailed configuration of each tool in {module_label}):")
    parts.append("")

    for tool in tools_sorted:
        parts.append(f"### Tool ID={tool.tool_id} | Type={tool.tool_type}")
        if tool.annotation:
            parts.append(f"    Annotation: {tool.annotation}")
        if tool.container_id and tool.container_id != container.tool_id:
            sub = workflow.get_container(tool.container_id)
            if sub:
                parts.append(f"    Inside sub-container: '{sub.name}'")

        config_text = _format_tool_config(tool)
        parts.append(config_text)

        if tool.configuration_xml and not tool.parsed_config:
            parts.append(f"    Raw Configuration XML:")
            for line in tool.configuration_xml.split("\n"):
                parts.append(f"      {line}")

        parts.append("")

    # Text Input inline data
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

    # Execution Summary
    parts.append("## EXECUTION SUMMARY")
    parts.append(f"  Total tools: {len(tools)}")
    parts.append(f"  Total connections: {len(internal_connections)}")
    parts.append(f"  External inputs: {len(external_inputs)}")
    parts.append(f"  External outputs: {len(external_outputs)}")
    parts.append("")

    return "\n".join(parts)

# COMMAND ----------

def build_system_prompt() -> str:
    """Build the system prompt for Claude AI code generation."""

    # IMPORTANT: Build Databricks notebook markers as variables.
    # If we write literal '# COMMAND ----------' at column 0 inside a triple-quoted
    # string, Databricks will interpret it as a cell separator when parsing this
    # notebook file, breaking the function. Same for '# MAGIC' and '# Databricks notebook source'.
    _CMD = "# COMMAND ----------"
    _NB = "# Databricks notebook source"
    _MD = "# MAGIC %md"
    _M = "# MAGIC"

    format_example = (
        f"{_NB}\n"
        f"\n"
        f"{_CMD}\n"
        f"\n"
        f"{_MD}\n"
        f"{_M} # Container: <container_name>\n"
        f"{_M} Auto-converted from Alteryx workflow to PySpark.\n"
        f"\n"
        f"{_CMD}\n"
        f"\n"
        f"from pyspark.sql import functions as F\n"
        f"from pyspark.sql.types import *\n"
        f"from pyspark.sql.window import Window\n"
        f"\n"
        f"{_CMD}\n"
        f"\n"
        f"{_MD}\n"
        f"{_M} ## Load Source Tables\n"
        f"\n"
        f"{_CMD}\n"
        f"\n"
        f"# Load customer data (Tool ID XX)\n"
        f"df_customers = spark.table(\"catalog.schema.customers\")\n"
        f"\n"
        f"# Load orders data (Tool ID YY)\n"
        f"df_orders = spark.table(\"catalog.schema.orders\")\n"
        f"\n"
        f"{_CMD}\n"
        f"\n"
        f"{_MD}\n"
        f"{_M} ## Step 1: Filter Active Customers (Tool ID XX)\n"
        f"\n"
        f"{_CMD}\n"
        f"\n"
        f"# Tool XX: Filter active customers\n"
        f"df_active_customers = df_customers.filter(F.col(\"status\") == \"Active\")\n"
        f"\n"
        f"{_CMD}\n"
        f"\n"
        f"{_MD}\n"
        f"{_M} ## Step 2: Join with Orders (Tool ID YY)\n"
        f"\n"
        f"{_CMD}\n"
        f"\n"
        f"# Tool YY: Join customers with orders on customer_id\n"
        f"df_customers_with_orders = df_active_customers.join(\n"
        f"    df_orders,\n"
        f"    df_active_customers[\"customer_id\"] == df_orders[\"cust_id\"],\n"
        f"    \"inner\"\n"
        f")\n"
        f"df_customers_with_orders = df_customers_with_orders.drop(df_orders[\"cust_id\"])\n"
        f"\n"
        f"{_CMD}\n"
        f"\n"
        f"{_MD}\n"
        f"{_M} ## Final Output\n"
        f"\n"
        f"{_CMD}\n"
        f"\n"
        f"df_customers_with_orders.createOrReplaceTempView(\"output_view\")"
    )

    return f"""You are an expert data engineer who specializes in converting Alteryx workflows to production-ready PySpark code for Databricks.

## YOUR TASK

Given a detailed description of an Alteryx module (either a named container or the root-level workflow with all tools outside containers), generate a COMPLETE, CORRECT, and EFFICIENT PySpark Databricks notebook that replicates the EXACT same data transformation logic.

A module can be:
- A **Container**: a named group of tools inside an Alteryx ToolContainer element
- **Root-level workflow**: all tools that sit at the root level of the workflow, outside any container

**Think step by step:**
1. READ the SOURCE DATA section - understand what input tables/sources exist
2. READ the DATA FLOW section - understand the execution order (which tool feeds which)
3. READ each TOOL's configuration - understand the exact transformation
4. GENERATE the code in correct order, following every connection

## VARIABLE NAMING - USE LOGICAL, MEANINGFUL NAMES

**This is critical: do NOT use generic names like df_1, df_join_42, df_filter_15.**

Naming rules (in priority order):
1. If the tool has a "Suggested var" hint, prefer that name
2. If the tool has an Annotation, derive the name from it
3. Otherwise, derive from the tool's purpose (e.g., df_active_records, df_nps_with_provider)

Good examples: df_active_customers, df_nps_with_provider, df_summary_by_region
Bad examples: df_filter_42, df_join_15, df_summarize_88

## UNDERSTANDING TOOL CONNECTIONS

### Multi-Output Tools
**Filter**: True output = matching rows, False = non-matching. Only generate outputs actually used downstream.
**Join**: Takes Left + Right inputs. Outputs: Join (matched), Left (left-unmatched), Right (right-unmatched).
**Unique**: Unique output = first occurrence, Dupes = duplicates.

### Connection Format
`Tool 100 (Filter) --[True]--> Tool 200 (Join) [Left]` means:
Tool 100's True output feeds into Tool 200's Left input.

## TOOL CONVERSION REFERENCE

- **Filter**: `.filter(condition)` / `.filter(~condition)`
- **Join**: `.join(right_df, condition, "inner")` with post-join drops/renames
- **Formula**: `.withColumn("field", expression)` for each FormulaField
- **Select**: `.select()` / `.drop()` / `.withColumnRenamed()`
- **Summarize**: `.groupBy().agg(F.sum().alias(), F.count().alias(), ...)`
- **Union**: `.unionByName(allowMissingColumns=True)`
- **Sort**: `.orderBy(F.col("x").desc())`
- **Unique**: `.dropDuplicates(subset=[...])`
- **CrossTab**: `.groupBy().pivot().agg()`
- **TextInput**: `spark.createDataFrame(data, schema)`
- **Transpose**: `stack()` unpivot expression
- **RecordID**: `.withColumn("id", F.monotonically_increasing_id())`
- **MultiRowFormula**: Window functions with `F.lag()`/`F.lead()`
- **RegEx**: `F.regexp_extract()` / `F.regexp_replace()` / `.rlike()`

## Alteryx Expression -> PySpark Conversion
- `[ColumnName]` -> `F.col("ColumnName")`
- `IF...THEN...ELSEIF...ELSE...ENDIF` -> `F.when().when().otherwise()`
- `Contains/StartsWith/EndsWith` -> `.contains()/.startswith()/.endswith()`
- `IFNULL([x], d)` -> `F.coalesce(F.col("x"), F.lit(d))`
- `Null()` -> `F.lit(None)`
- `ToString/ToNumber` -> `.cast("string")/.cast("double")`
- String `+` -> `F.concat()`
- `Upper/Lower/Trim/Left/Right` -> `F.upper/F.lower/F.trim/F.substring`
- `DateTimeParse/DateTimeFormat` -> `F.to_timestamp/F.date_format`
- `GetWord([x], n)` -> `F.split(F.col("x"), " ").getItem(n)`
- `Length/FindString/ReplaceString` -> `F.length/F.instr/F.regexp_replace`

## OUTPUT FORMAT - DATABRICKS NOTEBOOK (MANDATORY)

1. FIRST LINE: `{_NB}` (no exceptions)
2. Cell separator: blank line + `{_CMD}` + blank line
3. Markdown cells: `{_MD}` then `{_M}` lines
4. Each transformation step in its own cell with markdown header
5. Import `Window` if using window functions

Example:
```
{format_example}
```

## CHECKLIST (verify before responding)
1. First line is `{_NB}`
2. Every external input has a spark.table() call
3. Every tool has corresponding code
4. Every connection is represented
5. Variable names are LOGICAL and MEANINGFUL
6. All expressions correctly converted
7. No pandas, no syntax errors
"""
