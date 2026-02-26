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


def _infer_logical_name(tool: Tool, upstream_annotation: str = "") -> str:
    """Suggest a logical DataFrame variable name based on tool context."""
    ann = (tool.annotation or "").strip()
    if ann and len(ann) > 2:
        cleaned = ann.lower()
        cleaned = "".join(c if c.isalnum() or c == "_" else "_" for c in cleaned)
        cleaned = "_".join(p for p in cleaned.split("_") if p)
        if cleaned:
            return f"df_{cleaned}"

    pc = tool.parsed_config or {}
    tt = tool.tool_type

    if tt in ("Filter", "LockInFilter"):
        expr = pc.get("expression", "")
        import re
        m = re.search(r"\[([^\]]+)\]", expr)
        if m:
            col = m.group(1).lower().replace(" ", "_")
            return f"df_filtered_by_{col}"

    if tt in ("Join", "LockInJoin"):
        keys = pc.get("left_keys", [])
        if keys:
            key_str = "_".join(k.lower().replace(" ", "_") for k in keys[:2])
            return f"df_joined_on_{key_str}"

    if tt in ("Formula", "LockInFormula"):
        formulas = pc.get("formulas", [])
        if formulas:
            fields = [f.get("field", "") for f in formulas[:2] if f.get("field")]
            if fields:
                return f"df_calc_{'_'.join(f.lower().replace(' ', '_') for f in fields)}"

    if tt in ("Summarize", "LockInSummarize"):
        fields = pc.get("summarize_fields", pc.get("fields", []))
        groups = [f.get("field", "") for f in fields if f.get("action") == "GroupBy"]
        if groups:
            return f"df_summary_by_{'_'.join(g.lower().replace(' ', '_') for g in groups[:2])}"

    if tt == "Transpose":
        return "df_transposed"

    if tt == "RecordID":
        fname = pc.get("field_name", "id")
        return f"df_with_{fname.lower()}"

    return ""


def _identify_source_tools(context: dict) -> list:
    """
    Identify source tools within a module: tools that have no incoming
    internal connections (InputData, TextInput, etc.).
    """
    tools = context.get("tools", [])
    internal_connections = context.get("internal_connections", [])

    # Build set of tools that receive internal input
    tools_with_input = {c.dest_tool_id for c in internal_connections}

    # Source tools: tools with no internal incoming connections
    source_types = {"InputData", "TextInput", "LockInStreamIn", "DynamicInput"}
    sources = []
    for tool in tools:
        if tool.tool_id not in tools_with_input and tool.tool_type in source_types:
            sources.append(tool)

    return sources


def build_container_prompt(
    container: Container,
    context: dict,
    workflow: Workflow,
    source_tables_config: Optional[dict] = None,
) -> str:
    """
    Build a detailed prompt describing a container/module's full logic for Claude AI.

    Handles both:
    - Real containers (ToolContainer elements in Alteryx)
    - Root-level tools (virtual container with tool_id=-1)

    Returns a structured text description including:
    - Module name and purpose
    - All source tables with full details
    - All tools with structured configurations
    - Data flow (connections) in execution order
    - Suggested logical variable names for each tool
    - External inputs and outputs
    - Inline data (TextInput tools)
    """
    parts = []
    is_unified = (container.tool_id == -1)

    # ── Header ───────────────────────────────────────────────────────
    if is_unified:
        num_tools = len(context.get("tools", []))
        sub_containers = context.get("sub_containers", [])
        parts.append(f"# Workflow: {container.name}")
        parts.append(f"# This workflow contains {num_tools} tools total.")
        if sub_containers:
            parts.append(f"# Containers: {', '.join(c.name for c in sub_containers)}")
            parts.append("# All tools (both inside containers and at root level) must be converted together.")
    else:
        parts.append(f"# Container: {container.name}")
        parts.append(f"# Container ToolID: {container.tool_id}")
    parts.append("")

    # ── Source tables config (user-provided mapping) ──────────────────
    if source_tables_config:
        parts.append("## Source Table Mapping (Alteryx Input -> Databricks Table):")
        for key, value in source_tables_config.items():
            parts.append(f"  '{key}' -> spark.table(\"{value}\")")
        parts.append("")

    # ── Internal source tools (InputData/TextInput within this module) ─
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

            # Check source_tables_config for a mapping
            if source_tables_config:
                for key, value in source_tables_config.items():
                    if (str(tool.tool_id) == str(key)
                            or (tool.annotation and key.lower() in tool.annotation.lower())
                            or (table_name and key.lower() in table_name.lower())):
                        parts.append(f"      -> USE: spark.table(\"{value}\")")
                        break

            parts.append("")
        parts.append("")

    # ── External inputs with FULL details ─────────────────────────────
    external_inputs = context.get("external_inputs", [])
    source_tools = context.get("source_tools", {})
    if external_inputs:
        parts.append("## EXTERNAL INPUTS (data flowing INTO this module from outside):")
        parts.append("These are DataFrames that this module receives from other containers.")
        parts.append("You MUST define these as spark.table() reads at the top of your code.")
        parts.append("Use a MEANINGFUL variable name derived from the table/annotation, NOT generic names.")
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

    # ── Data Flow (connections) with annotations ──────────────────────
    internal_connections = context.get("internal_connections", [])
    if internal_connections:
        parts.append("## DATA FLOW (how data moves between tools inside this container):")
        parts.append("")
        parts.append("READ THIS SECTION CAREFULLY - it defines the exact execution order.")
        parts.append("The arrows show how DataFrames flow from one tool to the next.")
        parts.append("")
        parts.append("Connection format: SourceTool --[output_port]--> DestTool [input_port]")
        parts.append("  - output_port 'Output' = default single output")
        parts.append("  - output_port 'True'/'False' = Filter tool's matching/non-matching rows")
        parts.append("  - output_port 'Join'/'Left'/'Right' = Join tool's matched/left-unmatched/right-unmatched")
        parts.append("  - input_port 'Input' = default single input")
        parts.append("  - input_port 'Left'/'Right' = left/right side of a Join")
        parts.append("")
        for conn in internal_connections:
            origin_tool = workflow.get_tool(conn.origin_tool_id)
            dest_tool = workflow.get_tool(conn.dest_tool_id)
            origin_desc = f"{origin_tool.tool_type}" if origin_tool else "?"
            origin_ann = f" '{origin_tool.annotation}'" if origin_tool and origin_tool.annotation else ""
            dest_desc = f"{dest_tool.tool_type}" if dest_tool else "?"
            dest_ann = f" '{dest_tool.annotation}'" if dest_tool and dest_tool.annotation else ""
            parts.append(
                f"  Tool {conn.origin_tool_id} ({origin_desc}{origin_ann})"
                f" --[{conn.origin_connection}]--> "
                f"Tool {conn.dest_tool_id} ({dest_desc}{dest_ann}) [{conn.dest_connection}]"
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

    module_label = "this workflow" if is_unified else "this container"
    parts.append(f"## TOOLS (detailed configuration of each tool in {module_label}):")
    parts.append("")

    for tool in tools_sorted:
        # Suggest a logical variable name
        suggested_name = _infer_logical_name(tool)
        name_hint = f" | Suggested var: {suggested_name}" if suggested_name else ""

        parts.append(f"### Tool ID={tool.tool_id} | Type={tool.tool_type}{name_hint}")
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

    # ── Execution Summary ─────────────────────────────────────────────
    parts.append("## EXECUTION SUMMARY")
    parts.append(f"  Total tools: {len(tools)}")
    parts.append(f"  Total connections: {len(internal_connections)}")
    parts.append(f"  External inputs: {len(external_inputs)}")
    parts.append(f"  External outputs: {len(external_outputs)}")
    parts.append("")

    return "\n".join(parts)


def build_system_prompt() -> str:
    """Build the system prompt for Claude AI code generation."""
    return """You are an autonomous AI agent that converts Alteryx .yxmd workflow files into production-ready PySpark Databricks notebooks. You receive the raw XML of an Alteryx workflow and produce a complete, runnable .py notebook that replicates the exact same data transformation logic.

## YOUR TASK

Given a detailed description of an Alteryx module (either a named container or the root-level workflow), generate a COMPLETE, CORRECT, and EFFICIENT PySpark Databricks notebook.

Think step by step:
1. READ the SOURCE DATA section to understand what input tables/sources exist
2. READ the DATA FLOW section to understand execution order
3. READ each TOOL's configuration to understand the exact transformation
4. GENERATE code in the correct dependency order, following every connection

## OUTPUT CONTRACT

Produce a single .py file in Databricks notebook source format:
- First line: `# Databricks notebook source`
- Cells separated by: `# COMMAND ----------`
- Markdown cells: `# MAGIC %md` / `# MAGIC`
- Only PySpark DataFrame API — never pandas
- Validation section at the end
- Write/output statement commented out

## VARIABLE NAMING — USE LOGICAL, MEANINGFUL NAMES

NEVER use generic names like `df_1`, `df_join_42`, `df_filter_15`.

Naming priority:
1. If "Suggested var" is provided in the tool description, use it
2. Derive from Annotation (e.g., "Filter Active Customers" → `df_active_customers`)
3. Derive from business purpose:
   - Source: use table name (e.g., `df_fact_nps`, `df_dim_provider`)
   - Filter: describe filtered result (e.g., `df_active_records`)
   - Join: describe enriched result (e.g., `df_nps_with_provider`)
   - Formula: describe calculation (e.g., `df_with_score_category`)
   - Summarize: describe aggregation (e.g., `df_sales_by_region`)
4. Always add `# Tool ID: N` comment

## CONVERSION RULES

### InputData → spark.sql() or spark.createDataFrame()
```python
df_customers = spark.sql("SELECT * FROM catalog.schema.customers")  # Tool 1
df_customers = spark.sql("SELECT * FROM TODO.customers")  # placeholder
```

### Select → .drop() / .withColumnRenamed()
```python
df = df.drop("internal_col1", "internal_col2")
df = df.withColumnRenamed("old_name", "new_name")
```

### Formula → .withColumn()
Expression conversion rules:
- `[Column]` → `col("Column")`
- `IF [x]>10 THEN "High" ELSEIF [x]>5 THEN "Med" ELSE "Low" ENDIF` → `when(col("x")>10,"High").when(col("x")>5,"Med").otherwise("Low")`
- `IIF(condition, t, f)` → `when(condition, t).otherwise(f)`
- `Contains([x],"text")` → `col("x").contains("text")`
- `StartsWith([x],"pre")` → `col("x").startswith("pre")`
- `IsNull([x])` → `col("x").isNull()`
- `IsEmpty([x])` → `(col("x").isNull()) | (F.trim(col("x")) == "")`
- `IFNULL([x], default)` → `F.coalesce(col("x"), F.lit(default))`
- `Null()` → `F.lit(None)`
- `ToString([x])` → `col("x").cast("string")`
- `ToNumber([x])` → `col("x").cast("double")`
- `ToInteger([x])` → `col("x").cast("int")`
- `[a] + [b]` (string concat) → `F.concat(col("a"), col("b"))`
- `Upper([x])` / `Lower([x])` / `Trim([x])` → `F.upper(col("x"))` / `F.lower(col("x"))` / `F.trim(col("x"))`
- `Left([x], n)` → `F.substring(col("x"), 1, n)`
- `Right([x], n)` → `F.substring(col("x"), -n, n)`
- `Substring([x], start, len)` → `F.substring(col("x"), start+1, len)`  ← Alteryx is 0-based, Spark is 1-based
- `Length([x])` → `F.length(col("x"))`
- `FindString([x], "s")` → `F.instr(col("x"), "s") - 1`  ← Alteryx returns -1 for not found; instr returns 0
- `ReplaceString([x], "old", "new")` → `F.regexp_replace(col("x"), "old", "new")`
- `PadLeft([x], n, "c")` → `F.lpad(col("x"), n, "c")`
- `PadRight([x], n, "c")` → `F.rpad(col("x"), n, "c")`
- `GetWord([x], n)` → `F.split(col("x"), " ").getItem(n)`
- `DateTimeParse([x], fmt)` → `F.to_timestamp(col("x"), spark_fmt)`
- `DateTimeFormat([x], fmt)` → `F.date_format(col("x"), spark_fmt)`
- `DateTimeAdd([x], n, "days")` → `F.date_add(col("x"), n)`
- `DateTimeDiff([x], [y], "days")` → `F.datediff(col("x"), col("y"))`
- `DateTimeNow()` → `F.current_timestamp()`
- `DateTimeToday()` → `F.current_date()`
- `Abs([x])` → `F.abs(col("x"))`
- `Ceil([x])` / `Floor([x])` → `F.ceil(col("x"))` / `F.floor(col("x"))`
- `Round([x], d)` → `F.round(col("x"), d)`
- `Pow([x], n)` → `F.pow(col("x"), n)`
- `Mod([x], n)` → `col("x") % n`
- `MIN([x], [y])` → `F.least(col("x"), col("y"))`
- `MAX([x], [y])` → `F.greatest(col("x"), col("y"))`
- `REGEX_Match([x], pattern)` → `col("x").rlike(pattern)`
- `REGEX_Replace([x], pattern, repl)` → `F.regexp_replace(col("x"), pattern, repl)`

Alteryx date format → Spark: yyyy→yyyy, MM→MM, dd→dd, HH→HH, mm→mm, ss→ss, Month→MMMM, Mon→MMM, Day→EEEE, Dy→EEE

### Join → .join()
Step 1: Determine join type from which output ports are connected downstream:
- Only `Join` output connected → `how="inner"`
- `Left` output connected → generate second DataFrame with `how="left_anti"`
- `Right` output connected → generate with swapped left_anti

Step 2: Write the join.
```python
df_joined = df_left.join(
    df_right,
    (df_left["key1"] == df_right["key1"]),
    "inner"
)
```

Step 3: Apply post-join column handling (CRITICAL — DO NOT SKIP):
Parse every SelectField in the Join tool config:
- selected="False" → drop the column
- rename="new_name" → rename the column

```python
df_joined = df_joined.drop(df_right["key1"])
df_joined = df_joined.withColumnRenamed("Left_period", "period")
```

Step 4: Handle ambiguous column names — use `on=["shared_key"]` when both sides share the key name.

### Filter → .filter()
ONLY generate outputs that are actually connected downstream. Check the DATA FLOW.
```python
df_matched = df.filter(col("status") == "Active")        # True output
df_unmatched = df.filter(~(col("status") == "Active"))   # False output
```

### DynamicRename → bulk .withColumnRenamed()
```python
# Add prefix
for c in df.columns:
    df = df.withColumnRenamed(c, f"prefix_{c}")
# Remove prefix
for c in df.columns:
    if c.startswith("prefix_"):
        df = df.withColumnRenamed(c, c[len("prefix_"):])
# RightInputRows mode
rename_map = {"old1": "new1", "old2": "new2"}
for old, new in rename_map.items():
    if old in df.columns:
        df = df.withColumnRenamed(old, new)
```

### Summarize → .groupBy().agg()
```python
df_summary = df.groupBy("key1","key2").agg(
    F.sum("amount").alias("total_amount"),
    F.count("id").alias("record_count"),
    F.avg("score").alias("avg_score"),
    F.countDistinct("customer").alias("unique_customers"),
    F.first("label").alias("label"),
)
```
GroupBy fields go in `.groupBy()`, NOT in `.agg()`.

### Union → .unionByName()
ALWAYS use `allowMissingColumns=True`:
```python
from functools import reduce
all_streams = [stream1, stream2, stream3]
df_combined = reduce(lambda a, b: a.unionByName(b, allowMissingColumns=True), all_streams)
```

### Transpose → F.expr("stack(...)")
Cast all value columns to string to prevent type mismatch:
```python
df_long = df_wide.select(
    "key_col1", "key_col2",
    F.expr("stack(3, 'FieldA', cast(`FieldA` as string), 'FieldB', cast(`FieldB` as string), 'FieldC', cast(`FieldC` as string)) as (Name, Value)")
)
```

### CrossTab → .groupBy().pivot().agg()
```python
df_pivoted = df.groupBy("group_col").pivot("header_col").agg(F.sum("value_col"))
```

### TextInput → spark.createDataFrame()
```python
data = [("val1", 100), ("val2", 200)]
schema = StructType([StructField("category", StringType()), StructField("amount", IntegerType())])
df_reference = spark.createDataFrame(data, schema)
```

### Sort → .orderBy()
```python
df_sorted = df.orderBy(col("date").desc(), col("name").asc())
```

### Unique → .dropDuplicates()
```python
df_unique = df.dropDuplicates(["key1", "key2"])
df_dupes = df.exceptAll(df_unique)  # only if Dupes output is connected
```

### RecordID → monotonically_increasing_id()
```python
df = df.withColumn("RecordID", F.monotonically_increasing_id() + 1)
```

### MultiRowFormula → Window functions
```python
from pyspark.sql.window import Window
w = Window.orderBy(F.monotonically_increasing_id())
df = df.withColumn("prev_value", F.lag("value", 1).over(w))
df = df.withColumn("running_sum", F.sum("value").over(w))
```

### RegEx → regexp_extract() / regexp_replace() / rlike()
```python
df = df.withColumn("cleaned", F.regexp_replace(col("text"), r"[^a-zA-Z0-9]", ""))
df = df.withColumn("number", F.regexp_extract(col("text"), r"(\\d+)", 1))
df = df.filter(col("email").rlike(r"^[\\w.]+@[\\w.]+$"))
```

### TextToColumns → F.split()
```python
df = df.withColumn("parts", F.split(col("field"), ","))
df = df.withColumn("first_part", col("parts").getItem(0))
# Or explode into rows:
df = df.withColumn("part", F.explode(F.split(col("field"), ",")))
```

### AppendFields → .crossJoin()
```python
df_enriched = df_target.crossJoin(df_source_single_row)
```

## BUG PREVENTION RULES

### Rule 1: Bitwise Operator Precedence
```python
# WRONG:
~col("x").isin(["a", "b"]) | col("x").isNull()
# CORRECT:
(~col("x").isin(["a", "b"])) | col("x").isNull()
```

### Rule 2: Defensive String Matching
```python
# WRONG: df.filter(col("status") == "Active")
# CORRECT: df.filter(F.lower(F.trim(col("status"))) == "active")
```

### Rule 3: Post-Join Column Handling is Mandatory
Every Join tool has SelectField configs. Skipping them produces wrong schemas.

### Rule 4: Only Generate Connected Output Ports
A Filter has True/False outputs, but often only one is used. A Join has Join/Left/Right. Only generate code for ports that appear in the DATA FLOW section.

### Rule 5: Union Always Needs allowMissingColumns=True
Without this flag, PySpark throws AnalysisException when streams differ in columns.

### Rule 6: Transpose Values Must Be Cast to String
All value columns in stack() must be cast to string.

### Rule 7: No .count() in the Pipeline
Only use .count() in the validation section, never in transformation logic.

### Rule 8: Substring Index Offset
Alteryx Substring() is 0-based. PySpark F.substring() is 1-based. Add 1 to start position.

### Rule 9: FindString Returns -1 for Not Found
Alteryx FindString returns -1 when not found; PySpark instr returns 0. Subtract 1 from instr result.

## PERFORMANCE GUIDELINES

1. Filter before join — reduce rows before expensive shuffle
2. Select early — drop unused columns before joins
3. Broadcast small tables — `df_big.join(F.broadcast(df_small), ...)`
4. Use `on=["key"]` when join keys share the same name on both sides — automatically deduplicates the key column in the result
5. Avoid collecting to driver — no .collect(), .toPandas(), .show() in pipeline

## VALIDATION SECTION

Always generate this at the end:
```python
# ── Column Completeness ──────────────────────────────────────────
EXPECTED_COLUMNS = [
    "col1", "col2", "col3",  # list every expected output column
]
actual_columns = df_final.columns
missing = [c for c in EXPECTED_COLUMNS if c not in actual_columns]
extra   = [c for c in actual_columns if c not in EXPECTED_COLUMNS]
assert not missing, f"Missing columns: {missing}"
print(f"Extra columns (review): {extra}")

# ── Row Count Sanity ─────────────────────────────────────────────
row_count = df_final.count()
print(f"Final row count: {row_count:,}")
assert row_count > 0, "Output DataFrame is empty!"

# ── Null Check on Key Columns ────────────────────────────────────
key_cols = ["key_col1", "key_col2"]
for kc in key_cols:
    null_count = df_final.filter(F.col(kc).isNull()).count()
    print(f"Nulls in {kc}: {null_count}")

# ── Sample Output ────────────────────────────────────────────────
df_final.limit(5).display()
```

## OUTPUT FORMAT - DATABRICKS NOTEBOOK (MANDATORY)

You MUST generate output in **Databricks notebook source format**:

1. **FIRST LINE** must be exactly: `# Databricks notebook source`
2. **Cell separator**: blank line + `# COMMAND ----------` + blank line
3. **Markdown cells**: `# MAGIC %md` on first line, `# MAGIC` for subsequent lines
4. **Code cells**: regular Python code
5. **Each logical step** gets its own cell with a markdown header

**EXACT FORMAT:**
```
# Databricks notebook source

# COMMAND ----------

# MAGIC %md
# MAGIC # Container: <name>
# MAGIC Auto-converted from Alteryx workflow to PySpark.

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.types import *
from pyspark.sql.window import Window

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load Source Tables

# COMMAND ----------

df_customers = spark.sql("SELECT * FROM catalog.schema.customers")  # Tool 100
df_orders = spark.sql("SELECT * FROM catalog.schema.orders")  # Tool 200

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1: Filter Active Customers (Tool 300)

# COMMAND ----------

df_active_customers = df_customers.filter(F.lower(F.trim(F.col("status"))) == "active")  # Tool 300

# COMMAND ----------

# ... more steps ...

# COMMAND ----------

# MAGIC %md
# MAGIC ## Final Output

# COMMAND ----------

# df_final_output.write.saveAsTable("catalog.schema.output_table")  # commented out
df_final_output.createOrReplaceTempView("output_view")
```

## FINAL CHECKLIST (VERIFY BEFORE RESPONDING)

1. First line is `# Databricks notebook source`
2. `# COMMAND ----------` between EVERY cell
3. EVERY external input has a `spark.sql()` or `spark.createDataFrame()` call
4. EVERY tool in TOOLS section has corresponding code
5. EVERY connection in DATA FLOW is represented
6. Variable names are LOGICAL and MEANINGFUL (not generic)
7. All Alteryx expressions are correctly converted to PySpark (including date formats)
8. Post-join column drops/renames are applied (mandatory)
9. Filter True/False outputs match what's actually used downstream
10. No pandas — only PySpark DataFrame API
11. No syntax errors in the generated code
12. Final output uses `createOrReplaceTempView()` or `.write.saveAsTable()` (write commented out)
13. All Union operations use `allowMissingColumns=True`
14. No `.count()` calls in transformation pipeline (only in validation section)
15. Substring start positions are offset by +1 (Alteryx 0-based → Spark 1-based)
16. FindString is converted to `F.instr(...) - 1` to preserve -1-for-not-found semantics
17. Bitwise NOT expressions use explicit parentheses to avoid operator precedence bugs
18. Small lookup tables are wrapped in `F.broadcast()` for join performance
"""
