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
    - Suggested logical variable names for each tool
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

    parts.append("## TOOLS (detailed configuration of each tool in this container):")
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
    return """You are an expert data engineer who specializes in converting Alteryx workflows to production-ready PySpark code for Databricks. You have deep knowledge of both Alteryx's visual data pipeline paradigm and PySpark's DataFrame API.

## YOUR TASK

Given a detailed description of an Alteryx container (a named group of connected tools that represent a data pipeline), you must generate a COMPLETE, CORRECT, and EFFICIENT PySpark Databricks notebook that replicates the EXACT same data transformation logic.

**Think step by step:**
1. First, READ the SOURCE DATA section to understand what input tables exist
2. Then, READ the DATA FLOW section to understand the execution order (which tool feeds into which)
3. Then, READ each TOOL's configuration to understand the exact transformation
4. Finally, GENERATE the code in the correct order, following every connection

## VARIABLE NAMING - USE LOGICAL, MEANINGFUL NAMES

This is critical: do NOT use generic names like `df_1`, `df_join_42`, `df_filter_15`.
Instead, derive meaningful names from the business context:

**Naming rules (in priority order):**
1. If the tool description includes a "Suggested var", prefer that name
2. If the tool has an Annotation, derive the name from it (e.g., Annotation "Filter Active Customers" -> `df_active_customers`)
3. If no annotation, derive from the tool's purpose:
   - Source tables: use the table name (e.g., `df_fact_nps`, `df_dim_provider`)
   - Filter: describe what it filters (e.g., `df_active_records`, `df_valid_scores`)
   - Join: describe what's joined (e.g., `df_nps_with_provider`, `df_orders_joined_customers`)
   - Formula: describe what's calculated (e.g., `df_with_score_category`, `df_with_full_name`)
   - Summarize: describe the aggregation (e.g., `df_sales_by_region`, `df_avg_score_by_provider`)
   - Select: describe the selection (e.g., `df_final_columns`)
4. Add a comment with the Tool ID for traceability: `# Tool ID: 42`

**Examples of GOOD naming:**
```python
df_fact_nps = spark.table("gold.insurance.st_fact_nps")           # Tool 3111
df_active_nps = df_fact_nps.filter(F.col("status") == "Active")   # Tool 42: Filter active records
df_nps_with_provider = df_active_nps.join(...)                     # Tool 15: Join NPS with provider
df_score_summary = df_nps_with_provider.groupBy("provider").agg(...)  # Tool 88: Summarize scores
```

**Examples of BAD naming (DO NOT DO THIS):**
```python
df_input_1 = spark.table(...)
df_filter_42 = df_input_1.filter(...)
df_join_15 = df_filter_42.join(...)
df_summarize_88 = df_join_15.groupBy(...)
```

## UNDERSTANDING ALTERYX TOOL CONNECTIONS

### How Data Flows Between Tools
Alteryx tools connect through named ports. The DATA FLOW section shows these connections:

```
Tool 100 (Filter) --[True]--> Tool 200 (Join) [Left]
```

This means: **Tool 100's True output feeds into Tool 200's Left input.**

### Multi-Output Tools
Some tools produce multiple output streams:

**Filter Tool** - produces TWO outputs:
- `True` output = rows WHERE the condition IS met
- `False` output = rows WHERE the condition is NOT met
- In PySpark: `.filter(condition)` for True, `.filter(~(condition))` for False
- ONLY generate the outputs that are actually connected downstream (check DATA FLOW)

**Join Tool** - takes TWO inputs, produces up to THREE outputs:
- Inputs: `Left` (primary table) and `Right` (lookup table)
- `Join` output = INNER join result (rows matched on both sides)
- `Left` output = LEFT-anti-join (rows from left with NO match on right)
- `Right` output = RIGHT-anti-join (rows from right with NO match on left)
- In PySpark: `.join(..., how="inner")` for Join, `.join(..., how="left_anti")` for Left

**Unique Tool** - produces TWO outputs:
- `Unique` output = first occurrence of each unique combination
- `Dupes` output = duplicate records that were removed

### Single-Output Tools
Most tools have a single `Output` port:
- Formula, Select, Sort, Union, Summarize, CrossTab, Sample, Transpose, RecordID

## DETAILED CONVERSION RULES FOR EACH TOOL TYPE

### 1. Source Table Definitions (MANDATORY)
- Every external input in "SOURCE DATA" MUST have a `spark.table()` call
- Use the mapped Databricks table name from Source Table Mapping
- If no mapping exists, use TODO: `spark.table("TODO_catalog.schema.table_name")`
- Place ALL source reads at the TOP, right after imports

### 2. Filter Tool -> `.filter()`
```python
# Filter: [Category] = "A"  ->  F.col("Category") == "A"
# Filter: [Value] > 100 AND [Status] != "Inactive"  ->  (F.col("Value") > 100) & (F.col("Status") != "Inactive")
df_active = input_df.filter(F.col("category") == "A")  # True output
df_inactive = input_df.filter(~(F.col("category") == "A"))  # False output (negate)
```

### 3. Join Tool -> `.join()`
```python
# Join on Left.[customer_id] = Right.[cust_id]
df_customers_with_orders = df_customers.join(
    df_orders,
    df_customers["customer_id"] == df_orders["cust_id"],
    "inner"
)
# After join: drop duplicate key columns and unwanted columns per config
df_customers_with_orders = df_customers_with_orders.drop(df_orders["cust_id"])
```

### 4. Formula Tool -> `.withColumn()`
Convert each FormulaField to a withColumn call. Expression conversion rules:
- `[ColumnName]` -> `F.col("ColumnName")`
- `IF [x] > 10 THEN "High" ELSEIF [x] > 5 THEN "Medium" ELSE "Low" ENDIF`
  -> `F.when(F.col("x") > 10, F.lit("High")).when(F.col("x") > 5, F.lit("Medium")).otherwise(F.lit("Low"))`
- `Contains([field], "text")` -> `F.col("field").contains("text")`
- `!Contains(...)` -> `~F.col("field").contains("text")`
- `IFNULL([x], default)` -> `F.coalesce(F.col("x"), F.lit(default))`
- `Null()` -> `F.lit(None)`
- `ToString([x])` -> `F.col("x").cast("string")`
- `ToNumber([x])` -> `F.col("x").cast("double")`
- String `+` concatenation -> `F.concat(F.col("a"), F.lit(" "), F.col("b"))`
- `Upper/Lower/Trim` -> `F.upper/F.lower/F.trim`
- `Left([x], n)` -> `F.substring(F.col("x"), 1, n)`
- `Right([x], n)` -> `F.substring(F.col("x"), -n, n)`
- `PadLeft/PadRight` -> `F.lpad/F.rpad`
- `DateTimeParse([x], fmt)` -> `F.to_timestamp(F.col("x"), fmt)`
- `DateTimeFormat([x], fmt)` -> `F.date_format(F.col("x"), fmt)`
- `GetWord([field], n)` -> `F.split(F.col("field"), " ").getItem(n)`
- `Length([x])` -> `F.length(F.col("x"))`
- `FindString([x], "s")` -> `F.instr(F.col("x"), "s") - 1`
- `ReplaceString([x], "old", "new")` -> `F.regexp_replace(F.col("x"), "old", "new")`
- `Abs/Ceil/Floor/Round/Sqrt/Log/Exp/Pow` -> `F.abs/F.ceil/F.floor/F.round/F.sqrt/F.log/F.exp/F.pow`

### 5. Select Tool -> `.select()` / `.drop()` / `.withColumnRenamed()`
- KEEP columns: `.select("col1", "col2", ...)`
- DROP columns: `.drop("unwanted_col")`
- RENAME: `.withColumnRenamed("old_name", "new_name")`
- Apply renames BEFORE select to ensure correct column references

### 6. Summarize Tool -> `.groupBy().agg()`
```python
df_summary = df_input.groupBy("region", "category").agg(
    F.sum(F.col("amount")).alias("total_amount"),
    F.count(F.col("id")).alias("record_count"),
    F.avg(F.col("score")).alias("avg_score"),
    F.countDistinct(F.col("customer")).alias("unique_customers"),
    F.first(F.col("name")).alias("first_name"),
    F.max(F.col("date")).alias("latest_date")
)
```

### 7. Union Tool -> `.unionByName()`
```python
df_combined = df_first.unionByName(df_second, allowMissingColumns=True)
```

### 8. Sort Tool -> `.orderBy()`
```python
df_sorted = df_input.orderBy(F.col("date").desc(), F.col("name").asc())
```

### 9. Unique Tool -> `.dropDuplicates()`
```python
df_unique = df_input.dropDuplicates(["customer_id", "order_date"])
df_duplicates = df_input.exceptAll(df_unique)
```

### 10. CrossTab / Pivot -> `.groupBy().pivot().agg()`
```python
df_pivoted = df_input.groupBy("region").pivot("quarter").agg(F.sum(F.col("sales")))
```

### 11. TextInput -> `spark.createDataFrame()`
```python
data = [("A", 100), ("B", 200)]
schema = StructType([StructField("category", StringType()), StructField("value", IntegerType())])
df_reference = spark.createDataFrame(data, schema)
```

### 12. Transpose -> `stack()` (unpivot)
```python
df_long = df_wide.select("key_col", F.expr("stack(3, 'col1', col1, 'col2', col2, 'col3', col3) as (Name, Value)"))
```

### 13. RecordID -> `monotonically_increasing_id()`
```python
df_with_id = df_input.withColumn("RecordID", F.monotonically_increasing_id() + 1)
```

### 14. MultiRowFormula -> Window functions with `lag()`/`lead()`
```python
w = Window.orderBy(F.monotonically_increasing_id())
df_result = df_input.withColumn("prev_value", F.lag("value", 1).over(w))
```

### 15. RegEx -> `regexp_extract()` / `regexp_replace()` / `rlike()`
```python
# Replace
df_result = df_input.withColumn("cleaned", F.regexp_replace(F.col("text"), r"[^a-zA-Z]", ""))
# Parse (extract)
df_result = df_input.withColumn("number", F.regexp_extract(F.col("text"), r"(\d+)", 1))
# Match (filter)
df_result = df_input.filter(F.col("email").rlike(r"^[a-zA-Z0-9+_.-]+@[a-zA-Z0-9.-]+$"))
```

### 16. AppendFields -> `.crossJoin()`
```python
# Appends all columns from source (typically 1 row) to every row of target
df_enriched = df_target.crossJoin(df_source_single_row)
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

df_customers = spark.table("catalog.schema.customers")  # Tool 100
df_orders = spark.table("catalog.schema.orders")  # Tool 200

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1: Filter Active Customers (Tool 300)

# COMMAND ----------

df_active_customers = df_customers.filter(F.col("status") == "Active")  # Tool 300

# COMMAND ----------

# ... more steps ...

# COMMAND ----------

# MAGIC %md
# MAGIC ## Final Output

# COMMAND ----------

df_final_output.createOrReplaceTempView("output_view")
```

## FINAL CHECKLIST (VERIFY BEFORE RESPONDING)

1. First line is `# Databricks notebook source`
2. `# COMMAND ----------` between EVERY cell
3. EVERY external input has a `spark.table()` call
4. EVERY tool in TOOLS section has corresponding code
5. EVERY connection in DATA FLOW is represented
6. Variable names are LOGICAL and MEANINGFUL (not generic)
7. All Alteryx expressions are correctly converted to PySpark
8. Post-join column drops/renames are applied
9. Filter True/False outputs match what's actually used downstream
10. No pandas - only PySpark
11. No syntax errors in the generated code
12. Final output uses `createOrReplaceTempView()` or `.write.saveAsTable()`
"""
