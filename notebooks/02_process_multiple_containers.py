# Databricks notebook source
# MAGIC %md
# MAGIC # Alteryx to PySpark - Multi-Container Converter
# MAGIC
# MAGIC Converts a **specific container** from an Alteryx `.yxmd` workflow into production-ready PySpark code using **Claude AI**.
# MAGIC
# MAGIC Use this notebook when your workflow has **multiple containers** and you want to convert **only one container at a time**.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC **Prerequisites:**
# MAGIC 1. Run `00_setup` notebook first to install dependencies
# MAGIC 2. Store your Anthropic API key in Databricks Secrets
# MAGIC 3. Upload your `.yxmd` file to a Databricks Volume or DBFS
# MAGIC
# MAGIC **How to use:**
# MAGIC 1. Configure the widgets at the top of this notebook
# MAGIC 2. Set **Run Mode** to `list_containers` first to see all available containers
# MAGIC 3. Enter the exact **Container Name** you want to convert
# MAGIC 4. Set **Run Mode** to `convert` and click **Run All**
# MAGIC
# MAGIC ---

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1: Configuration

# COMMAND ----------

# -- Databricks Widgets --
dbutils.widgets.text(
    "workflow_path",
    "",
    "1. Workflow File Path (.yxmd)"
)
dbutils.widgets.text(
    "output_dir",
    "/Workspace/Users/shared/alteryx_converter/output",
    "2. Output Directory"
)
dbutils.widgets.text(
    "container_name",
    "",
    "3. Container Name (required)"
)
dbutils.widgets.dropdown(
    "run_mode",
    "list_containers",
    ["list_containers", "convert", "dry_run"],
    "4. Run Mode"
)
dbutils.widgets.dropdown(
    "max_retries",
    "2",
    ["0", "1", "2", "3", "4", "5"],
    "5. Max Retries"
)
dbutils.widgets.text(
    "secret_scope",
    "alteryx-converter",
    "6. Secret Scope Name"
)
dbutils.widgets.text(
    "secret_key",
    "anthropic-api-key",
    "7. Secret Key Name"
)
dbutils.widgets.text(
    "source_tables_json",
    "",
    "8. Source Tables JSON Path (optional)"
)

# COMMAND ----------

# -- Read widget values --

WORKFLOW_PATH = dbutils.widgets.get("workflow_path")
OUTPUT_DIR = dbutils.widgets.get("output_dir")
CONTAINER_NAME = dbutils.widgets.get("container_name").strip() or None
RUN_MODE = dbutils.widgets.get("run_mode")
MODEL = "claude-sonnet-4-20250514"
MAX_RETRIES = int(dbutils.widgets.get("max_retries"))
SECRET_SCOPE = dbutils.widgets.get("secret_scope")
SECRET_KEY = dbutils.widgets.get("secret_key")
SOURCE_TABLES_JSON = dbutils.widgets.get("source_tables_json").strip() or None

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2: Load Dependencies

# COMMAND ----------

# MAGIC %run ./models

# COMMAND ----------

# MAGIC %run ./context_builder

# COMMAND ----------

# MAGIC %run ./ai_generator

# COMMAND ----------

# MAGIC %run ./parser

# COMMAND ----------

# MAGIC %run ./utils

# COMMAND ----------

import time
import os
import json

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3: Validate Configuration

# COMMAND ----------

def _validate():
    errors = []
    if not WORKFLOW_PATH:
        errors.append("Workflow file path is required. Set the 'workflow_path' widget.")
    if RUN_MODE == "convert":
        if not CONTAINER_NAME:
            errors.append(
                "Container name is required for 'convert' mode. "
                "Run with mode='list_containers' first to see available containers, "
                "then set the 'container_name' widget."
            )
        try:
            dbutils.secrets.get(scope=SECRET_SCOPE, key=SECRET_KEY)
        except Exception:
            errors.append(
                f"API key not found in secrets (scope='{SECRET_SCOPE}', key='{SECRET_KEY}'). "
                "Run 00_setup notebook first."
            )
    if errors:
        for err in errors:
            print(f"CONFIG ERROR: {err}")
        raise ValueError("Configuration validation failed. See errors above.")
    print("Configuration validated successfully.")
    print(f"  Workflow:   {WORKFLOW_PATH}")
    print(f"  Output:     {OUTPUT_DIR}")
    print(f"  Mode:       {RUN_MODE}")
    print(f"  Model:      {MODEL}")
    print(f"  Container:  {CONTAINER_NAME or '(not set - use list_containers mode first)'}")

_validate()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4: Parse the Alteryx Workflow

# COMMAND ----------

print_banner()

# Support both DBFS and local/Volumes paths
workflow_path = WORKFLOW_PATH
if workflow_path.startswith("dbfs:"):
    local_path = "/tmp/workflow_to_parse.yxmd"
    dbutils.fs.cp(workflow_path, f"file:{local_path}")
    workflow_path = local_path

print(f"Parsing workflow: {workflow_path}")

# COMMAND ----------

t0 = time.time()
parser_obj = AlteryxWorkflowParser(workflow_path)
workflow = parser_obj.parse()
parse_time = time.time() - t0

root_tools = workflow.get_root_tools()
container_tool_count = sum(len(c.child_tools) for c in workflow.containers)

print(f"  Parsed in {parse_time:.1f}s")
print(f"  Containers: {len(workflow.containers)}")
print(f"  Tools in containers: {container_tool_count}")
print(f"  Root-level tools: {len(root_tools)}")
print(f"  Total tools: {len(workflow.all_tools)}")
print(f"  Total connections: {len(workflow.connections)}")

if root_tools:
    print(f"  * Root-level tools can be converted as 'Main_Workflow'")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 5: List All Containers

# COMMAND ----------

print("\nContainers found in workflow:")
print("-" * 70)
idx = 1
for container in workflow.containers:
    tool_count = len(container.child_tools)
    disabled_tag = " [DISABLED]" if container.disabled else ""
    print(f"  {idx:2d}. {container.name:<40s} ({tool_count} tools){disabled_tag}")
    idx += 1
if root_tools:
    print(f"  {idx:2d}. {'Main_Workflow':<40s} ({len(root_tools)} tools) [ROOT-LEVEL]")
print("-" * 70)
print(f"  Total: {len(workflow.containers)} containers" + (f" + root-level tools" if root_tools else ""))
print(f"\n  To convert a specific container, set 'container_name' widget to one of the names above.")

# COMMAND ----------

if RUN_MODE == "list_containers":
    print("\nRun mode: list_containers -- listing complete.")
    print("Set the 'container_name' widget and change run_mode to 'convert' to proceed.")
    dbutils.notebook.exit("LISTED_CONTAINERS")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 6: Find and Validate Selected Container

# COMMAND ----------

# Build the conversion list for the selected container
items_to_convert = []

# Search through named containers
for container in workflow.containers:
    if container.name.lower() == CONTAINER_NAME.lower():
        ctx = workflow.get_container_context(container.tool_id)
        items_to_convert.append({
            "container": container,
            "context": ctx,
            "is_root": False,
        })
        break

# Check if user requested root-level tools
if not items_to_convert and CONTAINER_NAME.lower() == "main_workflow":
    root_ctx = workflow.get_root_context()
    if root_ctx:
        items_to_convert.append({
            "container": root_ctx["container"],
            "context": root_ctx,
            "is_root": True,
        })

if not items_to_convert:
    print(f"ERROR: Container '{CONTAINER_NAME}' not found in this workflow.")
    print("\nAvailable containers:")
    for c in workflow.containers:
        print(f"  - {c.name}")
    if root_tools:
        print(f"  - Main_Workflow (root-level tools)")
    print("\nPlease set the 'container_name' widget to one of the names above.")
    dbutils.notebook.exit("CONTAINER_NOT_FOUND")

item = items_to_convert[0]
selected_container = item["container"]
selected_context = item["context"]
is_root = item["is_root"]

tag = " [ROOT-LEVEL]" if is_root else ""
print(f"Selected container: {selected_container.name}{tag}")
print(f"  Tools: {len(selected_context['tools'])}")
print(f"  Internal connections: {len(selected_context['internal_connections'])}")
print(f"  External inputs: {len(selected_context['external_inputs'])}")
print(f"  External outputs: {len(selected_context['external_outputs'])}")
print(f"  Sub-containers: {len(selected_context['sub_containers'])}")

# COMMAND ----------

# -- Dry-run mode --
if RUN_MODE == "dry_run":
    print(f"\nDry-run mode -- detailed context for container '{selected_container.name}':\n")
    print(f"  Tools ({len(selected_context['tools'])}):")
    for tool in sorted(selected_context["tools"], key=lambda t: t.tool_id):
        print(f"    - Tool {tool.tool_id}: {tool.tool_type} | {tool.annotation or '(no annotation)'}")

    print(f"\n  Internal connections ({len(selected_context['internal_connections'])}):")
    for conn in selected_context["internal_connections"]:
        origin = workflow.get_tool(conn.origin_tool_id)
        dest = workflow.get_tool(conn.dest_tool_id)
        origin_name = f"{origin.tool_type}" if origin else "?"
        dest_name = f"{dest.tool_type}" if dest else "?"
        print(f"    Tool {conn.origin_tool_id} ({origin_name}) --[{conn.origin_connection}]--> Tool {conn.dest_tool_id} ({dest_name})")

    print(f"\n  External inputs ({len(selected_context['external_inputs'])}):")
    for conn in selected_context["external_inputs"]:
        src = workflow.get_tool(conn.origin_tool_id)
        src_name = f"{src.tool_type} '{src.annotation}'" if src else "?"
        src_container = workflow._find_tool_container_name(conn.origin_tool_id)
        print(f"    From Tool {conn.origin_tool_id} ({src_name}, container='{src_container}') --> Tool {conn.dest_tool_id}")

    print(f"\n  External outputs ({len(selected_context['external_outputs'])}):")
    for conn in selected_context["external_outputs"]:
        dest = workflow.get_tool(conn.dest_tool_id)
        dest_name = f"{dest.tool_type} '{dest.annotation}'" if dest else "?"
        dest_container = workflow._find_tool_container_name(conn.dest_tool_id)
        print(f"    Tool {conn.origin_tool_id} --> Tool {conn.dest_tool_id} ({dest_name}, container='{dest_container}')")

    dbutils.notebook.exit("DRY_RUN_COMPLETE")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 7: Generate PySpark Code via Claude AI

# COMMAND ----------

# -- Set up output directory --
try:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Output directory: {OUTPUT_DIR}")
except Exception:
    try:
        dbutils.fs.mkdirs(OUTPUT_DIR)
        print(f"Output directory (DBFS): {OUTPUT_DIR}")
    except Exception as e:
        print(f"WARNING: Could not create output directory: {e}")

# COMMAND ----------

# -- Load source tables config --
source_tables = None
if SOURCE_TABLES_JSON:
    try:
        content = dbutils.fs.head(SOURCE_TABLES_JSON, 65536)
        source_tables = json.loads(content)
        print(f"Loaded source tables config: {len(source_tables)} entries")
    except Exception as e:
        print(f"WARNING: Could not load source tables config: {e}")

# COMMAND ----------

# -- Initialize Claude AI generator --
api_key = dbutils.secrets.get(scope=SECRET_SCOPE, key=SECRET_KEY)

generator = ClaudeCodeGenerator(
    api_key=api_key,
    model=MODEL,
    max_retries=MAX_RETRIES,
    source_tables_config=source_tables,
)

print(f"Claude AI generator initialized (model={MODEL}, retries={MAX_RETRIES})")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Generate Code for Selected Container

# COMMAND ----------

print(f"\n{'='*60}")
print(f"Generating: {selected_container.name}{tag}")
print(f"  Tools: {len(selected_context['tools'])}")
print(f"  Connections: {len(selected_context['internal_connections'])}")
print(f"{'='*60}")

t0 = time.time()
results = []

try:
    code = generator.generate_container_code(
        container=selected_container,
        context=selected_context,
        workflow=workflow,
    )
    gen_time = time.time() - t0

    # Build safe filename from container name
    safe_name = selected_container.name.lower().replace(" ", "_").replace("-", "_")
    safe_name = "".join(c for c in safe_name if c.isalnum() or c == "_")
    output_file = os.path.join(OUTPUT_DIR, f"{safe_name}.py")

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(code)

    results.append({
        "container": selected_container.name,
        "file": output_file,
        "status": "Success",
        "time": f"{gen_time:.1f}s",
        "tools": len(selected_context["tools"]),
    })
    print(f"   Written to: {output_file} ({gen_time:.1f}s)")

except Exception as e:
    gen_time = time.time() - t0
    results.append({
        "container": selected_container.name,
        "file": "-",
        "status": f"Failed: {str(e)[:80]}",
        "time": f"{gen_time:.1f}s",
        "tools": len(selected_context["tools"]),
    })
    print(f"   Failed: {e}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 8: Conversion Summary

# COMMAND ----------

print_summary(results)

# COMMAND ----------

# -- Rich HTML summary for Databricks UI --
try:
    display_summary_table(results)
except Exception:
    pass

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 9: View Generated File

# COMMAND ----------

print(f"\nGenerated files in: {OUTPUT_DIR}")
print("-" * 60)
try:
    for f in sorted(os.listdir(OUTPUT_DIR)):
        if f.endswith(".py"):
            filepath = os.path.join(OUTPUT_DIR, f)
            size = os.path.getsize(filepath)
            print(f"  {f:<40s} ({size:,} bytes)")
except Exception as e:
    print(f"  Could not list files: {e}")

# COMMAND ----------

# -- Return summary as notebook exit value --
success_count = sum(1 for r in results if "Success" in r["status"])
container_label = selected_container.name if CONTAINER_NAME else "none"
dbutils.notebook.exit(f"COMPLETE: container='{container_label}' | {success_count}/1 converted successfully")
