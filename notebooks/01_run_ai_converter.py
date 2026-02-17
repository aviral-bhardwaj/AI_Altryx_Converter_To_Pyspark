# Databricks notebook source
# MAGIC %md
# MAGIC # Alteryx to PySpark - AI-Powered Converter
# MAGIC
# MAGIC Converts Alteryx `.yxmd` workflows into production-ready PySpark code using **Claude AI**.
# MAGIC Generates one output file per container (module).
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
# MAGIC 2. Click **Run All** or run cells individually
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
    "3. Container Name (blank = all)"
)

dbutils.widgets.dropdown(
    "run_mode",
    "convert",
    ["convert", "list_containers", "dry_run"],
    "4. Run Mode"
)

dbutils.widgets.dropdown(
    "model",
    "claude-sonnet-4-20250514",
    ["claude-sonnet-4-20250514", "claude-opus-4-20250514", "claude-haiku-4-5-20251001"],
    "5. Claude Model"
)

dbutils.widgets.dropdown(
    "max_retries",
    "2",
    ["0", "1", "2", "3", "4", "5"],
    "6. Max Retries"
)

dbutils.widgets.text(
    "secret_scope",
    "alteryx-converter",
    "7. Secret Scope Name"
)

dbutils.widgets.text(
    "secret_key",
    "anthropic-api-key",
    "8. Secret Key Name"
)

dbutils.widgets.text(
    "source_tables_json",
    "",
    "9. Source Tables JSON Path (optional)"
)

# COMMAND ----------

# -- Read widget values --

WORKFLOW_PATH = dbutils.widgets.get("workflow_path")
OUTPUT_DIR = dbutils.widgets.get("output_dir")
CONTAINER_NAME = dbutils.widgets.get("container_name").strip() or None
RUN_MODE = dbutils.widgets.get("run_mode")
MODEL = dbutils.widgets.get("model")
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
    print(f"  Workflow:  {WORKFLOW_PATH}")
    print(f"  Output:    {OUTPUT_DIR}")
    print(f"  Mode:      {RUN_MODE}")
    print(f"  Model:     {MODEL}")
    print(f"  Container: {CONTAINER_NAME or '(all)'}")

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
    print(f"  * Root-level tools will be processed as 'Main_Workflow' module")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 5: List Modules

# COMMAND ----------

print("\nModules found in workflow:")
print("-" * 60)
idx = 1
for container in workflow.containers:
    tool_count = len(container.child_tools)
    disabled_tag = " [DISABLED]" if container.disabled else ""
    print(f"  {idx:2d}. {container.name:<40s} ({tool_count} tools){disabled_tag}")
    idx += 1
if root_tools:
    print(f"  {idx:2d}. {'Main_Workflow':<40s} ({len(root_tools)} tools) [ROOT-LEVEL]")
print("-" * 60)

# COMMAND ----------

if RUN_MODE == "list_containers":
    print("\nRun mode: list_containers -- listing complete.")
    dbutils.notebook.exit("LISTED_CONTAINERS")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 6: Build Conversion List (Containers + Root-Level Tools)

# COMMAND ----------

# Build the list of (container, context) items to convert
items_to_convert = []

# 1) Named containers
for container in workflow.containers:
    ctx = workflow.get_container_context(container.tool_id)
    items_to_convert.append({
        "container": container,
        "context": ctx,
        "is_root": False,
    })

# 2) Root-level tools (outside any container)
root_ctx = workflow.get_root_context()
if root_ctx:
    items_to_convert.append({
        "container": root_ctx["container"],
        "context": root_ctx,
        "is_root": True,
    })

# Filter if requested
if CONTAINER_NAME:
    items_to_convert = [
        item for item in items_to_convert
        if item["container"].name.lower() == CONTAINER_NAME.lower()
    ]
    if not items_to_convert:
        print(f"ERROR: Module '{CONTAINER_NAME}' not found.")
        print("Available modules:")
        for c in workflow.containers:
            print(f"  - {c.name}")
        if root_tools:
            print(f"  - Main_Workflow (root-level)")
        dbutils.notebook.exit("MODULE_NOT_FOUND")
    print(f"Filtered to module: '{CONTAINER_NAME}'")
else:
    print(f"Converting all {len(items_to_convert)} modules")

# COMMAND ----------

# -- Dry-run mode --
if RUN_MODE == "dry_run":
    print("\nDry-run mode -- extracted context for each module:\n")
    for item in items_to_convert:
        container = item["container"]
        context = item["context"]
        tag = " [ROOT-LEVEL]" if item["is_root"] else ""
        print(f"-- {container.name}{tag} --")
        print(f"   Tools: {len(context['tools'])}")
        print(f"   Internal connections: {len(context['internal_connections'])}")
        print(f"   External inputs: {len(context['external_inputs'])}")
        print(f"   External outputs: {len(context['external_outputs'])}")
        print(f"   Sub-containers: {len(context['sub_containers'])}")
        print()
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
# MAGIC ### Generate Code for Each Module

# COMMAND ----------

results = []
total = len(items_to_convert)

for i, item in enumerate(items_to_convert, 1):
    container = item["container"]
    context = item["context"]
    tag = " [ROOT-LEVEL]" if item["is_root"] else ""

    print(f"\n{'='*60}")
    print(f"[{i}/{total}] Generating: {container.name}{tag}")
    print(f"{'='*60}")

    t0 = time.time()

    try:
        code = generator.generate_container_code(
            container=container,
            context=context,
            workflow=workflow,
        )
        gen_time = time.time() - t0

        # Build safe filename
        safe_name = container.name.lower().replace(" ", "_").replace("-", "_")
        safe_name = "".join(c for c in safe_name if c.isalnum() or c == "_")
        output_file = os.path.join(OUTPUT_DIR, f"{safe_name}.py")

        # Write output file
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(code)

        results.append({
            "container": container.name,
            "file": output_file,
            "status": "Success",
            "time": f"{gen_time:.1f}s",
            "tools": len(context["tools"]),
        })
        print(f"   Written to: {output_file} ({gen_time:.1f}s)")

    except Exception as e:
        gen_time = time.time() - t0
        results.append({
            "container": container.name,
            "file": "-",
            "status": f"Failed: {str(e)[:80]}",
            "time": f"{gen_time:.1f}s",
            "tools": len(context["tools"]),
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
# MAGIC ## Step 9: View Generated Files

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
total_count = len(results)
dbutils.notebook.exit(f"COMPLETE: {success_count}/{total_count} modules converted successfully")
