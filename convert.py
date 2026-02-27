#!/usr/bin/env python3
"""
Alteryx to PySpark Converter - Main Entry Point
=================================================
Converts Alteryx .yxmd workflows into production-ready PySpark code.

Supports two modes:
  - **ai** (default): Uses Claude AI for intelligent code generation
  - **deterministic**: Uses rule-based converters (no API key needed)

Processes ALL tools in the workflow — both those inside ToolContainer
elements and those at the root level — as one combined pipeline.

Usage:
    python convert.py <workflow.yxmd> [--mode deterministic] [--output-dir ./output]
    python convert.py <workflow.yxmd> --mode ai  # requires ANTHROPIC_API_KEY
    python convert.py <workflow.yxmd> --validate --validate-target-columns col1,col2

Environment:
    ANTHROPIC_API_KEY  - Required for AI mode. Your Claude API key.
    CLAUDE_MODEL       - Optional. Default: claude-sonnet-4-20250514
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

from src.parser import AlteryxWorkflowParser
from src.utils import print_banner, print_summary


def main():
    parser = argparse.ArgumentParser(
        description="Convert Alteryx .yxmd workflows to PySpark"
    )
    parser.add_argument(
        "workflow",
        help="Path to the .yxmd Alteryx workflow file",
    )
    parser.add_argument(
        "--output-dir", "-o",
        default="./output",
        help="Directory to write generated PySpark files (default: ./output)",
    )
    parser.add_argument(
        "--mode", "-m",
        choices=["ai", "deterministic"],
        default="deterministic",
        help="Generation mode: 'ai' (Claude AI) or 'deterministic' (rule-based). Default: deterministic",
    )
    parser.add_argument(
        "--list-containers", "-l",
        action="store_true",
        help="List all containers and root-level tools in the workflow and exit.",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("CLAUDE_MODEL", "claude-sonnet-4-20250514"),
        help="Claude model to use in AI mode (default: claude-sonnet-4-20250514)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse and show workflow structure without generating code.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=2,
        help="Max retries on AI generation failure (default: 2)",
    )
    parser.add_argument(
        "--source-tables-config",
        default=None,
        help="Optional JSON file mapping Alteryx input tool IDs/names to "
             "Databricks catalog.schema.table paths.",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Run validation after generation. Requires --validate-target-columns.",
    )
    parser.add_argument(
        "--validate-target-columns",
        default=None,
        help="Comma-separated list of expected target columns for validation.",
    )
    parser.add_argument(
        "--validate-expected-rows",
        type=int,
        default=None,
        help="Expected row count for validation.",
    )
    args = parser.parse_args()

    print_banner()

    # ── Validate inputs ──────────────────────────────────────────────
    workflow_path = Path(args.workflow)
    if not workflow_path.exists():
        print(f"  Workflow file not found: {workflow_path}")
        sys.exit(1)

    if args.mode == "ai":
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key and not args.dry_run and not args.list_containers:
            print("  ANTHROPIC_API_KEY environment variable is required for AI mode.")
            print("   Set it with: export ANTHROPIC_API_KEY=sk-ant-...")
            print("   Or use --mode deterministic for rule-based conversion.")
            sys.exit(1)

    # ── Parse the workflow ───────────────────────────────────────────
    print(f"Parsing workflow: {workflow_path.name}")
    t0 = time.time()
    parser_obj = AlteryxWorkflowParser(str(workflow_path))
    workflow = parser_obj.parse()
    parse_time = time.time() - t0

    root_tools = workflow.get_root_tools()
    container_tool_count = sum(len(c.child_tools) for c in workflow.containers)

    print(f"   Parsed in {parse_time:.1f}s")
    print(f"   Containers: {len(workflow.containers)}")
    print(f"   Tools in containers: {container_tool_count}")
    print(f"   Root-level tools: {len(root_tools)}")
    print(f"   Total tools: {len(workflow.all_tools)}")
    print(f"   Total connections: {len(workflow.connections)}")

    # ── List containers mode ─────────────────────────────────────────
    if args.list_containers:
        print("\nWorkflow structure:")
        print("-" * 60)
        idx = 1
        for container in workflow.containers:
            tool_count = len(container.child_tools)
            disabled_tag = " [DISABLED]" if container.disabled else ""
            print(f"   {idx:2d}. Container: {container.name:<35s} ({tool_count} tools){disabled_tag}")
            idx += 1
        if root_tools:
            print(f"   {idx:2d}. Root-level tools                        ({len(root_tools)} tools)")
        print("-" * 60)
        print(f"   Total: {len(workflow.all_tools)} tools, {len(workflow.connections)} connections")
        print(f"\n   All tools will be converted as a single unified notebook.")
        sys.exit(0)

    # ── Dry-run mode ─────────────────────────────────────────────────
    unified_ctx = workflow.get_unified_context()

    if args.dry_run:
        print("\nDry-run mode -- unified workflow context:\n")
        print(f"  Total tools: {len(unified_ctx['tools'])}")
        print(f"  Total connections: {len(unified_ctx['internal_connections'])}")
        print(f"  Containers: {len(unified_ctx['sub_containers'])}")
        for sc in unified_ctx["sub_containers"]:
            print(f"    - {sc.name} ({len(sc.child_tools)} tools)")
        print(f"  Text inputs: {len(unified_ctx['text_input_data'])}")
        print()
        print("  All tools will be converted as a single unified notebook.")
        sys.exit(0)

    # ── Load source tables config ────────────────────────────────────
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    source_tables = None
    if args.source_tables_config:
        with open(args.source_tables_config) as f:
            source_tables = json.load(f)

    # ── Determine output filename ────────────────────────────────────
    workflow_name = workflow_path.stem
    safe_name = workflow_name.lower().replace(" ", "_").replace("-", "_")
    safe_name = "".join(c for c in safe_name if c.isalnum() or c == "_")

    print(f"\n{'='*60}")
    print(f"Mode: {args.mode}")
    print(f"Generating unified notebook: {safe_name}.py")
    print(f"  Tools: {len(unified_ctx['tools'])}")
    print(f"  Connections: {len(unified_ctx['internal_connections'])}")
    print(f"{'='*60}")

    t0 = time.time()
    results = []

    try:
        if args.mode == "ai":
            code = _generate_ai(
                workflow=workflow,
                unified_ctx=unified_ctx,
                api_key=os.environ.get("ANTHROPIC_API_KEY"),
                model=args.model,
                max_retries=args.max_retries,
                source_tables=source_tables,
            )
        else:
            code = _generate_deterministic(
                workflow=workflow,
                unified_ctx=unified_ctx,
                workflow_name=workflow_name,
                source_tables=source_tables,
            )

        gen_time = time.time() - t0

        output_file = output_dir / f"{safe_name}.py"
        output_file.write_text(code, encoding="utf-8")

        results.append({
            "container": f"{workflow_name} (unified)",
            "file": str(output_file),
            "status": "Success",
            "time": f"{gen_time:.1f}s",
            "tools": len(unified_ctx["tools"]),
        })
        print(f"   Written to: {output_file} ({gen_time:.1f}s)")

    except Exception as e:
        gen_time = time.time() - t0
        results.append({
            "container": f"{workflow_name} (unified)",
            "file": "-",
            "status": f"Failed: {str(e)[:80]}",
            "time": f"{gen_time:.1f}s",
            "tools": len(unified_ctx["tools"]),
        })
        print(f"   Failed: {e}")

    # ── Summary ──────────────────────────────────────────────────────
    print_summary(results)

    # ── Validation (optional) ────────────────────────────────────────
    if args.validate:
        _run_validation(args, workflow, unified_ctx, output_dir, safe_name)


def _generate_ai(workflow, unified_ctx, api_key, model, max_retries, source_tables):
    """Generate PySpark code using Claude AI."""
    from src.ai_generator import ClaudeCodeGenerator

    generator = ClaudeCodeGenerator(
        api_key=api_key,
        model=model,
        max_retries=max_retries,
        source_tables_config=source_tables,
    )
    return generator.generate_container_code(
        container=unified_ctx["container"],
        context=unified_ctx,
        workflow=workflow,
    )


def _generate_deterministic(workflow, unified_ctx, workflow_name, source_tables):
    """Generate PySpark code using deterministic rule-based converters."""
    from src.pyspark_generator import PySparkCodeGenerator

    generator = PySparkCodeGenerator(source_tables_config=source_tables)
    return generator.generate(
        workflow=workflow,
        workflow_name=workflow_name,
        context=unified_ctx,
    )


def _run_validation(args, workflow, unified_ctx, output_dir, safe_name):
    """Run the validation framework and save the report."""
    from src.validation import WorkflowValidator

    print(f"\n{'='*60}")
    print("Running validation...")
    print(f"{'='*60}")

    validator = WorkflowValidator(workflow_name=safe_name)

    # Build source columns from parsed tools
    source_columns = []
    for tool in unified_ctx["tools"]:
        pc = tool.parsed_config or {}
        fields = pc.get("fields", [])
        for f in fields:
            name = f.get("name", "")
            if name and name not in source_columns:
                source_columns.append(name)

    # Target columns from CLI arg
    target_columns = None
    if args.validate_target_columns:
        target_columns = [c.strip() for c in args.validate_target_columns.split(",")]

    report = validator.validate(
        source_columns=source_columns or None,
        target_columns=target_columns,
        source_row_count=args.validate_expected_rows,
        target_row_count=args.validate_expected_rows,
    )

    # Save reports
    report_json = output_dir / f"{safe_name}_validation.json"
    report_json.write_text(validator.to_json(report), encoding="utf-8")
    print(f"   JSON report: {report_json}")

    report_html = output_dir / f"{safe_name}_validation.html"
    report_html.write_text(validator.to_html(report), encoding="utf-8")
    print(f"   HTML report: {report_html}")

    print(f"\n   Overall status: {report.status.value}")
    for rec in report.recommendations:
        print(f"   - {rec}")


if __name__ == "__main__":
    main()
