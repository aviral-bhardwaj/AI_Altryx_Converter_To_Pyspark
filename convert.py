#!/usr/bin/env python3
"""
Alteryx to PySpark AI Converter - Main Entry Point
===================================================
Uses Claude AI to convert Alteryx .yxmd workflows into production-ready
PySpark code, generating one file per module (container + root-level tools).

Handles both:
  - Container-based tools (inside ToolContainer elements)
  - Root-level tools (outside any container, treated as "Main_Workflow")

Usage:
    python convert.py <workflow.yxmd> [--output-dir ./output] [--container mod_provider]

Environment:
    ANTHROPIC_API_KEY  - Required. Your Claude API key.
    CLAUDE_MODEL       - Optional. Default: claude-sonnet-4-20250514
"""

import argparse
import os
import sys
import time
from pathlib import Path

from src.parser import AlteryxWorkflowParser
from src.ai_generator import ClaudeCodeGenerator
from src.utils import print_banner, print_summary


def _build_conversion_list(workflow, container_filter=None):
    """
    Build the list of (container, context) pairs to convert.

    Returns a list of dicts with keys:
      - container: Container object (real or virtual)
      - context: dict from get_container_context() or get_root_context()
      - is_root: bool indicating if this is the root-level virtual container

    If container_filter is set, only matching items are returned.
    """
    items = []

    # 1) Named containers from the workflow
    for container in workflow.containers:
        ctx = workflow.get_container_context(container.tool_id)
        items.append({
            "container": container,
            "context": ctx,
            "is_root": False,
        })

    # 2) Root-level tools (outside any container)
    root_ctx = workflow.get_root_context()
    if root_ctx:
        root_container = root_ctx["container"]
        items.append({
            "container": root_container,
            "context": root_ctx,
            "is_root": True,
        })

    # Filter if requested
    if container_filter:
        filtered = [
            item for item in items
            if item["container"].name.lower() == container_filter.lower()
        ]
        return filtered

    return items


def main():
    parser = argparse.ArgumentParser(
        description="Convert Alteryx .yxmd workflows to PySpark using Claude AI"
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
        "--container", "-c",
        default=None,
        help="Convert only a specific module by name (e.g., 'mod_provider' or "
             "'Main_Workflow' for root-level tools). If omitted, all modules "
             "are converted.",
    )
    parser.add_argument(
        "--list-containers", "-l",
        action="store_true",
        help="List all modules (containers + root-level tools) and exit.",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("CLAUDE_MODEL", "claude-sonnet-4-20250514"),
        help="Claude model to use (default: claude-sonnet-4-20250514)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse and extract all modules but don't call Claude AI.",
    )
    parser.add_argument(
        "--analyze",
        action="store_true",
        help="Autonomously analyze the workflow and print a detailed report "
             "with complexity assessment, data sources, and suggested config.",
    )
    parser.add_argument(
        "--generate-config",
        action="store_true",
        help="Auto-generate a YAML config file for the workflow.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=2,
        help="Max retries per module on AI generation failure (default: 2)",
    )
    parser.add_argument(
        "--source-tables-config",
        default=None,
        help="Optional JSON file mapping Alteryx input tool IDs/names to "
             "Databricks catalog.schema.table paths.",
    )
    args = parser.parse_args()

    print_banner()

    # ── Validate inputs ──────────────────────────────────────────────
    workflow_path = Path(args.workflow)
    if not workflow_path.exists():
        print(f"  Workflow file not found: {workflow_path}")
        sys.exit(1)

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key and not args.dry_run and not args.list_containers \
            and not args.analyze and not args.generate_config:
        print("  ANTHROPIC_API_KEY environment variable is required.")
        print("   Set it with: export ANTHROPIC_API_KEY=sk-ant-...")
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

    if root_tools:
        print(f"   * Root-level tools will be processed as 'Main_Workflow' module")

    # ── List containers mode ─────────────────────────────────────────
    if args.list_containers:
        print("\nModules found:")
        idx = 1
        for container in workflow.containers:
            tool_count = len(container.child_tools)
            disabled_tag = " [DISABLED]" if container.disabled else ""
            print(f"   {idx:2d}. {container.name:<40s} ({tool_count} tools){disabled_tag}")
            idx += 1
        if root_tools:
            print(f"   {idx:2d}. {'Main_Workflow':<40s} ({len(root_tools)} tools) [ROOT-LEVEL]")
        sys.exit(0)

    # ── Autonomous analysis mode ──────────────────────────────────────
    if args.analyze:
        try:
            from alteryx_pyspark_converter.common.workflow_analyzer import WorkflowAnalyzer
            analyzer = WorkflowAnalyzer(workflow)
            report = analyzer.print_report()
            print(report)
        except ImportError:
            # Minimal analysis if the analyzer isn't available
            print("\nWorkflow Analysis:")
            for container in workflow.containers:
                context = workflow.get_container_context(container.tool_id)
                print(f"\n  Container: {container.name}")
                print(f"    Tools: {len(container.child_tools)}")
                print(f"    Inputs: {len(context['external_inputs'])}")
                print(f"    Outputs: {len(context['external_outputs'])}")
            if root_tools:
                root_ctx = workflow.get_root_context()
                if root_ctx:
                    print(f"\n  Main_Workflow (root-level tools):")
                    print(f"    Tools: {len(root_tools)}")
                    print(f"    Internal connections: {len(root_ctx['internal_connections'])}")
                    print(f"    Inputs from containers: {len(root_ctx['external_inputs'])}")
                    print(f"    Outputs to containers: {len(root_ctx['external_outputs'])}")
        sys.exit(0)

    # ── Auto-generate config mode ─────────────────────────────────────
    if args.generate_config:
        try:
            from alteryx_pyspark_converter.common.workflow_analyzer import WorkflowAnalyzer
            analyzer = WorkflowAnalyzer(workflow)
            container_name = args.container or ""
            config_yaml = analyzer.generate_config_yaml(container_name)
            config_file = Path(args.output_dir) / "auto_config.yaml"
            config_file.parent.mkdir(parents=True, exist_ok=True)
            config_file.write_text(config_yaml, encoding="utf-8")
            print(f"\nAuto-generated config written to: {config_file}")
            print(config_yaml)
        except ImportError:
            print("Could not load workflow analyzer")
        sys.exit(0)

    # ── Build the conversion list (containers + root-level) ──────────
    items_to_convert = _build_conversion_list(workflow, args.container)
    if args.container and not items_to_convert:
        print(f"  Module '{args.container}' not found.")
        print("   Use --list-containers to see available modules.")
        sys.exit(1)

    # ── Dry-run mode ─────────────────────────────────────────────────
    if args.dry_run:
        print("\nDry-run mode -- extracted context for each module:\n")
        for item in items_to_convert:
            container = item["container"]
            context = item["context"]
            is_root = item["is_root"]
            tag = " [ROOT-LEVEL]" if is_root else ""
            print(f"-- {container.name}{tag} --")
            print(f"   Tools: {len(context['tools'])}")
            print(f"   Internal connections: {len(context['internal_connections'])}")
            print(f"   External inputs: {len(context['external_inputs'])}")
            print(f"   External outputs: {len(context['external_outputs'])}")
            print(f"   Sub-containers: {len(context['sub_containers'])}")
            print()
        sys.exit(0)

    # ── Generate PySpark code via Claude AI ──────────────────────────
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load source tables config if provided
    source_tables = None
    if args.source_tables_config:
        import json
        with open(args.source_tables_config) as f:
            source_tables = json.load(f)

    generator = ClaudeCodeGenerator(
        api_key=api_key,
        model=args.model,
        max_retries=args.max_retries,
        source_tables_config=source_tables,
    )

    results = []
    total = len(items_to_convert)

    for i, item in enumerate(items_to_convert, 1):
        container = item["container"]
        context = item["context"]
        is_root = item["is_root"]
        tag = " [ROOT-LEVEL]" if is_root else ""

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

            # Write output file
            safe_name = container.name.lower().replace(" ", "_").replace("-", "_")
            safe_name = "".join(c for c in safe_name if c.isalnum() or c == "_")
            output_file = output_dir / f"{safe_name}.py"
            output_file.write_text(code, encoding="utf-8")

            results.append({
                "container": container.name,
                "file": str(output_file),
                "status": "Success",
                "time": f"{gen_time:.1f}s",
                "tools": len(context["tools"]),
            })
            print(f"   Written to: {output_file} ({gen_time:.1f}s)")

        except Exception as e:
            gen_time = time.time() - t0
            results.append({
                "container": container.name,
                "file": "—",
                "status": f"Failed: {str(e)[:80]}",
                "time": f"{gen_time:.1f}s",
                "tools": len(context["tools"]),
            })
            print(f"   Failed: {e}")

    # ── Summary ──────────────────────────────────────────────────────
    print_summary(results)


if __name__ == "__main__":
    main()
