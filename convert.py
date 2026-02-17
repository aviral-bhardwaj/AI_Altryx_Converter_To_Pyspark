#!/usr/bin/env python3
"""
Alteryx to PySpark AI Converter - Main Entry Point
===================================================
Uses Claude AI to convert Alteryx .yxmd workflows into production-ready
PySpark code, generating one file per container (module).

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
        help="Convert only a specific container by name (e.g., 'mod_provider'). "
             "If omitted, all containers are converted.",
    )
    parser.add_argument(
        "--list-containers", "-l",
        action="store_true",
        help="List all containers in the workflow and exit.",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("CLAUDE_MODEL", "claude-sonnet-4-20250514"),
        help="Claude model to use (default: claude-sonnet-4-20250514)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse and extract containers but don't call Claude AI.",
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
        help="Max retries per container on AI generation failure (default: 2)",
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
        print(f"❌ Workflow file not found: {workflow_path}")
        sys.exit(1)

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key and not args.dry_run and not args.list_containers \
            and not args.analyze and not args.generate_config:
        print("❌ ANTHROPIC_API_KEY environment variable is required.")
        print("   Set it with: export ANTHROPIC_API_KEY=sk-ant-...")
        sys.exit(1)

    # ── Parse the workflow ───────────────────────────────────────────
    print(f"📂 Parsing workflow: {workflow_path.name}")
    t0 = time.time()
    parser_obj = AlteryxWorkflowParser(str(workflow_path))
    workflow = parser_obj.parse()
    parse_time = time.time() - t0
    print(f"   ✅ Parsed in {parse_time:.1f}s")
    print(f"   📦 Containers: {len(workflow.containers)}")
    print(f"   🔧 Total tools: {len(workflow.all_tools)}")
    print(f"   🔗 Total connections: {len(workflow.connections)}")

    # ── List containers mode ─────────────────────────────────────────
    if args.list_containers:
        print("\n📋 Containers found:")
        for i, container in enumerate(workflow.containers, 1):
            tool_count = len(container.child_tools)
            print(f"   {i:2d}. {container.name:<40s} ({tool_count} tools)")
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
            print("\n📊 Workflow Analysis:")
            for container in workflow.containers:
                context = workflow.get_container_context(container.tool_id)
                print(f"\n  Container: {container.name}")
                print(f"    Tools: {len(container.child_tools)}")
                print(f"    Inputs: {len(context['external_inputs'])}")
                print(f"    Outputs: {len(context['external_outputs'])}")
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
            print(f"\n📝 Auto-generated config written to: {config_file}")
            print(config_yaml)
        except ImportError:
            print("❌ Could not load workflow analyzer")
        sys.exit(0)

    # ── Filter to specific container if requested ────────────────────
    containers_to_convert = workflow.containers
    if args.container:
        containers_to_convert = [
            c for c in workflow.containers
            if c.name.lower() == args.container.lower()
        ]
        if not containers_to_convert:
            print(f"❌ Container '{args.container}' not found.")
            print("   Use --list-containers to see available containers.")
            sys.exit(1)

    # ── Dry-run mode ─────────────────────────────────────────────────
    if args.dry_run:
        print("\n🔍 Dry-run mode — extracted context for each container:\n")
        for container in containers_to_convert:
            context = workflow.get_container_context(container.tool_id)
            print(f"── {container.name} ──")
            print(f"   Tools: {len(container.child_tools)}")
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
    for i, container in enumerate(containers_to_convert, 1):
        print(f"\n{'='*60}")
        print(f"🤖 [{i}/{len(containers_to_convert)}] Generating: {container.name}")
        print(f"{'='*60}")

        context = workflow.get_container_context(container.tool_id)
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
                "status": "✅ Success",
                "time": f"{gen_time:.1f}s",
                "tools": len(container.child_tools),
            })
            print(f"   ✅ Written to: {output_file} ({gen_time:.1f}s)")

        except Exception as e:
            gen_time = time.time() - t0
            results.append({
                "container": container.name,
                "file": "—",
                "status": f"❌ Failed: {str(e)[:80]}",
                "time": f"{gen_time:.1f}s",
                "tools": len(container.child_tools),
            })
            print(f"   ❌ Failed: {e}")

    # ── Summary ──────────────────────────────────────────────────────
    print_summary(results)


if __name__ == "__main__":
    main()
