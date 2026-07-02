"""
Migration job validation gate.

Runs as the final task of ``alteryx_migration_job`` (see
``resources/alteryx_migration_job.yml``): reads the batch conversion summary
produced by the Skill Mode notebook and fails the job when any workflow
conversion finished in FAIL/ERROR state, so bad notebooks never reach the
target catalog silently.

Usage: python job_validate.py <output_dir>
"""

import json
import sys
from pathlib import Path


def main(output_dir: str) -> int:
    out = Path(output_dir)
    candidates = [out / "conversion_summary.json", out / "batch_summary.json"]
    summary_file = next((c for c in candidates if c.exists()), None)
    if summary_file is None:
        print(f"ERROR: no conversion summary found under {out} "
              f"(expected one of {[c.name for c in candidates]})")
        return 1

    payload = json.loads(summary_file.read_text())
    entries = (
        [{"workflow": k, **v} for k, v in payload.items()]
        if isinstance(payload, dict) else payload
    )

    failed = []
    for entry in entries:
        status = str(entry.get("status", ""))
        print(f"  {entry.get('workflow', '?'):<50} {status}")
        if status.startswith(("FAIL", "ERROR")):
            failed.append(entry)

    if failed:
        print(f"\nVALIDATION GATE FAILED: {len(failed)}/{len(entries)} workflow(s) "
              f"did not converge — inspect the error report cell in each notebook.")
        return 1

    print(f"\nValidation gate passed: {len(entries)} workflow(s) converted cleanly.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "./output"))
