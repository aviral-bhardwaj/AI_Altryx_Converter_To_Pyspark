# Databricks notebook source
# MAGIC %md
# MAGIC # Claude AI Code Generator
# MAGIC Sends container context to Claude API and returns generated PySpark code.
# MAGIC Handles retries, validation, and code extraction.
# MAGIC
# MAGIC **Usage:** This notebook is imported by the orchestrator via `%run ./ai_generator`
# MAGIC
# MAGIC **Requires:** `%run ./context_builder` must be executed first (which itself runs `%run ./models`).

# COMMAND ----------

# MAGIC %run ./context_builder

# COMMAND ----------

import json
import logging
import re
import time
from typing import Optional

logger = logging.getLogger("alteryx_converter.ai_generator")

# COMMAND ----------

# -- Try to import anthropic SDK; fall back to urllib --
try:
    import anthropic
    HAS_ANTHROPIC_SDK = True
except ImportError:
    HAS_ANTHROPIC_SDK = False
    import urllib.request
    import urllib.error

# COMMAND ----------

class ClaudeCodeGenerator:
    """Generates PySpark code for Alteryx containers using Claude AI."""

    def __init__(
        self,
        api_key: str,
        model: str = "claude-sonnet-4-20250514",
        max_retries: int = 2,
        source_tables_config: Optional[dict] = None,
    ):
        self.api_key = api_key
        self.model = model
        self.max_retries = max_retries
        self.source_tables_config = source_tables_config

        if HAS_ANTHROPIC_SDK:
            self.client = anthropic.Anthropic(api_key=api_key)
            logger.info("Using anthropic SDK")
        else:
            self.client = None
            logger.info("Using urllib fallback (pip install anthropic for better experience)")

    def generate_container_code(
        self,
        container: Container,
        context: dict,
        workflow: Workflow,
    ) -> str:
        """
        Generate PySpark code for a single container.

        Args:
            container: The container to generate code for.
            context: Output of workflow.get_container_context().
            workflow: The full parsed workflow (for cross-references).

        Returns:
            Generated PySpark code as a string.
        """
        # Build the prompt
        container_description = build_container_prompt(
            container=container,
            context=context,
            workflow=workflow,
            source_tables_config=self.source_tables_config,
        )

        system_prompt = build_system_prompt()

        # Count tools and connections for the checklist
        num_tools = len(context.get("tools", []))
        num_ext_inputs = len(context.get("external_inputs", []))
        num_int_conns = len(context.get("internal_connections", []))

        user_message = f"""Convert the following Alteryx container to a complete PySpark Databricks notebook.

{container_description}

## CHECKLIST - Your generated code MUST include:
1. First line MUST be: # Databricks notebook source
2. Use # COMMAND ---------- between every cell
3. Use # MAGIC %md header cells before each code section
4. spark.table() calls for ALL {num_ext_inputs} external source inputs
5. Transformation code for ALL {num_tools} tools listed above
6. Proper data flow following ALL {num_int_conns} internal connections
7. For each Join: use the exact join keys specified, handle post-join column drops/renames
8. For each Filter: generate the True/False outputs that are used downstream
9. For each Formula: convert ALL formula fields to .withColumn() calls
10. For each Select: drop deselected columns, apply renames
11. Final output as createOrReplaceTempView() or write statement

CRITICAL: Output must be in Databricks notebook format (NOT plain Python).
Output ONLY the code. No markdown fences, no explanations before or after."""

        # Call Claude with retries
        code = None
        last_error = None

        for attempt in range(1, self.max_retries + 2):  # +2 because range is exclusive and we start at 1
            try:
                logger.info(
                    f"Calling Claude API for '{container.name}' "
                    f"(attempt {attempt}/{self.max_retries + 1})"
                )
                print(f"   API call attempt {attempt}...")

                raw_response = self._call_claude(system_prompt, user_message)
                code = self._extract_code(raw_response)

                if code and self._validate_code(code):
                    code = self._ensure_databricks_format(code, container.name)
                    logger.info(f"Successfully generated code for '{container.name}'")
                    return code
                else:
                    last_error = "Generated code failed validation"
                    logger.warning(f"Attempt {attempt}: {last_error}")

            except Exception as e:
                last_error = str(e)
                logger.warning(f"Attempt {attempt} failed: {last_error}")
                if attempt <= self.max_retries:
                    wait = 2 ** attempt
                    print(f"   Waiting {wait}s before retry...")
                    time.sleep(wait)

        raise RuntimeError(
            f"Failed to generate code for '{container.name}' "
            f"after {self.max_retries + 1} attempts. Last error: {last_error}"
        )

    def _call_claude(self, system_prompt: str, user_message: str) -> str:
        """Call Claude API and return the text response."""
        if HAS_ANTHROPIC_SDK:
            return self._call_with_sdk(system_prompt, user_message)
        else:
            return self._call_with_urllib(system_prompt, user_message)

    def _call_with_sdk(self, system_prompt: str, user_message: str) -> str:
        """Call Claude using the anthropic Python SDK."""
        message = self.client.messages.create(
            model=self.model,
            max_tokens=16000,
            system=system_prompt,
            messages=[
                {"role": "user", "content": user_message},
            ],
        )
        # Extract text from response
        text_parts = []
        for block in message.content:
            if hasattr(block, "text"):
                text_parts.append(block.text)
        return "\n".join(text_parts)

    def _call_with_urllib(self, system_prompt: str, user_message: str) -> str:
        """Call Claude API using urllib (no SDK dependency)."""
        url = "https://api.anthropic.com/v1/messages"
        payload = json.dumps({
            "model": self.model,
            "max_tokens": 16000,
            "system": system_prompt,
            "messages": [
                {"role": "user", "content": user_message},
            ],
        }).encode("utf-8")

        req = urllib.request.Request(
            url,
            data=payload,
            headers={
                "Content-Type": "application/json",
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=300) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Claude API error {e.code}: {body}")

        text_parts = []
        for block in data.get("content", []):
            if block.get("type") == "text":
                text_parts.append(block["text"])
        return "\n".join(text_parts)

    def _extract_code(self, raw_response: str) -> str:
        """Extract Python code from Claude's response, removing markdown fences."""
        text = raw_response.strip()

        # Try to extract from ```python ... ``` blocks
        pattern = r"```(?:python)?\s*\n(.*?)```"
        matches = re.findall(pattern, text, re.DOTALL)
        if matches:
            # Return the longest match (most likely the full code)
            return max(matches, key=len).strip()

        # If no fences, check if the whole response looks like Python code
        if text.startswith(("#", '"""', "from ", "import ")):
            return text

        # Last resort: return as-is
        return text

    def _validate_code(self, code: str) -> bool:
        """Validate that the generated code is reasonable PySpark."""
        if not code or len(code) < 50:
            logger.warning("Generated code is too short")
            return False

        # Must reference spark
        if "spark" not in code.lower():
            logger.warning("Generated code doesn't appear to contain PySpark")
            return False

        # Strip Databricks magic/command lines and notebook header before syntax check
        lines = []
        for line in code.split("\n"):
            stripped = line.strip()
            if stripped.startswith("# MAGIC") or stripped.startswith("# COMMAND"):
                continue
            if stripped == "# Databricks notebook source":
                continue
            lines.append(line)
        check_code = "\n".join(lines)

        # Try to compile (syntax check)
        try:
            compile(check_code, "<generated>", "exec")
        except SyntaxError as e:
            logger.warning(f"Generated code has syntax error: {e}")
            return False

        return True

    def _ensure_databricks_format(self, code: str, container_name: str) -> str:
        """
        Post-process generated code to ensure it is valid Databricks notebook format.

        Ensures:
        - First line is '# Databricks notebook source'
        - Cells separated by '# COMMAND ----------'
        - Markdown header cells use '# MAGIC %md'
        - Proper blank lines around COMMAND separators
        """
        lines = code.strip().split("\n")

        # Check if already in Databricks format
        has_notebook_header = lines[0].strip() == "# Databricks notebook source"
        has_command_seps = any(
            line.strip().startswith("# COMMAND") for line in lines
        )

        if has_notebook_header and has_command_seps:
            # Already in Databricks format - just normalize separators
            return self._normalize_command_separators(code)

        # Need to convert plain Python into Databricks notebook format
        return self._convert_to_databricks_notebook(code, container_name)

    def _normalize_command_separators(self, code: str) -> str:
        """Normalize COMMAND separators to have exactly one blank line before and after."""
        lines = code.split("\n")
        result = []
        i = 0
        while i < len(lines):
            stripped = lines[i].strip()
            if stripped.startswith("# COMMAND"):
                # Remove trailing blank lines before separator
                while result and result[-1].strip() == "":
                    result.pop()
                result.append("")
                result.append("# COMMAND ----------")
                result.append("")
                # Skip any blank lines after separator
                i += 1
                while i < len(lines) and lines[i].strip() == "":
                    i += 1
                continue
            result.append(lines[i])
            i += 1
        return "\n".join(result)

    def _convert_to_databricks_notebook(self, code: str, container_name: str) -> str:
        """Convert plain Python code to Databricks notebook format with proper cells."""
        import re as _re
        lines = code.strip().split("\n")

        cells = []

        # Cell 1: Title markdown
        cells.append(
            "# MAGIC %md\n"
            f"# MAGIC # Container: {container_name}\n"
            "# MAGIC Converted from Alteryx workflow to PySpark for Databricks."
        )

        # Split code into logical sections at comment headers
        current_section_lines = []
        current_header = None

        for line in lines:
            stripped = line.strip()

            # Skip existing notebook markers
            if stripped == "# Databricks notebook source":
                continue

            # Detect section headers (lines like "# ===" or "# ---" or "# STEP")
            is_header = (
                (stripped.startswith("# =") and len(stripped) > 10)
                or (stripped.startswith("# ---") and len(stripped) > 10)
                or _re.match(r"^# (STEP|Step|SOURCE|Source|FINAL|Final|IMPORT|Import)", stripped)
                or _re.match(r"^# (MAGIC|COMMAND)", stripped)
            )

            if is_header and not stripped.startswith("# MAGIC") and not stripped.startswith("# COMMAND"):
                # Start a new section - flush the current one
                if current_section_lines:
                    cell_content = "\n".join(current_section_lines).strip()
                    if cell_content:
                        if current_header:
                            clean_header = current_header.strip("# =-").strip()
                            if clean_header:
                                cells.append(
                                    f"# MAGIC %md\n"
                                    f"# MAGIC ## {clean_header}"
                                )
                        cells.append(cell_content)
                    current_section_lines = []

                current_header = stripped
                continue

            current_section_lines.append(line)

        # Flush remaining
        if current_section_lines:
            cell_content = "\n".join(current_section_lines).strip()
            if cell_content:
                if current_header:
                    clean_header = current_header.strip("# =-").strip()
                    if clean_header:
                        cells.append(
                            f"# MAGIC %md\n"
                            f"# MAGIC ## {clean_header}"
                        )
                cells.append(cell_content)

        # If no sections were detected, just put entire code in one cell
        if len(cells) <= 1:
            cells.append(code.strip())

        # Assemble into Databricks notebook format
        parts = ["# Databricks notebook source"]
        for cell in cells:
            parts.append("")
            parts.append("# COMMAND ----------")
            parts.append("")
            parts.append(cell)

        return "\n".join(parts) + "\n"
