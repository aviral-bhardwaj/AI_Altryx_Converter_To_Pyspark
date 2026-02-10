"""
Claude AI Code Generator
=========================
Sends container context to Claude API and returns generated PySpark code.
Handles retries, validation, and code extraction.
"""

import json
import logging
import re
import time
from typing import Optional

from .models import Workflow, Container
from .context_builder import build_container_prompt, build_system_prompt

logger = logging.getLogger(__name__)

# ── Try to import anthropic SDK; fall back to requests ───────────────
try:
    import anthropic
    HAS_ANTHROPIC_SDK = True
except ImportError:
    HAS_ANTHROPIC_SDK = False
    import urllib.request
    import urllib.error


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
1. spark.table() calls for ALL {num_ext_inputs} external source inputs
2. Transformation code for ALL {num_tools} tools listed above
3. Proper data flow following ALL {num_int_conns} internal connections
4. For each Join: use the exact join keys specified, handle post-join column drops/renames
5. For each Filter: generate the True/False outputs that are used downstream
6. For each Formula: convert ALL formula fields to .withColumn() calls
7. For each Select: drop deselected columns, apply renames
8. Final output as createOrReplaceTempView() or write statement

Output ONLY the Python code. No markdown fences, no explanations before or after."""

        # Call Claude with retries
        code = None
        last_error = None

        for attempt in range(1, self.max_retries + 2):  # +2 because range is exclusive and we start at 1
            try:
                logger.info(
                    f"Calling Claude API for '{container.name}' "
                    f"(attempt {attempt}/{self.max_retries + 1})"
                )
                print(f"   🔄 API call attempt {attempt}...")

                raw_response = self._call_claude(system_prompt, user_message)
                code = self._extract_code(raw_response)

                if code and self._validate_code(code):
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
                    print(f"   ⏳ Waiting {wait}s before retry...")
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
        if text.startswith(("#", "\"\"\"", "from ", "import ")):
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

        # Strip Databricks magic commands before syntax check
        lines = []
        for line in code.split("\n"):
            stripped = line.strip()
            if stripped.startswith("# MAGIC") or stripped.startswith("# COMMAND"):
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
