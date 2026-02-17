"""Intelligent semantic variable naming for generated PySpark DataFrames.

Instead of generic names like df_join_42, df_filter_15, this module
analyzes tool context (annotations, configurations, join keys, filter
conditions, data lineage) to produce meaningful names like:

    df_active_customers, df_orders_joined_products, df_sales_by_region
"""

import re
import logging
from typing import Optional

from ...phase1_parser.models.tool import Tool
from ...phase1_parser.models.workflow import Workflow
from .flow_analyzer import FlowAnalyzer

logger = logging.getLogger(__name__)

# Words to strip from annotations that add no naming value
_NOISE_WORDS = frozenset({
    "tool", "the", "a", "an", "this", "and", "or", "is", "are",
    "of", "for", "to", "from", "in", "with", "by", "on", "at",
    "data", "records", "rows", "output", "result", "results",
    "step", "process", "processing", "module",
})

# Maximum length for a variable name segment
_MAX_SEGMENT_LEN = 30


def _clean_for_python(raw: str) -> str:
    """Sanitise a raw string into a valid Python identifier fragment."""
    name = raw.lower().strip()
    # Replace non-alnum with underscore
    name = re.sub(r"[^a-z0-9]+", "_", name)
    name = name.strip("_")
    # Remove noise words
    parts = [p for p in name.split("_") if p and p not in _NOISE_WORDS]
    name = "_".join(parts)
    # Truncate
    if len(name) > _MAX_SEGMENT_LEN:
        name = name[:_MAX_SEGMENT_LEN].rstrip("_")
    # Ensure it doesn't start with a digit
    if name and name[0].isdigit():
        name = f"t_{name}"
    return name


class SemanticNamer:
    """Resolve human-readable DataFrame variable names from tool context.

    The naming priority is:

    1.  **Annotation-based** – If the tool has a user-written annotation
        (e.g. "Filter Category A"), derive the name from it.
    2.  **Configuration-based** – Infer semantics from the tool config
        (join keys, filter expressions, aggregation fields, etc.).
    3.  **Data-lineage-based** – Propagate the upstream name and append
        a short action suffix (``_filtered``, ``_joined``, …).
    4.  **Fallback** – ``df_<tool_type>_<tool_id>`` (same as before).

    The resolver also guarantees **uniqueness** across the notebook.
    """

    def __init__(self, workflow: Workflow, flow_analyzer: FlowAnalyzer):
        self.workflow = workflow
        self.flow_analyzer = flow_analyzer
        # Track names already assigned  →  prevents collisions
        self._used_names: set[str] = set()
        # tool_id → resolved base name (without output-port suffix)
        self._cache: dict[int, str] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def resolve_name(
        self,
        tool: Tool,
        container_id: int,
        upstream_names: dict[str, str] | None = None,
    ) -> str:
        """Return a unique, meaningful ``df_…`` prefix for *tool*.

        Parameters
        ----------
        tool:
            The tool to name.
        container_id:
            Container the tool belongs to.
        upstream_names:
            Mapping ``{connection_name: df_var_name}`` of the tool's
            resolved input DataFrames (already named).  Used for
            lineage-based naming.

        Returns
        -------
        str
            A Python-safe ``df_…`` variable name prefix, guaranteed
            unique within this notebook.
        """
        if tool.tool_id in self._cache:
            return self._cache[tool.tool_id]

        # 1. Try annotation
        name = self._from_annotation(tool)

        # 2. Try configuration semantics
        if not name:
            name = self._from_config(tool, upstream_names)

        # 3. Try data-lineage propagation
        if not name:
            name = self._from_lineage(tool, upstream_names)

        # 4. Fallback
        if not name:
            name = f"df_{tool.tool_type.lower()}_{tool.tool_id}"

        # Ensure the "df_" prefix
        if not name.startswith("df_"):
            name = f"df_{name}"

        # Make unique
        name = self._make_unique(name)

        self._cache[tool.tool_id] = name
        return name

    def resolve_input_source_name(
        self,
        table_key: str,
        table_name: str,
        tool_id: int,
    ) -> str:
        """Name an external input table load variable.

        Instead of ``df_input_fact_nps``, produce something like
        ``df_fact_nps`` derived from the *table_key* or Databricks path.
        """
        # Prefer the table_key (short config name)
        segment = _clean_for_python(table_key)
        if not segment:
            # Fall back to last part of databricks table path
            segment = _clean_for_python(table_name.rsplit(".", 1)[-1])
        if not segment:
            segment = f"input_{tool_id}"

        name = f"df_{segment}"
        name = self._make_unique(name)
        self._cache[tool_id] = name
        return name

    # ------------------------------------------------------------------
    # Naming strategies
    # ------------------------------------------------------------------

    def _from_annotation(self, tool: Tool) -> str | None:
        """Derive a name from the tool's annotation text."""
        ann = (tool.annotation or "").strip()
        if not ann or len(ann) < 2:
            return None

        cleaned = _clean_for_python(ann)
        if not cleaned or len(cleaned) < 2:
            return None

        return f"df_{cleaned}"

    def _from_config(
        self,
        tool: Tool,
        upstream_names: dict[str, str] | None,
    ) -> str | None:
        """Infer a name from the tool's configuration."""
        cfg = tool.configuration or {}
        tt = tool.tool_type

        # --- Filter: describe what it filters on ---------------------------
        if tt in ("Filter", "LockInFilter"):
            expr = cfg.get("expression", "")
            col = self._extract_primary_column(expr)
            if col:
                short = _clean_for_python(col)
                return f"df_filtered_by_{short}" if short else None

        # --- Join: describe the join context --------------------------------
        if tt in ("Join", "LockInJoin"):
            left_keys = cfg.get("left_keys", [])
            right_keys = cfg.get("right_keys", [])
            # Derive from the upstream names if possible
            left_src = self._upstream_short(upstream_names, "Input", "Left")
            right_src = self._upstream_short(upstream_names, "Right")
            if left_src and right_src:
                return f"df_{left_src}_joined_{right_src}"
            # Fall back to join keys
            key_name = _clean_for_python(
                "_".join(left_keys[:2]) if left_keys else ""
            )
            if key_name:
                return f"df_joined_on_{key_name}"

        # --- Formula: describe created/updated columns ----------------------
        if tt in ("Formula", "LockInFormula"):
            formulas = cfg.get("formulas", [])
            if formulas:
                fields = [f.get("field", "") for f in formulas[:2]]
                short = _clean_for_python("_".join(f for f in fields if f))
                if short:
                    return f"df_calc_{short}"

        # --- Summarize: describe aggregation --------------------------------
        if tt in ("Summarize", "LockInSummarize"):
            fields = cfg.get("fields", [])
            groups = [f.get("field", "") for f in fields if f.get("action") == "GroupBy"]
            aggs = [f.get("action", "") for f in fields if f.get("action") != "GroupBy"]
            if groups:
                grp = _clean_for_python("_".join(groups[:2]))
                if grp:
                    return f"df_summary_by_{grp}"
            if aggs:
                return f"df_aggregated"

        # --- Select: describe what columns remain ---------------------------
        if tt in ("Select", "LockInSelect"):
            parent = self._upstream_short(upstream_names, "Input")
            if parent:
                return f"df_{parent}_selected"

        # --- Sort: describe sort columns ------------------------------------
        if tt in ("Sort", "LockInSort"):
            fields = cfg.get("fields", [])
            if fields:
                col = _clean_for_python(fields[0].get("field", ""))
                if col:
                    return f"df_sorted_by_{col}"

        # --- Unique: describe dedup -----------------------------------------
        if tt in ("Unique", "LockInUnique"):
            ufields = cfg.get("unique_fields", [])
            if ufields:
                col = _clean_for_python("_".join(ufields[:2]))
                if col:
                    return f"df_unique_by_{col}"

        # --- Union ---
        if tt in ("Union", "LockInUnion"):
            return "df_unioned"

        # --- CrossTab / Pivot -----------------------------------------------
        if tt in ("CrossTab", "LockInCrossTab"):
            header = cfg.get("header_field", "")
            if header:
                short = _clean_for_python(header)
                if short:
                    return f"df_pivot_by_{short}"

        # --- TextInput: use annotation or just "inline_data" ----------------
        if tt == "TextInput":
            return None  # Let annotation or fallback handle it

        # --- Transpose -------------------------------------------------------
        if tt == "Transpose":
            return "df_transposed"

        # --- RecordID ---------------------------------------------------------
        if tt == "RecordID":
            field_name = cfg.get("field_name", "RecordID")
            short = _clean_for_python(field_name)
            return f"df_with_{short}" if short else None

        # --- RegEx ------------------------------------------------------------
        if tt == "RegEx":
            field = cfg.get("field", "")
            short = _clean_for_python(field)
            return f"df_regex_{short}" if short else None

        # --- Sample -----------------------------------------------------------
        if tt == "Sample":
            return "df_sampled"

        # --- MultiRowFormula --------------------------------------------------
        if tt == "MultiRowFormula":
            field = cfg.get("field", "")
            short = _clean_for_python(field)
            return f"df_multirow_{short}" if short else None

        return None

    def _from_lineage(
        self,
        tool: Tool,
        upstream_names: dict[str, str] | None,
    ) -> str | None:
        """Propagate the upstream name with an action suffix."""
        if not upstream_names:
            return None

        parent = self._upstream_short(upstream_names, "Input", "Left", "Output")
        if not parent:
            return None

        tt = tool.tool_type
        suffix_map = {
            "Filter": "filtered",
            "Formula": "transformed",
            "Select": "selected",
            "Sort": "sorted",
            "Unique": "deduped",
            "Sample": "sampled",
            "Join": "joined",
            "Union": "unioned",
            "Summarize": "aggregated",
            "CrossTab": "pivoted",
            "Transpose": "transposed",
            "RecordID": "with_id",
            "RegEx": "regex",
            "MultiRowFormula": "multirow",
        }
        # Also handle LockIn variants
        base_type = tt.replace("LockIn", "")
        suffix = suffix_map.get(tt) or suffix_map.get(base_type, tool.tool_type.lower())

        return f"df_{parent}_{suffix}"

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _extract_primary_column(self, expression: str) -> str:
        """Pull the first [ColumnName] from an Alteryx expression."""
        m = re.search(r"\[([^\]]+)\]", expression)
        return m.group(1) if m else ""

    def _upstream_short(
        self,
        upstream_names: dict[str, str] | None,
        *preferred_keys: str,
    ) -> str | None:
        """Get a short identifier from an upstream DataFrame name.

        Given ``df_active_customers`` returns ``active_customers``.
        Tries *preferred_keys* in order, then falls back to any value.
        """
        if not upstream_names:
            return None

        df_name: str | None = None
        for key in preferred_keys:
            if key in upstream_names:
                df_name = upstream_names[key]
                break
        if df_name is None:
            # Take the first available
            df_name = next(iter(upstream_names.values()), None)
        if df_name is None:
            return None

        # Strip the "df_" prefix
        short = df_name.removeprefix("df_")
        # Truncate to keep names reasonable
        if len(short) > _MAX_SEGMENT_LEN:
            short = short[:_MAX_SEGMENT_LEN].rstrip("_")
        return short if short else None

    def _make_unique(self, name: str) -> str:
        """Ensure *name* is unique; append a counter suffix if needed."""
        if name not in self._used_names:
            self._used_names.add(name)
            return name

        for i in range(2, 100):
            candidate = f"{name}_{i}"
            if candidate not in self._used_names:
                self._used_names.add(candidate)
                return candidate

        # Extremely unlikely – fall through with tool ID
        import random
        candidate = f"{name}_{random.randint(100, 999)}"
        self._used_names.add(candidate)
        return candidate
