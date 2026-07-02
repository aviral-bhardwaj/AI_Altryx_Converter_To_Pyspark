"""
Tool Converter Base Classes
============================
Shared infrastructure for the plugin-style tool registry:

- ``ToolConverter``     : abstract base class every tool plugin implements
- ``GeneratorContext``  : mutable state threaded through code generation
                          (DataFrame variable names, connection graph, config)
- ``topological_sort``  : Kahn's algorithm over the workflow DAG

New Alteryx tools are added by subclassing ``ToolConverter`` in
``src/tools/`` and registering them via ``@register("ToolType")`` —
no changes to the core engine or notebook are required.
"""

import logging
import re
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from typing import Optional

from ..models import Workflow, Tool

logger = logging.getLogger(__name__)


def topological_sort(tools: list, connections: list) -> list:
    """
    Return tool IDs in execution order using Kahn's algorithm.
    Tools with no dependencies come first.
    """
    tool_ids = {t.tool_id for t in tools}
    in_degree = defaultdict(int)
    adjacency = defaultdict(list)

    for conn in connections:
        if conn.origin_tool_id in tool_ids and conn.dest_tool_id in tool_ids:
            adjacency[conn.origin_tool_id].append(conn.dest_tool_id)
            in_degree[conn.dest_tool_id] += 1

    # Initialize queue with tools that have no incoming edges
    queue = deque()
    for t in tools:
        if in_degree[t.tool_id] == 0:
            queue.append(t.tool_id)

    ordered = []
    while queue:
        tid = queue.popleft()
        ordered.append(tid)
        for neighbor in adjacency[tid]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    # If some tools weren't reached (cycle), append them at the end
    visited = set(ordered)
    for t in tools:
        if t.tool_id not in visited:
            ordered.append(t.tool_id)

    return ordered


class ToolConverter(ABC):
    """Base class for all tool converters (plugins)."""

    #: Tool types this converter handles; populated by @register.
    tool_types: tuple = ()

    @abstractmethod
    def convert(self, tool: Tool, ctx: "GeneratorContext") -> list:
        """
        Convert a tool into PySpark code lines.

        Args:
            tool: The Alteryx tool to convert.
            ctx: Generator context with variable mappings and connections.

        Returns:
            List of code line strings.
        """
        ...


class GeneratorContext:
    """
    Holds state during code generation:
    - Variable names for each tool's output DataFrame(s)
    - Connection graph
    - Source table configuration
    """

    def __init__(
        self,
        workflow: Workflow,
        tools: list,
        connections: list,
        source_tables_config: Optional[dict] = None,
        target_catalog: str = "",
        target_schema: str = "",
    ):
        self.workflow = workflow
        self.tools = {t.tool_id: t for t in tools}
        self.connections = connections
        self.source_tables_config = source_tables_config or {}
        self.target_catalog = target_catalog
        self.target_schema = target_schema

        # df_vars: tool_id -> { port_name: variable_name }
        # Default port is "Output"
        self.df_vars: dict = {}

        # Variable names already handed out — collisions get a _<tool_id>
        # suffix so two tools sharing an annotation never overwrite each other.
        self._used_var_names: set = set()

        # Build connection lookups
        self._incoming: dict = defaultdict(list)
        self._outgoing: dict = defaultdict(list)
        for conn in connections:
            self._incoming[conn.dest_tool_id].append(conn)
            self._outgoing[conn.origin_tool_id].append(conn)

    def get_input_var(self, tool_id: int, port: str = "Input") -> str:
        """Get the DataFrame variable name feeding into a tool's input port."""
        for conn in self._incoming[tool_id]:
            if conn.dest_connection == port or port == "Input":
                origin_port = conn.origin_connection
                vars_map = self.df_vars.get(conn.origin_tool_id, {})
                # Try exact port, then "Output" default
                if origin_port in vars_map:
                    return vars_map[origin_port]
                if "Output" in vars_map:
                    return vars_map["Output"]
        return f"df_{tool_id}_input"

    def get_input_var_for_port(self, tool_id: int, port: str) -> str:
        """Get input variable for a specific named port (e.g., 'Left', 'Right')."""
        for conn in self._incoming[tool_id]:
            if conn.dest_connection == port:
                origin_port = conn.origin_connection
                vars_map = self.df_vars.get(conn.origin_tool_id, {})
                if origin_port in vars_map:
                    return vars_map[origin_port]
                if "Output" in vars_map:
                    return vars_map["Output"]
        return f"df_{tool_id}_{port.lower()}_input"

    def set_output_var(self, tool_id: int, var_name: str, port: str = "Output"):
        """Set the output DataFrame variable name for a tool's port."""
        if tool_id not in self.df_vars:
            self.df_vars[tool_id] = {}
        self.df_vars[tool_id][port] = var_name
        self._used_var_names.add(var_name)

    def get_output_var(self, tool_id: int, port: str = "Output") -> str:
        """Get the output variable name for a tool."""
        vars_map = self.df_vars.get(tool_id, {})
        return vars_map.get(port, vars_map.get("Output", f"df_{tool_id}"))

    def is_port_connected(self, tool_id: int, port: str) -> bool:
        """Check if a specific output port is connected downstream."""
        for conn in self._outgoing[tool_id]:
            if conn.origin_connection == port:
                return True
        return False

    def get_outgoing(self, tool_id: int) -> list:
        """Get all outgoing connections from a tool."""
        return self._outgoing[tool_id]

    def get_incoming(self, tool_id: int) -> list:
        """Get all incoming connections to a tool."""
        return self._incoming[tool_id]

    def make_var_name(self, tool: Tool) -> str:
        """Generate a meaningful, collision-free variable name for a tool's output."""
        candidate = self._candidate_var_name(tool)
        # Two tools can share an annotation or a table leaf name; dedupe so
        # the second one never silently overwrites the first's DataFrame.
        if candidate in self._used_var_names:
            candidate = f"{candidate}_{tool.tool_id}"
        self._used_var_names.add(candidate)
        return candidate

    def _candidate_var_name(self, tool: Tool) -> str:
        ann = (tool.annotation or "").strip()
        if ann and len(ann) > 2:
            cleaned = ann.lower()
            cleaned = re.sub(r"[^a-z0-9_]", "_", cleaned)
            cleaned = re.sub(r"_+", "_", cleaned).strip("_")
            if cleaned and len(cleaned) <= 40:
                return f"df_{cleaned}"

        pc = tool.parsed_config or {}
        tt = tool.tool_type

        if tt in ("InputData", "LockInStreamIn"):
            table = pc.get("table_name", "")
            if table:
                name = table.rsplit(".", 1)[-1].rsplit("/", 1)[-1]
                name = re.sub(r"[^a-z0-9_]", "_", name.lower()).strip("_")
                if name:
                    return f"df_{name}"

        if tt == "TextInput":
            return f"df_text_input_{tool.tool_id}"

        return f"df_{tool.tool_id}"
