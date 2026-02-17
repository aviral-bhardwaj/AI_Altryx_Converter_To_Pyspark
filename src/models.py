"""
Data models for parsed Alteryx workflow elements.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Connection:
    """A connection between two tools."""
    origin_tool_id: int
    origin_connection: str  # e.g., "Output", "True", "False", "Join", "Left", "Right"
    dest_tool_id: int
    dest_connection: str    # e.g., "Input", "Left", "Right"
    wireless: bool = False

    def __repr__(self):
        return (f"Connection({self.origin_tool_id}:{self.origin_connection} "
                f"→ {self.dest_tool_id}:{self.dest_connection})")


@dataclass
class Tool:
    """A single Alteryx tool node."""
    tool_id: int
    plugin: str             # e.g., "AlteryxBasePluginsGui.Filter.Filter"
    tool_type: str          # Derived short name, e.g., "Filter", "Formula", "Join"
    position: dict          # x, y coordinates
    configuration_xml: str  # Raw XML of <Configuration> block
    annotation: str         # Human-readable annotation text
    container_id: Optional[int] = None  # Parent container ToolID, if any
    parsed_config: dict = field(default_factory=dict)  # Structured config extracted from XML

    @property
    def short_type(self) -> str:
        """Short readable name from plugin string."""
        return self.tool_type

    def __repr__(self):
        return f"Tool({self.tool_id}, {self.tool_type}, '{self.annotation[:40]}')"


@dataclass
class Container:
    """A Tool Container (grouping of tools)."""
    tool_id: int
    name: str               # Caption text
    child_tool_ids: list = field(default_factory=list)
    child_tools: list = field(default_factory=list)  # List[Tool]
    sub_container_ids: list = field(default_factory=list)
    parent_container_id: Optional[int] = None
    disabled: bool = False
    style: dict = field(default_factory=dict)

    def __repr__(self):
        return f"Container({self.tool_id}, '{self.name}', {len(self.child_tools)} tools)"


@dataclass
class Workflow:
    """Complete parsed Alteryx workflow."""
    containers: list        # List[Container] - top-level containers
    all_containers: dict    # {tool_id: Container} - all containers including nested
    all_tools: dict         # {tool_id: Tool}
    connections: list       # List[Connection]
    text_inputs: dict       # {tool_id: list[dict]} - inline data from TextInput tools
    metadata: dict = field(default_factory=dict)

    def get_tool(self, tool_id: int) -> Optional[Tool]:
        return self.all_tools.get(tool_id)

    def get_container(self, tool_id: int) -> Optional[Container]:
        return self.all_containers.get(tool_id)

    def get_incoming_connections(self, tool_id: int) -> list:
        """Get all connections where this tool is the destination."""
        return [c for c in self.connections if c.dest_tool_id == tool_id]

    def get_outgoing_connections(self, tool_id: int) -> list:
        """Get all connections where this tool is the origin."""
        return [c for c in self.connections if c.origin_tool_id == tool_id]

    def get_container_context(self, container_tool_id: int) -> dict:
        """
        Build the full context needed for Claude AI to generate code for a container.
        Includes: tools, connections, external inputs/outputs, sub-containers.
        """
        container = self.all_containers.get(container_tool_id)
        if not container:
            return {}

        # Collect ALL tool IDs in this container (including sub-containers recursively)
        all_tool_ids = set()
        self._collect_container_tool_ids(container_tool_id, all_tool_ids)

        # Internal connections (both endpoints inside this container)
        internal_connections = [
            c for c in self.connections
            if c.origin_tool_id in all_tool_ids and c.dest_tool_id in all_tool_ids
        ]

        # External inputs: connections coming FROM outside INTO this container
        external_inputs = [
            c for c in self.connections
            if c.dest_tool_id in all_tool_ids and c.origin_tool_id not in all_tool_ids
        ]

        # External outputs: connections going FROM this container TO outside
        external_outputs = [
            c for c in self.connections
            if c.origin_tool_id in all_tool_ids and c.dest_tool_id not in all_tool_ids
        ]

        # Sub-containers
        sub_containers = [
            self.all_containers[tid]
            for tid in container.sub_container_ids
            if tid in self.all_containers
        ]

        # Tools with their configs
        tools_in_container = [
            self.all_tools[tid]
            for tid in all_tool_ids
            if tid in self.all_tools
        ]

        # Text input data for tools in this container
        text_input_data = {
            tid: self.text_inputs[tid]
            for tid in all_tool_ids
            if tid in self.text_inputs
        }

        # Source tool info for external inputs (full Tool objects for config access)
        source_tools = {}
        for conn in external_inputs:
            if conn.origin_tool_id in self.all_tools:
                src = self.all_tools[conn.origin_tool_id]
                source_tools[conn.origin_tool_id] = {
                    "tool_id": src.tool_id,
                    "type": src.tool_type,
                    "annotation": src.annotation,
                    "container": self._find_tool_container_name(src.tool_id),
                    "connection_type": conn.origin_connection,
                    "parsed_config": src.parsed_config,
                    "configuration_xml": src.configuration_xml,
                }
                # Also include TextInput data from external sources
                if src.tool_id in self.text_inputs:
                    source_tools[conn.origin_tool_id]["text_input_data"] = self.text_inputs[src.tool_id]

        return {
            "container": container,
            "tools": tools_in_container,
            "internal_connections": internal_connections,
            "external_inputs": external_inputs,
            "external_outputs": external_outputs,
            "sub_containers": sub_containers,
            "text_input_data": text_input_data,
            "source_tools": source_tools,
        }

    def _collect_container_tool_ids(self, container_id: int, result: set):
        """Recursively collect all tool IDs in a container."""
        container = self.all_containers.get(container_id)
        if not container:
            return
        for tid in container.child_tool_ids:
            result.add(tid)
        for sub_id in container.sub_container_ids:
            self._collect_container_tool_ids(sub_id, result)

    def get_root_tools(self) -> list:
        """Get tools that are not inside any container."""
        return [t for t in self.all_tools.values() if t.container_id is None]

    def get_root_tool_ids(self) -> set:
        """Get IDs of tools that are not inside any container."""
        return {tid for tid, t in self.all_tools.items() if t.container_id is None}

    def get_root_context(self) -> dict:
        """
        Build context for root-level tools (tools not in any container).
        Returns the same structure as get_container_context() so the AI
        generator and context builder can handle it identically.
        """
        root_tool_ids = self.get_root_tool_ids()
        if not root_tool_ids:
            return {}

        # All container tool IDs (tools that belong to some container)
        container_tool_ids = set()
        for cid in self.all_containers:
            self._collect_container_tool_ids(cid, container_tool_ids)

        # Internal connections (both endpoints are root-level tools)
        internal_connections = [
            c for c in self.connections
            if c.origin_tool_id in root_tool_ids and c.dest_tool_id in root_tool_ids
        ]

        # External inputs: connections FROM container tools INTO root-level tools
        external_inputs = [
            c for c in self.connections
            if c.dest_tool_id in root_tool_ids and c.origin_tool_id not in root_tool_ids
        ]

        # External outputs: connections FROM root-level tools INTO container tools
        external_outputs = [
            c for c in self.connections
            if c.origin_tool_id in root_tool_ids and c.dest_tool_id not in root_tool_ids
        ]

        # Root-level tools list
        tools = [self.all_tools[tid] for tid in root_tool_ids if tid in self.all_tools]

        # Text input data for root-level tools
        text_input_data = {
            tid: self.text_inputs[tid]
            for tid in root_tool_ids
            if tid in self.text_inputs
        }

        # Source tool info for external inputs (from containers into root)
        source_tools = {}
        for conn in external_inputs:
            if conn.origin_tool_id in self.all_tools:
                src = self.all_tools[conn.origin_tool_id]
                source_tools[conn.origin_tool_id] = {
                    "tool_id": src.tool_id,
                    "type": src.tool_type,
                    "annotation": src.annotation,
                    "container": self._find_tool_container_name(src.tool_id),
                    "connection_type": conn.origin_connection,
                    "parsed_config": src.parsed_config,
                    "configuration_xml": src.configuration_xml,
                }
                if src.tool_id in self.text_inputs:
                    source_tools[conn.origin_tool_id]["text_input_data"] = self.text_inputs[src.tool_id]

        # Create a virtual container to represent root-level tools
        virtual_container = Container(
            tool_id=-1,
            name="Main_Workflow",
            child_tool_ids=list(root_tool_ids),
            child_tools=tools,
        )

        return {
            "container": virtual_container,
            "tools": tools,
            "internal_connections": internal_connections,
            "external_inputs": external_inputs,
            "external_outputs": external_outputs,
            "sub_containers": [],
            "text_input_data": text_input_data,
            "source_tools": source_tools,
        }

    def _find_tool_container_name(self, tool_id: int) -> str:
        """Find which container a tool belongs to."""
        tool = self.all_tools.get(tool_id)
        if tool and tool.container_id and tool.container_id in self.all_containers:
            return self.all_containers[tool.container_id].name
        return "Root"
