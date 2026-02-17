"""Analyze data flow from connections to determine execution paths."""

import logging
from typing import Optional

from ...phase1_parser.models.workflow import Workflow
from ...phase1_parser.models.connection import Connection
from ...phase1_parser.models.tool import Tool

logger = logging.getLogger(__name__)


def _conn_dest_id(conn) -> int:
    """Get destination tool ID from a connection, supporting both models."""
    return getattr(conn, 'destination_tool_id', None) or getattr(conn, 'dest_tool_id', 0)


def _conn_dest_conn(conn) -> str:
    """Get destination connection name from a connection, supporting both models."""
    return getattr(conn, 'destination_connection', None) or getattr(conn, 'dest_connection', 'Input')


class DataFlowEdge:
    """Represents a directed edge in the data flow graph."""

    def __init__(
        self,
        from_tool_id: int,
        to_tool_id: int,
        from_connection: str,
        to_connection: str,
    ):
        self.from_tool_id = from_tool_id
        self.to_tool_id = to_tool_id
        self.from_connection = from_connection
        self.to_connection = to_connection

    def __repr__(self) -> str:
        return (
            f"Edge({self.from_tool_id}:{self.from_connection} "
            f"-> {self.to_tool_id}:{self.to_connection})"
        )


class FlowAnalyzer:
    """
    Analyze data flow within a container to determine execution order
    and data dependencies.
    """

    def __init__(self, workflow: Workflow):
        self.workflow = workflow

    def get_container_flow(
        self, container_id: int
    ) -> dict[int, dict[str, list[DataFlowEdge]]]:
        """
        Get the data flow graph for tools within a container.

        Returns:
            Dictionary mapping tool_id to its input/output edges:
            {
                tool_id: {
                    "inputs": [DataFlowEdge, ...],
                    "outputs": [DataFlowEdge, ...]
                }
            }
        """
        container = self.workflow.get_container(container_id)
        if not container:
            return {}

        # Get all tool IDs in this container (including sub-containers)
        all_tool_ids = set(container.child_tool_ids)
        # Support both attribute names: child_container_ids (phase1) and sub_container_ids (src)
        sub_ids = getattr(container, 'child_container_ids', None) or \
                  getattr(container, 'sub_container_ids', [])
        for sub_id in sub_ids:
            sub = self.workflow.get_container(sub_id)
            if sub:
                all_tool_ids.update(sub.child_tool_ids)

        # Build flow graph
        flow: dict[int, dict[str, list[DataFlowEdge]]] = {}
        for tid in all_tool_ids:
            flow[tid] = {"inputs": [], "outputs": []}

        # Also add entries for external sources
        for conn in self.workflow.connections:
            dest_id = _conn_dest_id(conn)
            dest_conn = _conn_dest_conn(conn)

            if dest_id in all_tool_ids:
                edge = DataFlowEdge(
                    from_tool_id=conn.origin_tool_id,
                    to_tool_id=dest_id,
                    from_connection=conn.origin_connection,
                    to_connection=dest_conn,
                )

                if dest_id in flow:
                    flow[dest_id]["inputs"].append(edge)

                if conn.origin_tool_id in all_tool_ids:
                    if conn.origin_tool_id in flow:
                        flow[conn.origin_tool_id]["outputs"].append(edge)
                else:
                    # External source
                    if conn.origin_tool_id not in flow:
                        flow[conn.origin_tool_id] = {"inputs": [], "outputs": []}
                    flow[conn.origin_tool_id]["outputs"].append(edge)

        return flow

    def get_source_tools(self, container_id: int) -> list[int]:
        """
        Get tools that are data sources within a container.

        Source tools either:
        - Are TextInput tools (inline data)
        - Receive data from outside the container (external inputs)
        - Have no incoming connections within the container
        """
        flow = self.get_container_flow(container_id)
        container = self.workflow.get_container(container_id)
        if not container:
            return []

        all_tool_ids = set(container.child_tool_ids)
        for sub_id in container.child_container_ids:
            sub = self.workflow.get_container(sub_id)
            if sub:
                all_tool_ids.update(sub.child_tool_ids)

        sources = []
        for tid in all_tool_ids:
            tool = self.workflow.get_tool(tid)
            if tool is None:
                continue

            # TextInput is always a source
            if tool.tool_type == "TextInput":
                sources.append(tid)
                continue

            # Check if all inputs come from outside the container
            if tid in flow:
                internal_inputs = [
                    e for e in flow[tid]["inputs"]
                    if e.from_tool_id in all_tool_ids
                ]
                if not internal_inputs:
                    sources.append(tid)

        return sources

    def get_sink_tools(self, container_id: int) -> list[int]:
        """Get tools that are data sinks (no outgoing connections within container)."""
        flow = self.get_container_flow(container_id)
        container = self.workflow.get_container(container_id)
        if not container:
            return []

        all_tool_ids = set(container.child_tool_ids)
        for sub_id in container.child_container_ids:
            sub = self.workflow.get_container(sub_id)
            if sub:
                all_tool_ids.update(sub.child_tool_ids)

        sinks = []
        for tid in all_tool_ids:
            if tid in flow:
                internal_outputs = [
                    e for e in flow[tid]["outputs"]
                    if e.to_tool_id in all_tool_ids
                ]
                if not internal_outputs:
                    sinks.append(tid)

        return sinks

    def get_tool_input_df_names(
        self,
        tool_id: int,
        container_id: int,
        df_name_map: dict[int, dict[str, str]],
    ) -> dict[str, str]:
        """
        Determine DataFrame variable names for a tool's inputs.

        Args:
            tool_id: The tool to get inputs for.
            container_id: The container context.
            df_name_map: Mapping of tool_id -> {connection_name: df_var_name}.

        Returns:
            Dictionary mapping connection name to DataFrame variable name.
        """
        flow = self.get_container_flow(container_id)
        result: dict[str, str] = {}

        if tool_id not in flow:
            return result

        for edge in flow[tool_id]["inputs"]:
            from_id = edge.from_tool_id
            from_conn = edge.from_connection
            to_conn = edge.to_connection

            if from_id in df_name_map:
                # Look up the output name from the source tool
                if from_conn in df_name_map[from_id]:
                    result[to_conn] = df_name_map[from_id][from_conn]
                elif "Output" in df_name_map[from_id]:
                    result[to_conn] = df_name_map[from_id]["Output"]

        return result
