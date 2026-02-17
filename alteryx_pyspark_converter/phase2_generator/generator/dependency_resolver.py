"""Resolve tool execution order using topological sort."""

import logging
from collections import defaultdict

from ...phase1_parser.models.workflow import Workflow
from .flow_analyzer import FlowAnalyzer

logger = logging.getLogger(__name__)


class DependencyResolver:
    """
    Resolve the execution order of tools within a container.

    Uses topological sort on the data flow graph to determine the
    correct order in which tools should be executed.
    """

    def __init__(self, workflow: Workflow):
        self.workflow = workflow
        self.flow_analyzer = FlowAnalyzer(workflow)

    def resolve_execution_order(self, container_id: int) -> list[int]:
        """
        Determine the execution order for tools in a container.

        Uses Kahn's algorithm (BFS topological sort) to handle the
        data flow dependencies.

        Args:
            container_id: The container to resolve.

        Returns:
            Ordered list of tool IDs in execution order.

        Raises:
            ValueError: If there's a cycle in the data flow.
        """
        container = self.workflow.get_container(container_id)
        if not container:
            raise ValueError(f"Container {container_id} not found")

        # Get all tool IDs in the container
        all_tool_ids = set(container.child_tool_ids)
        # Support both attribute names: child_container_ids (phase1) and sub_container_ids (src)
        sub_ids = getattr(container, 'child_container_ids', None) or \
                  getattr(container, 'sub_container_ids', [])
        for sub_id in sub_ids:
            sub = self.workflow.get_container(sub_id)
            if sub:
                all_tool_ids.update(sub.child_tool_ids)

        # Build adjacency list and in-degree count (only for internal tools)
        adjacency: dict[int, list[int]] = defaultdict(list)
        in_degree: dict[int, int] = {tid: 0 for tid in all_tool_ids}

        flow = self.flow_analyzer.get_container_flow(container_id)

        for tid in all_tool_ids:
            if tid in flow:
                for edge in flow[tid]["inputs"]:
                    if edge.from_tool_id in all_tool_ids:
                        adjacency[edge.from_tool_id].append(tid)
                        in_degree[tid] = in_degree.get(tid, 0) + 1

        # Kahn's algorithm
        queue = [tid for tid in all_tool_ids if in_degree.get(tid, 0) == 0]
        # Sort for deterministic order
        queue.sort()
        result: list[int] = []

        while queue:
            # Process tools with no remaining dependencies
            current = queue.pop(0)
            result.append(current)

            for neighbor in sorted(adjacency.get(current, [])):
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
            queue.sort()

        # Check for cycles
        if len(result) != len(all_tool_ids):
            remaining = all_tool_ids - set(result)
            logger.error(
                "Cycle detected in data flow. Unresolved tools: %s",
                remaining,
            )
            raise ValueError(
                f"Cycle detected in data flow for container {container_id}. "
                f"Unresolved tools: {remaining}"
            )

        logger.info(
            "Resolved execution order for container %d: %d tools",
            container_id,
            len(result),
        )

        return result

    def get_tool_dependencies(
        self, tool_id: int, container_id: int
    ) -> list[int]:
        """
        Get direct dependencies (tools that must execute before this one).

        Args:
            tool_id: Tool to get dependencies for.
            container_id: Container context.

        Returns:
            List of tool IDs that this tool depends on.
        """
        container = self.workflow.get_container(container_id)
        if not container:
            return []

        all_tool_ids = set(container.child_tool_ids)
        for sub_id in container.child_container_ids:
            sub = self.workflow.get_container(sub_id)
            if sub:
                all_tool_ids.update(sub.child_tool_ids)

        flow = self.flow_analyzer.get_container_flow(container_id)
        deps = []

        if tool_id in flow:
            for edge in flow[tool_id]["inputs"]:
                if edge.from_tool_id in all_tool_ids:
                    deps.append(edge.from_tool_id)

        return sorted(set(deps))

    def validate_flow(self, container_id: int) -> dict:
        """
        Validate the data flow in a container.

        Returns:
            Dictionary with validation results.
        """
        container = self.workflow.get_container(container_id)
        if not container:
            return {"valid": False, "error": "Container not found"}

        issues = []

        try:
            order = self.resolve_execution_order(container_id)
        except ValueError as e:
            return {"valid": False, "error": str(e)}

        # Check for disconnected tools
        flow = self.flow_analyzer.get_container_flow(container_id)
        all_tool_ids = set(container.child_tool_ids)

        for tid in all_tool_ids:
            tool = self.workflow.get_tool(tid)
            if tool is None:
                issues.append(f"Tool {tid}: not found in workflow")
                continue

            if tid in flow:
                inputs = flow[tid]["inputs"]
                outputs = flow[tid]["outputs"]
                if not inputs and not outputs:
                    issues.append(
                        f"Tool {tid} ({tool.tool_type}): disconnected"
                    )

        return {
            "valid": len(issues) == 0,
            "execution_order": order,
            "issues": issues,
            "num_tools": len(all_tool_ids),
        }
