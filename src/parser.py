"""
Alteryx .yxmd Workflow Parser
=============================
Parses Alteryx workflow XML into structured models (Containers, Tools, Connections).
Handles nested containers, all tool types, and connection tracking.
"""

import xml.etree.ElementTree as ET
import re
import logging
from typing import Optional

from .models import Workflow, Container, Tool, Connection

logger = logging.getLogger(__name__)

# ── Plugin name → short tool type mapping ────────────────────────────
PLUGIN_TYPE_MAP = {
    "Filter": "Filter",
    "Formula": "Formula",
    "AlteryxSelect": "Select",
    "Join": "Join",
    "Union": "Union",
    "Summarize": "Summarize",
    "CrossTab": "CrossTab",
    "Sort": "Sort",
    "Sample": "Sample",
    "Unique": "Unique",
    "TextInput": "TextInput",
    "DbFileInput": "InputData",
    "DbFileOutput": "OutputData",
    "BrowseV2": "Browse",
    "TextToColumns": "TextToColumns",
    "RegEx": "RegEx",
    "FindReplace": "FindReplace",
    "GenerateRows": "GenerateRows",
    "MultiRowFormula": "MultiRowFormula",
    "MultiFieldFormula": "MultiFieldFormula",
    "RecordID": "RecordID",
    "Transpose": "Transpose",
    "DateTime": "DateTime",
    "DynamicInput": "DynamicInput",
    "DynamicRename": "DynamicRename",
    "AppendFields": "AppendFields",
    "BlockUntilDone": "BlockUntilDone",
    "RunCommand": "RunCommand",
    "Comment": "Comment",
    "ToolContainer": "Container",
    "LockInFilter": "LockInFilter",
    "LockInStreamOut": "LockInStreamOut",
    "LockInStreamIn": "LockInStreamIn",
    "LockInSelect": "LockInSelect",
    "LockInJoin": "LockInJoin",
    "LockInFormula": "LockInFormula",
}


def _extract_tool_type(plugin_str: str) -> str:
    """Extract short tool type from plugin string."""
    if not plugin_str:
        return "Unknown"
    # Try known suffixes
    for key, value in PLUGIN_TYPE_MAP.items():
        if key in plugin_str:
            return value
    # Fallback: last dotted segment
    parts = plugin_str.rsplit(".", 1)
    return parts[-1] if parts else "Unknown"


def _get_text(element, path: str, default: str = "") -> str:
    """Safely get text from an XML path."""
    el = element.find(path)
    if el is not None and el.text:
        return el.text.strip()
    return default


def _get_attr(element, path: str, attr: str, default: str = "") -> str:
    """Safely get attribute from an XML path."""
    el = element.find(path)
    if el is not None:
        return el.get(attr, default)
    return default


def _extract_select_fields(config) -> list:
    """Extract Select/column configuration from a tool's XML."""
    select_fields = []
    for path in ["SelectConfiguration/Configuration/OrderChanged",
                  "SelectConfiguration", "Configuration"]:
        container = config.find(path)
        if container is None:
            continue
        for sf in container.findall("SelectField"):
            entry = {
                "field": sf.get("field", ""),
                "selected": sf.get("selected", "True"),
                "rename": sf.get("rename", ""),
                "type": sf.get("type", ""),
            }
            select_fields.append(entry)
        if select_fields:
            return select_fields
    # Fallback: search anywhere
    for sf in config.findall(".//SelectField"):
        entry = {
            "field": sf.get("field", ""),
            "selected": sf.get("selected", "True"),
            "rename": sf.get("rename", ""),
            "type": sf.get("type", ""),
        }
        select_fields.append(entry)
    return select_fields


def _extract_structured_config(config, tool_type: str) -> dict:
    """Extract structured, human-readable config from a tool's XML <Configuration> element."""
    if config is None:
        return {}

    parsed = {}

    if tool_type in ("Filter", "LockInFilter"):
        parsed["expression"] = _get_text(config, "Expression")
        parsed["mode"] = _get_text(config, "Mode", "Custom")

    elif tool_type in ("Formula", "LockInFormula"):
        formulas = []
        for fe in config.findall(".//FormulaField"):
            formulas.append({
                "field": fe.get("field", ""),
                "expression": fe.get("expression", ""),
                "type": fe.get("type", ""),
                "size": fe.get("size", ""),
            })
        parsed["formulas"] = formulas

    elif tool_type in ("Join", "LockInJoin"):
        left_keys, right_keys = [], []
        for ji in config.findall("JoinInfo"):
            side = ji.get("connection", "")
            for f in ji.findall("Field"):
                if side == "Left":
                    left_keys.append(f.get("field", ""))
                elif side == "Right":
                    right_keys.append(f.get("field", ""))
        parsed["left_keys"] = left_keys
        parsed["right_keys"] = right_keys
        parsed["join_by_position"] = _get_text(config, "JoinByPosition", "False")
        select_config = _extract_select_fields(config)
        if select_config:
            parsed["select_config"] = select_config

    elif tool_type in ("Select", "LockInSelect"):
        parsed["select_fields"] = _extract_select_fields(config)

    elif tool_type == "Summarize":
        fields = []
        for sf in config.findall(".//SummarizeField"):
            fields.append({
                "field": sf.get("field", ""),
                "action": sf.get("action", ""),
                "rename": sf.get("rename", ""),
            })
        parsed["summarize_fields"] = fields

    elif tool_type == "CrossTab":
        parsed["group_fields"] = _get_text(config, "GroupFields")
        parsed["header_field"] = _get_text(config, "HeaderField")
        parsed["data_field"] = _get_text(config, "DataField")
        parsed["method"] = _get_text(config, "Method")

    elif tool_type == "Sort":
        fields = []
        for sf in config.findall(".//SortInfo/Field"):
            fields.append({
                "field": sf.get("field", ""),
                "order": sf.get("order", "Ascending"),
            })
        parsed["sort_fields"] = fields

    elif tool_type == "Unique":
        fields = []
        for uf in config.findall(".//UniqueFields/Field"):
            fields.append(uf.get("field", ""))
        for uf in config.findall(".//UniqueField"):
            f = uf.get("field", "")
            if f and f not in fields:
                fields.append(f)
        parsed["unique_fields"] = fields

    elif tool_type == "Union":
        parsed["mode"] = _get_text(config, "Mode", "Auto")
        parsed["by_name"] = _get_text(config, "SetByName", "True")

    elif tool_type == "Sample":
        parsed["n_records"] = _get_text(config, "NRecords")
        parsed["group_fields"] = _get_text(config, "GroupFields")
        parsed["mode"] = _get_text(config, "Mode")

    elif tool_type in ("InputData", "LockInStreamIn"):
        table = (_get_text(config, "TableName")
                 or _get_text(config, "File")
                 or _get_text(config, "Connection"))
        parsed["table_name"] = table
        fields = []
        for f in config.findall(".//Field"):
            name = f.get("name", "") or f.get("field", "")
            ftype = f.get("type", "")
            if name:
                fields.append({"name": name, "type": ftype})
        if fields:
            parsed["fields"] = fields

    elif tool_type in ("OutputData", "LockInStreamOut"):
        parsed["table_name"] = (_get_text(config, "TableName")
                                or _get_text(config, "File"))

    elif tool_type == "MultiRowFormula":
        formulas = []
        for fe in config.findall(".//FormulaField"):
            formulas.append({
                "field": fe.get("field", ""),
                "expression": fe.get("expression", ""),
                "type": fe.get("type", ""),
            })
        parsed["formulas"] = formulas
        parsed["num_rows"] = _get_text(config, "NumRows")
        parsed["direction"] = _get_text(config, "UpdateDirection")

    elif tool_type == "RegEx":
        parsed["expression"] = _get_text(config, "Expression")
        parsed["field"] = _get_text(config, "Field")
        parsed["output_method"] = _get_text(config, "OutputMethod")

    elif tool_type == "Transpose":
        key_fields = [kf.get("field", "") for kf in config.findall(".//KeyFields/Field")]
        data_fields = [df.get("field", "") for df in config.findall(".//DataFields/Field")]
        parsed["key_fields"] = key_fields
        parsed["data_fields"] = data_fields

    elif tool_type == "RecordID":
        parsed["field_name"] = _get_text(config, "FieldName")
        parsed["start_value"] = _get_text(config, "StartValue")

    elif tool_type == "AppendFields":
        parsed["mode"] = _get_text(config, "CartesianMode")

    return parsed


class AlteryxWorkflowParser:
    """Parses an Alteryx .yxmd file into a Workflow model."""

    def __init__(self, filepath: str):
        self.filepath = filepath
        self.tree = None
        self.root = None

    def parse(self) -> Workflow:
        """Parse the workflow file and return a Workflow model."""
        self.tree = ET.parse(self.filepath)
        self.root = self.tree.getroot()

        all_tools = {}
        all_containers = {}
        text_inputs = {}
        top_level_containers = []

        # ── Parse all nodes (recursive for containers) ───────────────
        nodes_element = self.root.find("Nodes")
        if nodes_element is not None:
            self._parse_nodes(
                nodes_element, None, all_tools, all_containers, text_inputs
            )

        # ── Identify top-level containers ────────────────────────────
        for cid, container in all_containers.items():
            if container.parent_container_id is None:
                top_level_containers.append(container)

        # ── Resolve child tools on each container ────────────────────
        for cid, container in all_containers.items():
            container.child_tools = [
                all_tools[tid]
                for tid in container.child_tool_ids
                if tid in all_tools
            ]

        # ── Parse connections ────────────────────────────────────────
        connections = self._parse_connections()

        # ── Build metadata ───────────────────────────────────────────
        metadata = {
            "yxmd_version": self.root.get("yxmdVer", "unknown"),
        }

        workflow = Workflow(
            containers=sorted(top_level_containers, key=lambda c: c.tool_id),
            all_containers=all_containers,
            all_tools=all_tools,
            connections=connections,
            text_inputs=text_inputs,
            metadata=metadata,
        )

        logger.info(
            f"Parsed workflow: {len(top_level_containers)} containers, "
            f"{len(all_tools)} tools, {len(connections)} connections"
        )
        return workflow

    def _parse_nodes(
        self,
        parent_element,
        parent_container_id: Optional[int],
        all_tools: dict,
        all_containers: dict,
        text_inputs: dict,
    ):
        """Recursively parse Node elements."""
        for node in parent_element.findall("Node"):
            tool_id = int(node.get("ToolID"))
            gui = node.find("GuiSettings")
            plugin = gui.get("Plugin", "") if gui is not None else ""
            tool_type = _extract_tool_type(plugin)

            # ── Container node ───────────────────────────────────────
            if tool_type == "Container":
                container = self._parse_container(
                    node, tool_id, parent_container_id
                )
                all_containers[tool_id] = container

                # Register parent → child relationship
                if parent_container_id is not None:
                    parent = all_containers.get(parent_container_id)
                    if parent:
                        parent.sub_container_ids.append(tool_id)

                # Recurse into child nodes
                child_nodes = node.find("ChildNodes")
                if child_nodes is not None:
                    self._parse_nodes(
                        child_nodes, tool_id,
                        all_tools, all_containers, text_inputs
                    )
                continue

            # ── TextBox (annotation-only, skip) ──────────────────────
            if "TextBox" in plugin:
                continue

            # ── Regular tool node ────────────────────────────────────
            tool = self._parse_tool(node, tool_id, plugin, tool_type, parent_container_id)
            all_tools[tool_id] = tool

            # Register in parent container
            if parent_container_id is not None:
                parent = all_containers.get(parent_container_id)
                if parent:
                    parent.child_tool_ids.append(tool_id)

            # ── Extract TextInput inline data ────────────────────────
            if tool_type == "TextInput":
                data = self._parse_text_input_data(node)
                if data:
                    text_inputs[tool_id] = data

    def _parse_container(
        self, node, tool_id: int, parent_id: Optional[int]
    ) -> Container:
        """Parse a ToolContainer node."""
        props = node.find("Properties")
        config = props.find("Configuration") if props is not None else None

        name = ""
        disabled = False
        style = {}

        if config is not None:
            name = _get_text(config, "Caption")
            disabled = _get_attr(config, "Disabled", "value") == "True"
            style_el = config.find("Style")
            if style_el is not None:
                style = dict(style_el.attrib)

        return Container(
            tool_id=tool_id,
            name=name or f"Container_{tool_id}",
            parent_container_id=parent_id,
            disabled=disabled,
            style=style,
        )

    def _parse_tool(
        self, node, tool_id: int, plugin: str,
        tool_type: str, container_id: Optional[int]
    ) -> Tool:
        """Parse a regular tool node."""
        gui = node.find("GuiSettings")
        pos_el = gui.find("Position") if gui is not None else None
        position = {}
        if pos_el is not None:
            position = {
                "x": int(pos_el.get("x", 0)),
                "y": int(pos_el.get("y", 0)),
            }

        # Configuration XML
        props = node.find("Properties")
        config = props.find("Configuration") if props is not None else None
        config_xml = ""
        if config is not None:
            config_xml = ET.tostring(config, encoding="unicode", method="xml")

        # Extract structured configuration from XML
        parsed_config = _extract_structured_config(config, tool_type)

        # Annotation
        annotation = ""
        if props is not None:
            ann = props.find("Annotation")
            if ann is not None:
                annotation = (
                    _get_text(ann, "AnnotationText")
                    or _get_text(ann, "DefaultAnnotationText")
                    or _get_text(ann, "Name")
                )

        return Tool(
            tool_id=tool_id,
            plugin=plugin,
            tool_type=tool_type,
            position=position,
            configuration_xml=config_xml,
            annotation=annotation,
            container_id=container_id,
            parsed_config=parsed_config,
        )

    def _parse_text_input_data(self, node) -> list:
        """Extract inline data from a TextInput tool."""
        props = node.find("Properties")
        config = props.find("Configuration") if props is not None else None
        if config is None:
            return []

        fields = []
        fields_el = config.find("Fields")
        if fields_el is not None:
            for f in fields_el.findall("Field"):
                fields.append(f.get("name", ""))

        rows = []
        data_el = config.find("Data")
        if data_el is not None:
            for row_el in data_el.findall("r"):
                row = {}
                cells = row_el.findall("c")
                for i, cell in enumerate(cells):
                    if i < len(fields):
                        row[fields[i]] = cell.text or ""
                rows.append(row)

        return rows

    def _parse_connections(self) -> list:
        """Parse all Connection elements."""
        connections = []
        conn_element = self.root.find("Connections")
        if conn_element is None:
            return connections

        for conn in conn_element.findall("Connection"):
            wireless = conn.get("Wireless", "False") == "True"
            origin = conn.find("Origin")
            dest = conn.find("Destination")
            if origin is not None and dest is not None:
                connections.append(Connection(
                    origin_tool_id=int(origin.get("ToolID")),
                    origin_connection=origin.get("Connection", "Output"),
                    dest_tool_id=int(dest.get("ToolID")),
                    dest_connection=dest.get("Connection", "Input"),
                    wireless=wireless,
                ))

        logger.info(f"Parsed {len(connections)} connections")
        return connections
