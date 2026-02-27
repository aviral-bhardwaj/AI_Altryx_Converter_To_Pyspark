"""
Tests for the Alteryx .yxmd workflow parser.
"""

import os
import tempfile
import pytest
import xml.etree.ElementTree as ET

from src.parser import (
    AlteryxWorkflowParser,
    _extract_tool_type,
    _get_text,
    _get_attr,
    _extract_select_fields,
    _extract_structured_config,
    PLUGIN_TYPE_MAP,
)
from src.models import Workflow, Container, Tool, Connection


# ── Plugin Type Extraction ────────────────────────────────────────

class TestExtractToolType:
    def test_filter(self):
        assert _extract_tool_type("AlteryxBasePluginsGui.Filter.Filter") == "Filter"

    def test_formula(self):
        assert _extract_tool_type("AlteryxBasePluginsGui.Formula.Formula") == "Formula"

    def test_join(self):
        assert _extract_tool_type("AlteryxBasePluginsGui.Join.Join") == "Join"

    def test_select(self):
        assert _extract_tool_type("AlteryxBasePluginsGui.AlteryxSelect.AlteryxSelect") == "Select"

    def test_input_data(self):
        assert _extract_tool_type("AlteryxBasePluginsGui.DbFileInput.DbFileInput") == "InputData"

    def test_output_data(self):
        assert _extract_tool_type("AlteryxBasePluginsGui.DbFileOutput.DbFileOutput") == "OutputData"

    def test_union(self):
        assert _extract_tool_type("AlteryxBasePluginsGui.Union.Union") == "Union"

    def test_summarize(self):
        assert _extract_tool_type("AlteryxBasePluginsGui.Summarize.Summarize") == "Summarize"

    def test_container(self):
        assert _extract_tool_type("AlteryxGuiToolkit.ToolContainer.ToolContainer") == "Container"

    def test_unknown_plugin(self):
        result = _extract_tool_type("SomeVendor.NewTool.NewTool")
        assert result == "NewTool"  # Falls back to last dotted segment

    def test_empty_string(self):
        assert _extract_tool_type("") == "Unknown"

    def test_none(self):
        assert _extract_tool_type(None) == "Unknown"

    def test_all_known_types(self):
        """Verify all entries in PLUGIN_TYPE_MAP produce their mapped values."""
        for key, expected in PLUGIN_TYPE_MAP.items():
            result = _extract_tool_type(f"SomePrefix.{key}.{key}")
            assert result == expected, f"Expected {expected} for {key}, got {result}"


# ── XML Helpers ───────────────────────────────────────────────────

class TestXMLHelpers:
    def test_get_text_existing(self):
        root = ET.fromstring("<Root><Child>Hello</Child></Root>")
        assert _get_text(root, "Child") == "Hello"

    def test_get_text_missing(self):
        root = ET.fromstring("<Root></Root>")
        assert _get_text(root, "Child") == ""

    def test_get_text_default(self):
        root = ET.fromstring("<Root></Root>")
        assert _get_text(root, "Child", "default") == "default"

    def test_get_text_empty(self):
        root = ET.fromstring("<Root><Child></Child></Root>")
        assert _get_text(root, "Child") == ""

    def test_get_attr_existing(self):
        root = ET.fromstring('<Root><Child key="val"/></Root>')
        assert _get_attr(root, "Child", "key") == "val"

    def test_get_attr_missing(self):
        root = ET.fromstring("<Root></Root>")
        assert _get_attr(root, "Child", "key") == ""


# ── Structured Config Extraction ─────────────────────────────────

class TestStructuredConfig:
    def test_filter_config(self):
        config = ET.fromstring('<Configuration><Expression>[Status] = "Active"</Expression></Configuration>')
        result = _extract_structured_config(config, "Filter")
        assert result["expression"] == '[Status] = "Active"'

    def test_formula_config(self):
        config = ET.fromstring(
            '<Configuration>'
            '<FormulaFields>'
            '<FormulaField field="result" expression="[A] + [B]" type="Double"/>'
            '</FormulaFields>'
            '</Configuration>'
        )
        result = _extract_structured_config(config, "Formula")
        assert len(result["formulas"]) == 1
        assert result["formulas"][0]["field"] == "result"

    def test_join_config(self):
        config = ET.fromstring(
            '<Configuration>'
            '<JoinInfo connection="Left"><Field field="id"/></JoinInfo>'
            '<JoinInfo connection="Right"><Field field="id"/></JoinInfo>'
            '</Configuration>'
        )
        result = _extract_structured_config(config, "Join")
        assert result["left_keys"] == ["id"]
        assert result["right_keys"] == ["id"]

    def test_summarize_config(self):
        config = ET.fromstring(
            '<Configuration>'
            '<SummarizeFields>'
            '<SummarizeField field="region" action="GroupBy"/>'
            '<SummarizeField field="amount" action="Sum" rename="total"/>'
            '</SummarizeFields>'
            '</Configuration>'
        )
        result = _extract_structured_config(config, "Summarize")
        assert len(result["summarize_fields"]) == 2
        assert result["summarize_fields"][0]["action"] == "GroupBy"

    def test_sort_config(self):
        config = ET.fromstring(
            '<Configuration>'
            '<SortInfo>'
            '<Field field="date" order="Descending"/>'
            '<Field field="name" order="Ascending"/>'
            '</SortInfo>'
            '</Configuration>'
        )
        result = _extract_structured_config(config, "Sort")
        assert len(result["sort_fields"]) == 2
        assert result["sort_fields"][0]["order"] == "Descending"

    def test_unique_config(self):
        config = ET.fromstring(
            '<Configuration>'
            '<UniqueFields>'
            '<Field field="id"/>'
            '<Field field="name"/>'
            '</UniqueFields>'
            '</Configuration>'
        )
        result = _extract_structured_config(config, "Unique")
        assert result["unique_fields"] == ["id", "name"]

    def test_input_data_config(self):
        config = ET.fromstring(
            '<Configuration>'
            '<TableName>catalog.schema.table</TableName>'
            '</Configuration>'
        )
        result = _extract_structured_config(config, "InputData")
        assert result["table_name"] == "catalog.schema.table"

    def test_transpose_config(self):
        config = ET.fromstring(
            '<Configuration>'
            '<KeyFields><Field field="id"/></KeyFields>'
            '<DataFields><Field field="q1"/><Field field="q2"/></DataFields>'
            '</Configuration>'
        )
        result = _extract_structured_config(config, "Transpose")
        assert result["key_fields"] == ["id"]
        assert result["data_fields"] == ["q1", "q2"]

    def test_record_id_config(self):
        config = ET.fromstring(
            '<Configuration>'
            '<FieldName>RecordID</FieldName>'
            '<StartValue>1</StartValue>'
            '</Configuration>'
        )
        result = _extract_structured_config(config, "RecordID")
        assert result["field_name"] == "RecordID"
        assert result["start_value"] == "1"

    def test_none_config(self):
        result = _extract_structured_config(None, "Filter")
        assert result == {}

    def test_unknown_type(self):
        config = ET.fromstring("<Configuration><SomeProp>value</SomeProp></Configuration>")
        result = _extract_structured_config(config, "UnknownTool")
        assert result == {}


# ── Select Fields Extraction ──────────────────────────────────────

class TestExtractSelectFields:
    def test_basic_select_fields(self):
        config = ET.fromstring(
            '<Configuration>'
            '<SelectField field="col1" selected="True" rename="" type=""/>'
            '<SelectField field="col2" selected="False" rename="" type=""/>'
            '</Configuration>'
        )
        fields = _extract_select_fields(config)
        assert len(fields) == 2
        assert fields[0]["field"] == "col1"
        assert fields[1]["selected"] == "False"

    def test_nested_select_configuration(self):
        config = ET.fromstring(
            '<Configuration>'
            '<SelectConfiguration>'
            '<Configuration>'
            '<OrderChanged>'
            '<SelectField field="col1" selected="True" rename="" type=""/>'
            '</OrderChanged>'
            '</Configuration>'
            '</SelectConfiguration>'
            '</Configuration>'
        )
        fields = _extract_select_fields(config)
        assert len(fields) >= 1


# ── Full Parser Integration ──────────────────────────────────────

class TestAlteryxWorkflowParser:
    def _write_workflow_xml(self, xml_content: str) -> str:
        """Write XML to a temp file and return the path."""
        fd, path = tempfile.mkstemp(suffix=".yxmd")
        with os.fdopen(fd, "w") as f:
            f.write(xml_content)
        return path

    def test_parse_minimal_workflow(self):
        xml = """<?xml version="1.0"?>
<AlteryxDocument yxmdVer="2024.1">
  <Nodes>
    <Node ToolID="1">
      <GuiSettings Plugin="AlteryxBasePluginsGui.DbFileInput.DbFileInput">
        <Position x="100" y="100"/>
      </GuiSettings>
      <Properties>
        <Configuration>
          <TableName>test_table</TableName>
        </Configuration>
        <Annotation>
          <DefaultAnnotationText>Load Data</DefaultAnnotationText>
        </Annotation>
      </Properties>
    </Node>
  </Nodes>
  <Connections/>
</AlteryxDocument>"""
        path = self._write_workflow_xml(xml)
        try:
            parser = AlteryxWorkflowParser(path)
            workflow = parser.parse()

            assert isinstance(workflow, Workflow)
            assert len(workflow.all_tools) == 1
            assert 1 in workflow.all_tools
            tool = workflow.all_tools[1]
            assert tool.tool_type == "InputData"
            assert tool.annotation == "Load Data"
            assert tool.parsed_config["table_name"] == "test_table"
        finally:
            os.unlink(path)

    def test_parse_connections(self):
        xml = """<?xml version="1.0"?>
<AlteryxDocument yxmdVer="2024.1">
  <Nodes>
    <Node ToolID="1">
      <GuiSettings Plugin="AlteryxBasePluginsGui.DbFileInput.DbFileInput">
        <Position x="100" y="100"/>
      </GuiSettings>
      <Properties>
        <Configuration><TableName>t</TableName></Configuration>
        <Annotation><DefaultAnnotationText/></Annotation>
      </Properties>
    </Node>
    <Node ToolID="2">
      <GuiSettings Plugin="AlteryxBasePluginsGui.Filter.Filter">
        <Position x="200" y="100"/>
      </GuiSettings>
      <Properties>
        <Configuration><Expression>[x] = 1</Expression></Configuration>
        <Annotation><DefaultAnnotationText/></Annotation>
      </Properties>
    </Node>
  </Nodes>
  <Connections>
    <Connection>
      <Origin ToolID="1" Connection="Output"/>
      <Destination ToolID="2" Connection="Input"/>
    </Connection>
  </Connections>
</AlteryxDocument>"""
        path = self._write_workflow_xml(xml)
        try:
            parser = AlteryxWorkflowParser(path)
            workflow = parser.parse()

            assert len(workflow.connections) == 1
            conn = workflow.connections[0]
            assert conn.origin_tool_id == 1
            assert conn.dest_tool_id == 2
            assert conn.origin_connection == "Output"
            assert conn.dest_connection == "Input"
        finally:
            os.unlink(path)

    def test_parse_container(self):
        xml = """<?xml version="1.0"?>
<AlteryxDocument yxmdVer="2024.1">
  <Nodes>
    <Node ToolID="100">
      <GuiSettings Plugin="AlteryxGuiToolkit.ToolContainer.ToolContainer">
        <Position x="50" y="50"/>
      </GuiSettings>
      <Properties>
        <Configuration>
          <Caption>My Container</Caption>
        </Configuration>
      </Properties>
      <ChildNodes>
        <Node ToolID="1">
          <GuiSettings Plugin="AlteryxBasePluginsGui.Filter.Filter">
            <Position x="100" y="100"/>
          </GuiSettings>
          <Properties>
            <Configuration><Expression>[x] = 1</Expression></Configuration>
            <Annotation><DefaultAnnotationText>Filter</DefaultAnnotationText></Annotation>
          </Properties>
        </Node>
      </ChildNodes>
    </Node>
  </Nodes>
  <Connections/>
</AlteryxDocument>"""
        path = self._write_workflow_xml(xml)
        try:
            parser = AlteryxWorkflowParser(path)
            workflow = parser.parse()

            assert len(workflow.containers) == 1
            container = workflow.containers[0]
            assert container.name == "My Container"
            assert len(container.child_tool_ids) == 1
            assert container.child_tool_ids[0] == 1
            tool = workflow.all_tools[1]
            assert tool.container_id == 100
        finally:
            os.unlink(path)

    def test_parse_text_input(self):
        xml = """<?xml version="1.0"?>
<AlteryxDocument yxmdVer="2024.1">
  <Nodes>
    <Node ToolID="5">
      <GuiSettings Plugin="AlteryxBasePluginsGui.TextInput.TextInput">
        <Position x="100" y="100"/>
      </GuiSettings>
      <Properties>
        <Configuration>
          <Fields>
            <Field name="id"/>
            <Field name="value"/>
          </Fields>
          <Data>
            <r><c>1</c><c>alpha</c></r>
            <r><c>2</c><c>beta</c></r>
          </Data>
        </Configuration>
        <Annotation><DefaultAnnotationText/></Annotation>
      </Properties>
    </Node>
  </Nodes>
  <Connections/>
</AlteryxDocument>"""
        path = self._write_workflow_xml(xml)
        try:
            parser = AlteryxWorkflowParser(path)
            workflow = parser.parse()

            assert 5 in workflow.text_inputs
            data = workflow.text_inputs[5]
            assert len(data) == 2
            assert data[0]["id"] == "1"
            assert data[0]["value"] == "alpha"
            assert data[1]["id"] == "2"
        finally:
            os.unlink(path)

    def test_workflow_methods(self):
        """Test Workflow model helper methods."""
        xml = """<?xml version="1.0"?>
<AlteryxDocument yxmdVer="2024.1">
  <Nodes>
    <Node ToolID="1">
      <GuiSettings Plugin="AlteryxBasePluginsGui.DbFileInput.DbFileInput">
        <Position x="100" y="100"/>
      </GuiSettings>
      <Properties>
        <Configuration><TableName>t</TableName></Configuration>
        <Annotation><DefaultAnnotationText>Input</DefaultAnnotationText></Annotation>
      </Properties>
    </Node>
    <Node ToolID="2">
      <GuiSettings Plugin="AlteryxBasePluginsGui.Filter.Filter">
        <Position x="200" y="100"/>
      </GuiSettings>
      <Properties>
        <Configuration><Expression>[x] = 1</Expression></Configuration>
        <Annotation><DefaultAnnotationText>Filter</DefaultAnnotationText></Annotation>
      </Properties>
    </Node>
  </Nodes>
  <Connections>
    <Connection>
      <Origin ToolID="1" Connection="Output"/>
      <Destination ToolID="2" Connection="Input"/>
    </Connection>
  </Connections>
</AlteryxDocument>"""
        path = self._write_workflow_xml(xml)
        try:
            parser = AlteryxWorkflowParser(path)
            workflow = parser.parse()

            # Test get_tool
            assert workflow.get_tool(1).tool_type == "InputData"
            assert workflow.get_tool(999) is None

            # Test get_incoming/outgoing connections
            incoming = workflow.get_incoming_connections(2)
            assert len(incoming) == 1
            assert incoming[0].origin_tool_id == 1

            outgoing = workflow.get_outgoing_connections(1)
            assert len(outgoing) == 1
            assert outgoing[0].dest_tool_id == 2

            # Test get_root_tools (both are root-level)
            assert len(workflow.get_root_tools()) == 2

            # Test get_unified_context
            ctx = workflow.get_unified_context()
            assert len(ctx["tools"]) == 2
            assert len(ctx["internal_connections"]) == 1
        finally:
            os.unlink(path)

    def test_textbox_skipped(self):
        """TextBox nodes should be skipped (not added as tools)."""
        xml = """<?xml version="1.0"?>
<AlteryxDocument yxmdVer="2024.1">
  <Nodes>
    <Node ToolID="1">
      <GuiSettings Plugin="AlteryxGuiToolkit.TextBox.TextBox">
        <Position x="100" y="100"/>
      </GuiSettings>
      <Properties>
        <Configuration/>
        <Annotation><DefaultAnnotationText>Note</DefaultAnnotationText></Annotation>
      </Properties>
    </Node>
  </Nodes>
  <Connections/>
</AlteryxDocument>"""
        path = self._write_workflow_xml(xml)
        try:
            parser = AlteryxWorkflowParser(path)
            workflow = parser.parse()
            assert len(workflow.all_tools) == 0
        finally:
            os.unlink(path)
