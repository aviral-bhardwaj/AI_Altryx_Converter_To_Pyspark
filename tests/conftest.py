"""
Shared test fixtures for the Alteryx to PySpark converter test suite.
"""

import pytest

from src.models import Workflow, Container, Tool, Connection


@pytest.fixture
def sample_tool_filter():
    """A sample Filter tool."""
    return Tool(
        tool_id=10,
        plugin="AlteryxBasePluginsGui.Filter.Filter",
        tool_type="Filter",
        position={"x": 200, "y": 100},
        configuration_xml='<Configuration><Expression>[Status] = "Active"</Expression></Configuration>',
        annotation="Filter Active Records",
        parsed_config={"expression": '[Status] = "Active"', "mode": "Custom"},
    )


@pytest.fixture
def sample_tool_formula():
    """A sample Formula tool."""
    return Tool(
        tool_id=20,
        plugin="AlteryxBasePluginsGui.Formula.Formula",
        tool_type="Formula",
        position={"x": 300, "y": 100},
        configuration_xml="<Configuration/>",
        annotation="Calculate Score",
        parsed_config={
            "formulas": [
                {"field": "score_category", "expression": 'IF [Score] > 8 THEN "High" ELSE "Low" ENDIF', "type": "V_String", "size": "20"},
                {"field": "full_name", "expression": "Trim([FirstName]) + ' ' + Trim([LastName])", "type": "V_String", "size": "100"},
            ]
        },
    )


@pytest.fixture
def sample_tool_input():
    """A sample InputData tool."""
    return Tool(
        tool_id=1,
        plugin="AlteryxBasePluginsGui.DbFileInput.DbFileInput",
        tool_type="InputData",
        position={"x": 50, "y": 100},
        configuration_xml="<Configuration/>",
        annotation="Load Customers",
        parsed_config={
            "table_name": "catalog.schema.customers",
            "fields": [
                {"name": "id", "type": "Int64"},
                {"name": "name", "type": "V_String"},
                {"name": "status", "type": "V_String"},
            ],
        },
    )


@pytest.fixture
def sample_tool_output():
    """A sample OutputData tool."""
    return Tool(
        tool_id=99,
        plugin="AlteryxBasePluginsGui.DbFileOutput.DbFileOutput",
        tool_type="OutputData",
        position={"x": 800, "y": 100},
        configuration_xml="<Configuration/>",
        annotation="Write Results",
        parsed_config={"table_name": "catalog.schema.output_table"},
    )


@pytest.fixture
def sample_tool_join():
    """A sample Join tool."""
    return Tool(
        tool_id=30,
        plugin="AlteryxBasePluginsGui.Join.Join",
        tool_type="Join",
        position={"x": 400, "y": 100},
        configuration_xml="<Configuration/>",
        annotation="Join Customer Orders",
        parsed_config={
            "left_keys": ["customer_id"],
            "right_keys": ["cust_id"],
            "join_by_position": "False",
            "select_config": [
                {"field": "Right_cust_id", "selected": "False", "rename": ""},
            ],
        },
    )


@pytest.fixture
def sample_tool_summarize():
    """A sample Summarize tool."""
    return Tool(
        tool_id=40,
        plugin="AlteryxBasePluginsGui.Summarize.Summarize",
        tool_type="Summarize",
        position={"x": 500, "y": 100},
        configuration_xml="<Configuration/>",
        annotation="Aggregate Sales",
        parsed_config={
            "summarize_fields": [
                {"field": "region", "action": "GroupBy", "rename": ""},
                {"field": "amount", "action": "Sum", "rename": "total_amount"},
                {"field": "order_id", "action": "Count", "rename": "order_count"},
                {"field": "customer_id", "action": "CountDistinct", "rename": "unique_customers"},
            ]
        },
    )


@pytest.fixture
def sample_tool_sort():
    """A sample Sort tool."""
    return Tool(
        tool_id=50,
        plugin="AlteryxBasePluginsGui.Sort.Sort",
        tool_type="Sort",
        position={"x": 600, "y": 100},
        configuration_xml="<Configuration/>",
        annotation="Sort by Date",
        parsed_config={
            "sort_fields": [
                {"field": "date", "order": "Descending"},
                {"field": "name", "order": "Ascending"},
            ]
        },
    )


@pytest.fixture
def sample_tool_select():
    """A sample Select tool."""
    return Tool(
        tool_id=15,
        plugin="AlteryxBasePluginsGui.AlteryxSelect.AlteryxSelect",
        tool_type="Select",
        position={"x": 250, "y": 100},
        configuration_xml="<Configuration/>",
        annotation="Rename Columns",
        parsed_config={
            "select_fields": [
                {"field": "old_name", "selected": "True", "rename": "new_name", "type": ""},
                {"field": "internal_col", "selected": "False", "rename": "", "type": ""},
                {"field": "score", "selected": "True", "rename": "", "type": "Double"},
            ]
        },
    )


@pytest.fixture
def sample_tool_unique():
    """A sample Unique tool."""
    return Tool(
        tool_id=60,
        plugin="AlteryxBasePluginsGui.Unique.Unique",
        tool_type="Unique",
        position={"x": 700, "y": 100},
        configuration_xml="<Configuration/>",
        annotation="Deduplicate by ID",
        parsed_config={"unique_fields": ["customer_id", "order_date"]},
    )


@pytest.fixture
def sample_connections():
    """A set of sample connections for a linear workflow."""
    return [
        Connection(origin_tool_id=1, origin_connection="Output", dest_tool_id=10, dest_connection="Input"),
        Connection(origin_tool_id=10, origin_connection="True", dest_tool_id=20, dest_connection="Input"),
        Connection(origin_tool_id=20, origin_connection="Output", dest_tool_id=99, dest_connection="Input"),
    ]


@pytest.fixture
def sample_workflow(sample_tool_input, sample_tool_filter, sample_tool_formula, sample_tool_output, sample_connections):
    """A simple linear workflow: Input -> Filter -> Formula -> Output."""
    tools = [sample_tool_input, sample_tool_filter, sample_tool_formula, sample_tool_output]
    return Workflow(
        containers=[],
        all_containers={},
        all_tools={t.tool_id: t for t in tools},
        connections=sample_connections,
        text_inputs={},
        metadata={"yxmd_version": "2024.1"},
    )
