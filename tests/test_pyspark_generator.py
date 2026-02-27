"""
Tests for the deterministic PySpark code generator.
"""

import pytest
from src.models import Workflow, Container, Tool, Connection
from src.pyspark_generator import (
    PySparkCodeGenerator,
    GeneratorContext,
    topological_sort,
    InputDataConverter,
    TextInputConverter,
    OutputDataConverter,
    FilterConverter,
    FormulaConverter,
    SelectConverter,
    JoinConverter,
    UnionConverter,
    SummarizeConverter,
    SortConverter,
    UniqueConverter,
    SampleConverter,
    CrossTabConverter,
    TransposeConverter,
    MultiRowFormulaConverter,
    RegExConverter,
    RecordIDConverter,
    AppendFieldsConverter,
    BrowseConverter,
    PassthroughConverter,
    get_converter,
)


# ── Helpers ────────────────────────────────────────────────────────

def _make_workflow(tools, connections, text_inputs=None):
    """Helper to build a Workflow from lists of tools and connections."""
    return Workflow(
        containers=[],
        all_containers={},
        all_tools={t.tool_id: t for t in tools},
        connections=connections,
        text_inputs=text_inputs or {},
    )


def _make_ctx(tools, connections, workflow=None, source_tables=None):
    """Helper to build a GeneratorContext."""
    wf = workflow or _make_workflow(tools, connections)
    return GeneratorContext(
        workflow=wf,
        tools=tools,
        connections=connections,
        source_tables_config=source_tables or {},
    )


# ── Topological Sort ──────────────────────────────────────────────

class TestTopologicalSort:
    def test_linear_chain(self):
        tools = [
            Tool(1, "", "InputData", {}, "", ""),
            Tool(2, "", "Filter", {}, "", ""),
            Tool(3, "", "Formula", {}, "", ""),
        ]
        conns = [
            Connection(1, "Output", 2, "Input"),
            Connection(2, "True", 3, "Input"),
        ]
        order = topological_sort(tools, conns)
        assert order == [1, 2, 3]

    def test_diamond_shape(self):
        tools = [
            Tool(1, "", "InputData", {}, "", ""),
            Tool(2, "", "Filter", {}, "", ""),
            Tool(3, "", "Formula", {}, "", ""),
            Tool(4, "", "Join", {}, "", ""),
        ]
        conns = [
            Connection(1, "Output", 2, "Input"),
            Connection(1, "Output", 3, "Input"),
            Connection(2, "Output", 4, "Left"),
            Connection(3, "Output", 4, "Right"),
        ]
        order = topological_sort(tools, conns)
        assert order[0] == 1  # Input first
        assert order[-1] == 4  # Join last
        assert set(order) == {1, 2, 3, 4}

    def test_no_connections(self):
        tools = [
            Tool(1, "", "InputData", {}, "", ""),
            Tool(2, "", "InputData", {}, "", ""),
        ]
        order = topological_sort(tools, [])
        assert set(order) == {1, 2}

    def test_single_tool(self):
        tools = [Tool(1, "", "InputData", {}, "", "")]
        order = topological_sort(tools, [])
        assert order == [1]


# ── GeneratorContext ──────────────────────────────────────────────

class TestGeneratorContext:
    def test_set_and_get_output_var(self):
        tools = [Tool(1, "", "InputData", {}, "", "")]
        ctx = _make_ctx(tools, [])
        ctx.set_output_var(1, "df_customers")
        assert ctx.get_output_var(1) == "df_customers"

    def test_get_input_var(self):
        tools = [
            Tool(1, "", "InputData", {}, "", ""),
            Tool(2, "", "Filter", {}, "", ""),
        ]
        conns = [Connection(1, "Output", 2, "Input")]
        ctx = _make_ctx(tools, conns)
        ctx.set_output_var(1, "df_customers")
        assert ctx.get_input_var(2) == "df_customers"

    def test_is_port_connected(self):
        tools = [
            Tool(10, "", "Filter", {}, "", ""),
            Tool(20, "", "Formula", {}, "", ""),
        ]
        conns = [Connection(10, "True", 20, "Input")]
        ctx = _make_ctx(tools, conns)
        assert ctx.is_port_connected(10, "True") is True
        assert ctx.is_port_connected(10, "False") is False

    def test_make_var_name_from_annotation(self):
        tool = Tool(1, "", "InputData", {}, "", "Load Customer Data")
        tools = [tool]
        ctx = _make_ctx(tools, [])
        assert ctx.make_var_name(tool) == "df_load_customer_data"

    def test_make_var_name_from_table(self):
        tool = Tool(1, "", "InputData", {}, "", "", parsed_config={"table_name": "catalog.schema.customers"})
        tools = [tool]
        ctx = _make_ctx(tools, [])
        assert ctx.make_var_name(tool) == "df_customers"

    def test_make_var_name_fallback(self):
        tool = Tool(42, "", "Formula", {}, "", "")
        tools = [tool]
        ctx = _make_ctx(tools, [])
        assert ctx.make_var_name(tool) == "df_42"


# ── Individual Converters ─────────────────────────────────────────

class TestInputDataConverter:
    def test_basic_table(self, sample_tool_input):
        ctx = _make_ctx([sample_tool_input], [])
        lines = InputDataConverter().convert(sample_tool_input, ctx)
        code = "\n".join(lines)
        assert "spark.table" in code
        assert "catalog.schema.customers" in code

    def test_csv_file(self):
        tool = Tool(1, "", "InputData", {}, "", "", parsed_config={"table_name": "/data/file.csv"})
        ctx = _make_ctx([tool], [])
        lines = InputDataConverter().convert(tool, ctx)
        code = "\n".join(lines)
        assert "spark.read.csv" in code

    def test_excel_file(self):
        tool = Tool(1, "", "InputData", {}, "", "", parsed_config={"table_name": "/data/file.xlsx"})
        ctx = _make_ctx([tool], [])
        lines = InputDataConverter().convert(tool, ctx)
        code = "\n".join(lines)
        assert "com.crealytics.spark.excel" in code

    def test_parquet_file(self):
        tool = Tool(1, "", "InputData", {}, "", "", parsed_config={"table_name": "/data/file.parquet"})
        ctx = _make_ctx([tool], [])
        lines = InputDataConverter().convert(tool, ctx)
        code = "\n".join(lines)
        assert "spark.read.parquet" in code

    def test_source_tables_config(self):
        tool = Tool(1, "", "InputData", {}, "", "customers", parsed_config={"table_name": "old_table"})
        ctx = _make_ctx([tool], [], source_tables={"1": "gold.customers"})
        lines = InputDataConverter().convert(tool, ctx)
        code = "\n".join(lines)
        assert "gold.customers" in code

    def test_unknown_table(self):
        tool = Tool(1, "", "InputData", {}, "", "", parsed_config={})
        ctx = _make_ctx([tool], [])
        lines = InputDataConverter().convert(tool, ctx)
        code = "\n".join(lines)
        assert "TODO" in code


class TestTextInputConverter:
    def test_with_data(self):
        tool = Tool(5, "", "TextInput", {}, "", "Lookup")
        wf = _make_workflow([tool], [], text_inputs={5: [{"id": "1", "name": "Alice"}, {"id": "2", "name": "Bob"}]})
        ctx = _make_ctx([tool], [], workflow=wf)
        lines = TextInputConverter().convert(tool, ctx)
        code = "\n".join(lines)
        assert "spark.createDataFrame" in code
        assert "Alice" in code
        assert "Bob" in code

    def test_empty_data(self):
        tool = Tool(5, "", "TextInput", {}, "", "Empty")
        wf = _make_workflow([tool], [], text_inputs={})
        ctx = _make_ctx([tool], [], workflow=wf)
        lines = TextInputConverter().convert(tool, ctx)
        code = "\n".join(lines)
        assert "createDataFrame" in code


class TestOutputDataConverter:
    def test_basic_output(self, sample_tool_output):
        tools = [Tool(1, "", "InputData", {}, "", ""), sample_tool_output]
        conns = [Connection(1, "Output", 99, "Input")]
        ctx = _make_ctx(tools, conns)
        ctx.set_output_var(1, "df_input")
        lines = OutputDataConverter().convert(sample_tool_output, ctx)
        code = "\n".join(lines)
        assert "createOrReplaceTempView" in code
        assert "delta" in code or "saveAsTable" in code


class TestFilterConverter:
    def test_true_output_only(self, sample_tool_filter):
        tools = [Tool(1, "", "InputData", {}, "", ""), sample_tool_filter, Tool(20, "", "Formula", {}, "", "")]
        conns = [
            Connection(1, "Output", 10, "Input"),
            Connection(10, "True", 20, "Input"),
        ]
        ctx = _make_ctx(tools, conns)
        ctx.set_output_var(1, "df_input")
        lines = FilterConverter().convert(sample_tool_filter, ctx)
        code = "\n".join(lines)
        assert ".filter(" in code
        assert "df_10_true" in code

    def test_both_outputs(self, sample_tool_filter):
        tools = [
            Tool(1, "", "InputData", {}, "", ""),
            sample_tool_filter,
            Tool(20, "", "Formula", {}, "", ""),
            Tool(21, "", "Formula", {}, "", ""),
        ]
        conns = [
            Connection(1, "Output", 10, "Input"),
            Connection(10, "True", 20, "Input"),
            Connection(10, "False", 21, "Input"),
        ]
        ctx = _make_ctx(tools, conns)
        ctx.set_output_var(1, "df_input")
        lines = FilterConverter().convert(sample_tool_filter, ctx)
        code = "\n".join(lines)
        assert "df_10_true" in code
        assert "df_10_false" in code
        assert "~(" in code  # Negation for False output


class TestFormulaConverter:
    def test_basic_formula(self, sample_tool_formula):
        tools = [Tool(1, "", "InputData", {}, "", ""), sample_tool_formula]
        conns = [Connection(1, "Output", 20, "Input")]
        ctx = _make_ctx(tools, conns)
        ctx.set_output_var(1, "df_input")
        lines = FormulaConverter().convert(sample_tool_formula, ctx)
        code = "\n".join(lines)
        assert ".withColumn(" in code
        assert "score_category" in code
        assert "full_name" in code

    def test_empty_formulas(self):
        tool = Tool(20, "", "Formula", {}, "", "", parsed_config={"formulas": []})
        ctx = _make_ctx([tool], [])
        lines = FormulaConverter().convert(tool, ctx)
        code = "\n".join(lines)
        assert "no formulas" in code


class TestSelectConverter:
    def test_rename_and_drop(self, sample_tool_select):
        tools = [Tool(1, "", "InputData", {}, "", ""), sample_tool_select]
        conns = [Connection(1, "Output", 15, "Input")]
        ctx = _make_ctx(tools, conns)
        ctx.set_output_var(1, "df_input")
        lines = SelectConverter().convert(sample_tool_select, ctx)
        code = "\n".join(lines)
        assert "withColumnRenamed" in code
        assert "old_name" in code
        assert "new_name" in code
        assert ".drop(" in code
        assert "internal_col" in code

    def test_type_cast(self, sample_tool_select):
        tools = [Tool(1, "", "InputData", {}, "", ""), sample_tool_select]
        conns = [Connection(1, "Output", 15, "Input")]
        ctx = _make_ctx(tools, conns)
        ctx.set_output_var(1, "df_input")
        lines = SelectConverter().convert(sample_tool_select, ctx)
        code = "\n".join(lines)
        assert '.cast("double")' in code


class TestJoinConverter:
    def test_inner_join(self, sample_tool_join):
        tools = [
            Tool(1, "", "InputData", {}, "", ""),
            Tool(2, "", "InputData", {}, "", ""),
            sample_tool_join,
            Tool(40, "", "Formula", {}, "", ""),
        ]
        conns = [
            Connection(1, "Output", 30, "Left"),
            Connection(2, "Output", 30, "Right"),
            Connection(30, "Join", 40, "Input"),
        ]
        ctx = _make_ctx(tools, conns)
        ctx.set_output_var(1, "df_left")
        ctx.set_output_var(2, "df_right")
        lines = JoinConverter().convert(sample_tool_join, ctx)
        code = "\n".join(lines)
        assert ".join(" in code
        assert '"inner"' in code
        assert "customer_id" in code
        assert "cust_id" in code

    def test_join_with_left_anti(self, sample_tool_join):
        tools = [
            Tool(1, "", "InputData", {}, "", ""),
            Tool(2, "", "InputData", {}, "", ""),
            sample_tool_join,
            Tool(40, "", "Formula", {}, "", ""),
            Tool(41, "", "Formula", {}, "", ""),
        ]
        conns = [
            Connection(1, "Output", 30, "Left"),
            Connection(2, "Output", 30, "Right"),
            Connection(30, "Join", 40, "Input"),
            Connection(30, "Left", 41, "Input"),
        ]
        ctx = _make_ctx(tools, conns)
        ctx.set_output_var(1, "df_left")
        ctx.set_output_var(2, "df_right")
        lines = JoinConverter().convert(sample_tool_join, ctx)
        code = "\n".join(lines)
        assert "inner" in code
        assert "left_anti" in code

    def test_same_key_join(self):
        tool = Tool(30, "", "Join", {}, "", "", parsed_config={
            "left_keys": ["id"], "right_keys": ["id"],
        })
        tools = [Tool(1, "", "InputData", {}, "", ""), Tool(2, "", "InputData", {}, "", ""), tool, Tool(40, "", "Formula", {}, "", "")]
        conns = [
            Connection(1, "Output", 30, "Left"),
            Connection(2, "Output", 30, "Right"),
            Connection(30, "Join", 40, "Input"),
        ]
        ctx = _make_ctx(tools, conns)
        ctx.set_output_var(1, "df_l")
        ctx.set_output_var(2, "df_r")
        lines = JoinConverter().convert(tool, ctx)
        code = "\n".join(lines)
        assert '["id"]' in code  # Uses list syntax for same-key join


class TestUnionConverter:
    def test_two_inputs(self):
        tool = Tool(30, "", "Union", {}, "", "Merge Streams")
        tools = [Tool(1, "", "InputData", {}, "", ""), Tool(2, "", "InputData", {}, "", ""), tool]
        conns = [
            Connection(1, "Output", 30, "Input"),
            Connection(2, "Output", 30, "Input"),
        ]
        ctx = _make_ctx(tools, conns)
        ctx.set_output_var(1, "df_a")
        ctx.set_output_var(2, "df_b")
        lines = UnionConverter().convert(tool, ctx)
        code = "\n".join(lines)
        assert "unionByName" in code
        assert "allowMissingColumns=True" in code

    def test_single_input(self):
        tool = Tool(30, "", "Union", {}, "", "")
        tools = [Tool(1, "", "InputData", {}, "", ""), tool]
        conns = [Connection(1, "Output", 30, "Input")]
        ctx = _make_ctx(tools, conns)
        ctx.set_output_var(1, "df_a")
        lines = UnionConverter().convert(tool, ctx)
        code = "\n".join(lines)
        assert "df_a" in code


class TestSummarizeConverter:
    def test_group_by_with_aggs(self, sample_tool_summarize):
        tools = [Tool(1, "", "InputData", {}, "", ""), sample_tool_summarize]
        conns = [Connection(1, "Output", 40, "Input")]
        ctx = _make_ctx(tools, conns)
        ctx.set_output_var(1, "df_input")
        lines = SummarizeConverter().convert(sample_tool_summarize, ctx)
        code = "\n".join(lines)
        assert ".groupBy(" in code
        assert "F.sum" in code
        assert "F.count" in code
        assert "F.countDistinct" in code
        assert "total_amount" in code
        assert "region" in code


class TestSortConverter:
    def test_multi_sort(self, sample_tool_sort):
        tools = [Tool(1, "", "InputData", {}, "", ""), sample_tool_sort]
        conns = [Connection(1, "Output", 50, "Input")]
        ctx = _make_ctx(tools, conns)
        ctx.set_output_var(1, "df_input")
        lines = SortConverter().convert(sample_tool_sort, ctx)
        code = "\n".join(lines)
        assert ".orderBy(" in code
        assert ".desc()" in code
        assert ".asc()" in code


class TestUniqueConverter:
    def test_unique(self, sample_tool_unique):
        tools = [Tool(1, "", "InputData", {}, "", ""), sample_tool_unique, Tool(70, "", "Formula", {}, "", "")]
        conns = [
            Connection(1, "Output", 60, "Input"),
            Connection(60, "Unique", 70, "Input"),
        ]
        ctx = _make_ctx(tools, conns)
        ctx.set_output_var(1, "df_input")
        lines = UniqueConverter().convert(sample_tool_unique, ctx)
        code = "\n".join(lines)
        assert ".dropDuplicates(" in code
        assert "customer_id" in code


class TestSampleConverter:
    def test_sample(self):
        tool = Tool(10, "", "Sample", {}, "", "", parsed_config={"n_records": "50"})
        tools = [Tool(1, "", "InputData", {}, "", ""), tool]
        conns = [Connection(1, "Output", 10, "Input")]
        ctx = _make_ctx(tools, conns)
        ctx.set_output_var(1, "df_input")
        lines = SampleConverter().convert(tool, ctx)
        code = "\n".join(lines)
        assert ".limit(50)" in code


class TestCrossTabConverter:
    def test_crosstab(self):
        tool = Tool(10, "", "CrossTab", {}, "", "", parsed_config={
            "group_fields": "region",
            "header_field": "product",
            "data_field": "sales",
            "method": "Sum",
        })
        tools = [Tool(1, "", "InputData", {}, "", ""), tool]
        conns = [Connection(1, "Output", 10, "Input")]
        ctx = _make_ctx(tools, conns)
        ctx.set_output_var(1, "df_input")
        lines = CrossTabConverter().convert(tool, ctx)
        code = "\n".join(lines)
        assert ".pivot(" in code
        assert "F.sum" in code


class TestTransposeConverter:
    def test_transpose(self):
        tool = Tool(10, "", "Transpose", {}, "", "", parsed_config={
            "key_fields": ["id", "name"],
            "data_fields": ["q1", "q2", "q3"],
        })
        tools = [Tool(1, "", "InputData", {}, "", ""), tool]
        conns = [Connection(1, "Output", 10, "Input")]
        ctx = _make_ctx(tools, conns)
        ctx.set_output_var(1, "df_input")
        lines = TransposeConverter().convert(tool, ctx)
        code = "\n".join(lines)
        assert "stack(" in code
        assert "cast" in code


class TestRecordIDConverter:
    def test_record_id(self):
        tool = Tool(10, "", "RecordID", {}, "", "", parsed_config={"field_name": "RowNum", "start_value": "1"})
        tools = [Tool(1, "", "InputData", {}, "", ""), tool]
        conns = [Connection(1, "Output", 10, "Input")]
        ctx = _make_ctx(tools, conns)
        ctx.set_output_var(1, "df_input")
        lines = RecordIDConverter().convert(tool, ctx)
        code = "\n".join(lines)
        assert "monotonically_increasing_id" in code
        assert "RowNum" in code


class TestBrowseConverter:
    def test_passthrough(self):
        tool = Tool(10, "", "Browse", {}, "", "")
        tools = [Tool(1, "", "InputData", {}, "", ""), tool]
        conns = [Connection(1, "Output", 10, "Input")]
        ctx = _make_ctx(tools, conns)
        ctx.set_output_var(1, "df_input")
        lines = BrowseConverter().convert(tool, ctx)
        code = "\n".join(lines)
        assert "passthrough" in code.lower()


# ── Converter Registry ────────────────────────────────────────────

class TestConverterRegistry:
    def test_known_types(self):
        known_types = [
            "InputData", "TextInput", "OutputData", "Filter", "Formula",
            "Select", "Join", "Union", "Summarize", "Sort", "Unique",
            "Sample", "CrossTab", "Transpose", "RegEx", "RecordID",
            "MultiRowFormula", "AppendFields", "Browse",
        ]
        for tt in known_types:
            converter = get_converter(tt)
            assert not isinstance(converter, PassthroughConverter), f"{tt} should have a real converter"

    def test_unknown_type_gets_passthrough(self):
        converter = get_converter("UnknownToolXYZ")
        assert isinstance(converter, PassthroughConverter)


# ── Full Generator ────────────────────────────────────────────────

class TestPySparkCodeGenerator:
    def test_generate_simple_workflow(self, sample_workflow):
        gen = PySparkCodeGenerator()
        code = gen.generate(sample_workflow, workflow_name="test_workflow")

        assert "# Databricks notebook source" in code
        assert "COMMAND ----------" in code
        assert "from pyspark.sql import functions as F" in code
        assert "spark" in code

    def test_generate_includes_all_tools(self, sample_workflow):
        gen = PySparkCodeGenerator()
        code = gen.generate(sample_workflow, workflow_name="test")

        # Should have code for Input, Filter, Formula, Output tools
        assert "Tool 1" in code
        assert "Tool 10" in code
        assert "Tool 20" in code
        assert "Tool 99" in code

    def test_generate_with_source_tables(self, sample_workflow):
        gen = PySparkCodeGenerator(source_tables_config={"1": "gold.customers"})
        code = gen.generate(sample_workflow, workflow_name="test")
        assert "gold.customers" in code

    def test_generate_has_validation_section(self, sample_workflow):
        gen = PySparkCodeGenerator()
        code = gen.generate(sample_workflow, workflow_name="test")
        assert "Validation" in code

    def test_generate_has_imports(self, sample_workflow):
        gen = PySparkCodeGenerator()
        code = gen.generate(sample_workflow, workflow_name="test")
        assert "from pyspark.sql import functions as F" in code
        assert "from pyspark.sql.window import Window" in code

    def test_generate_complexity_low(self):
        """Workflow with few tools gets 'Low' complexity."""
        tools = [
            Tool(1, "", "InputData", {}, "", "", parsed_config={"table_name": "t"}),
            Tool(2, "", "OutputData", {}, "", "", parsed_config={"table_name": "out"}),
        ]
        conns = [Connection(1, "Output", 2, "Input")]
        wf = _make_workflow(tools, conns)
        gen = PySparkCodeGenerator()
        code = gen.generate(wf, workflow_name="small")
        assert "Low" in code
