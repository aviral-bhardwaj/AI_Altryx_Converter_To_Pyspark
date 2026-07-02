"""
Data-level end-to-end tests: execute the GENERATED notebooks on a real local
Spark session and assert the output values match Alteryx semantics.

Skipped automatically when pyspark is not installed (e.g. lightweight CI);
they always run on Databricks or a dev machine with `pip install pyspark`.
"""

from pathlib import Path

import pytest

pyspark = pytest.importorskip("pyspark")

from src.converter_engine import ConverterEngine
from src.parser import AlteryxWorkflowParser
from src.self_correction import SelfCorrectingConverter
from src.validators import run_generated_code, strip_databricks_magics

SAMPLES_DIR = Path(__file__).parent / "sample_workflows"


@pytest.fixture(scope="module")
def spark():
    from pyspark.sql import SparkSession

    session = (
        SparkSession.builder.master("local[2]")
        .appName("altryx-e2e-tests")
        .config("spark.ui.enabled", "false")
        .config("spark.sql.shuffle.partitions", "2")
        .getOrCreate()
    )
    session.sparkContext.setLogLevel("ERROR")
    yield session
    session.stop()


def _exec(code, spark):
    """Execute generated notebook code, return the resulting namespace."""
    from functools import reduce

    from pyspark.sql import functions as F
    from pyspark.sql.window import Window

    ns = {"spark": spark, "F": F, "Window": Window, "reduce": reduce}
    exec(compile(strip_databricks_magics(code), "<generated>", "exec"), ns)
    return ns


def test_filter_select_data_values(spark):
    """Sample 01: Status='Active' AND Amount>100, rename + cast + drop."""
    wf = AlteryxWorkflowParser(str(SAMPLES_DIR / "01_simple_filter_select.yxmd")).parse()
    result = SelfCorrectingConverter().convert(wf, "s01", spark=spark)
    assert result.status == "PASS"
    assert result.runtime_result and result.runtime_result["ok"]

    final = _exec(result.code, spark)["df_4"]
    assert final.columns == ["OrderID", "CustomerName", "Amount"]
    assert dict(final.dtypes)["OrderID"] == "bigint"
    assert dict(final.dtypes)["Amount"] == "double"
    assert {tuple(r) for r in final.collect()} == {
        (1001, "Acme Corp", 250.0),
        (1003, "Initech", 1200.0),
    }


def test_join_summarize_data_values(spark):
    """Sample 02: inner join on CustomerID then groupBy Region+Segment."""
    txns = spark.createDataFrame(
        [(1, 10, "East", 100.0), (2, 10, "East", 50.0), (3, 20, "West", 200.0),
         (4, 30, "East", 75.0), (5, 99, "South", 999.0)],
        ["TxnID", "CustomerID", "Region", "Revenue"])
    txns.createOrReplaceTempView("transactions_src")

    wf = AlteryxWorkflowParser(str(SAMPLES_DIR / "02_join_summarize.yxmd")).parse()
    engine = ConverterEngine(source_tables_config={"1": "transactions_src"})
    result = SelfCorrectingConverter(engine=engine).convert(wf, "s02", spark=spark)
    assert result.status == "PASS"
    assert result.runtime_result and result.runtime_result["ok"]

    ns = _exec(result.code, spark)
    final = ns["df_5"]
    got = {(r["Region"], r["Segment"], r["TotalRevenue"], r["TxnCount"])
           for r in final.collect()}
    # Customer 99 has no segment row -> dropped by the inner join, like Alteryx's J port.
    assert got == {
        ("East", "Enterprise", 150.0, 2),
        ("West", "SMB", 200.0, 1),
        ("East", "Consumer", 75.0, 1),
    }
    # Join deselects the duplicated right key, mirroring the Alteryx Join config.
    assert "Right_CustomerID" not in ns["df_3_joined"].columns


def test_formula_union_sort_data_values(spark):
    """Sample 03: IF/THEN/ELSE formula, unionByName, sort."""
    wf = AlteryxWorkflowParser(str(SAMPLES_DIR / "03_formula_union_sort.yxmd")).parse()
    result = SelfCorrectingConverter().convert(wf, "s03", spark=spark)
    assert result.status == "PASS"
    assert result.runtime_result and result.runtime_result["ok"]

    final = _exec(result.code, spark)["df_7"]
    got = {(r["Product"], r["Q1Sales"], r["Region"], r["Tier"]) for r in final.collect()}
    assert got == {
        ("Widget", "1000", "East", "Standard"),
        ("Gadget", "2500", "East", "High"),
        ("Widget", "800", "West", "Standard"),
        ("Sprocket", "1900", "West", "High"),
    }


def test_filter_false_port_and_string_functions(spark, tmp_path):
    """True/False filter ports, Trim/UPPERCASE, IIF — hand-checked values."""
    xml = """<?xml version="1.0"?>
<AlteryxDocument yxmdVer="2023.1">
  <Nodes>
    <Node ToolID="1"><GuiSettings Plugin="AlteryxBasePluginsGui.TextInput.TextInput"><Position x="0" y="0"/></GuiSettings>
      <Properties><Configuration>
        <Fields><Field name="Name"/><Field name="Score"/></Fields>
        <Data><r><c>  alice  </c><c>90</c></r><r><c>bob</c><c>40</c></r><r><c>carol</c><c>75</c></r></Data>
      </Configuration><Annotation><DefaultAnnotationText>people</DefaultAnnotationText></Annotation></Properties></Node>
    <Node ToolID="2"><GuiSettings Plugin="AlteryxBasePluginsGui.Filter.Filter"><Position x="1" y="0"/></GuiSettings>
      <Properties><Configuration><Expression>[Score] &gt;= 50</Expression><Mode>Custom</Mode></Configuration>
      <Annotation><DefaultAnnotationText>pass mark</DefaultAnnotationText></Annotation></Properties></Node>
    <Node ToolID="3"><GuiSettings Plugin="AlteryxBasePluginsGui.Formula.Formula"><Position x="2" y="0"/></GuiSettings>
      <Properties><Configuration><FormulaFields>
        <FormulaField field="CleanName" expression="UPPERCASE(Trim([Name]))" type="V_String" size="64"/>
        <FormulaField field="Grade" expression="IIF([Score] &gt; 80, 'A', 'B')" type="V_String" size="4"/>
      </FormulaFields></Configuration><Annotation><DefaultAnnotationText>enrich</DefaultAnnotationText></Annotation></Properties></Node>
    <Node ToolID="4"><GuiSettings Plugin="AlteryxBasePluginsGui.BrowseV2.BrowseV2"><Position x="3" y="0"/></GuiSettings>
      <Properties><Configuration/></Properties></Node>
    <Node ToolID="5"><GuiSettings Plugin="AlteryxBasePluginsGui.BrowseV2.BrowseV2"><Position x="3" y="1"/></GuiSettings>
      <Properties><Configuration/></Properties></Node>
  </Nodes>
  <Connections>
    <Connection><Origin ToolID="1" Connection="Output"/><Destination ToolID="2" Connection="Input"/></Connection>
    <Connection><Origin ToolID="2" Connection="True"/><Destination ToolID="3" Connection="Input"/></Connection>
    <Connection><Origin ToolID="3" Connection="Output"/><Destination ToolID="4" Connection="Input"/></Connection>
    <Connection><Origin ToolID="2" Connection="False"/><Destination ToolID="5" Connection="Input"/></Connection>
  </Connections>
</AlteryxDocument>"""
    yxmd = tmp_path / "inline.yxmd"
    yxmd.write_text(xml)

    wf = AlteryxWorkflowParser(str(yxmd)).parse()
    result = SelfCorrectingConverter().convert(wf, "inline", spark=spark)
    assert result.status == "PASS"

    ns = _exec(result.code, spark)
    assert {(r["CleanName"], r["Grade"]) for r in ns["df_4"].collect()} == {
        ("ALICE", "A"), ("CAROL", "B")}
    assert {r["Name"] for r in ns["df_5"].collect()} == {"bob"}


def test_run_generated_code_reports(spark):
    """The API used by the Skill Mode notebook's section 9."""
    wf = AlteryxWorkflowParser(str(SAMPLES_DIR / "01_simple_filter_select.yxmd")).parse()
    result = SelfCorrectingConverter().convert(wf, "s01")
    outcome = run_generated_code(result.code, spark)
    assert outcome["ok"], outcome["error"]
    assert outcome["dataframes"]["df_4"]["rows"] == 2
    assert outcome["dataframes"]["df_4"]["columns"] == ["OrderID", "CustomerName", "Amount"]
