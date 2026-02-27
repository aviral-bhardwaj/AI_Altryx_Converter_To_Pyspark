"""
Tests for the validation framework.
"""

import json
import pytest
from src.validation import (
    SchemaValidator,
    RowCountValidator,
    DataComparator,
    AggregateValidator,
    ReconciliationReporter,
    WorkflowValidator,
    ValidationStatus,
    ValidationReport,
    ColumnComparison,
    AggregateComparison,
    SampleMismatch,
    ALTERYX_TO_SPARK_TYPE,
)


# ── Schema Validator ──────────────────────────────────────────────

class TestSchemaValidator:
    def setup_method(self):
        self.validator = SchemaValidator()

    def test_exact_match(self):
        status, comps, missing, extra, mismatches = self.validator.validate(
            ["col1", "col2", "col3"],
            ["col1", "col2", "col3"],
        )
        assert status == ValidationStatus.PASS
        assert len(missing) == 0
        assert len(extra) == 0

    def test_case_insensitive_match(self):
        status, comps, missing, extra, mismatches = self.validator.validate(
            ["Col1", "COL2"],
            ["col1", "col2"],
        )
        assert status == ValidationStatus.PASS
        assert len(missing) == 0

    def test_missing_columns(self):
        status, comps, missing, extra, mismatches = self.validator.validate(
            ["col1", "col2", "col3"],
            ["col1"],
        )
        assert status == ValidationStatus.FAIL
        assert "col2" in missing
        assert "col3" in missing

    def test_extra_columns(self):
        status, comps, missing, extra, mismatches = self.validator.validate(
            ["col1"],
            ["col1", "col2"],
        )
        assert status == ValidationStatus.WARNING
        assert "col2" in extra

    def test_type_match_same_type(self):
        status, comps, missing, extra, mismatches = self.validator.validate(
            ["col1"], ["col1"],
            source_types={"col1": "String"},
            target_types={"col1": "StringType"},
        )
        assert status == ValidationStatus.PASS
        assert len(mismatches) == 0

    def test_type_mismatch(self):
        status, comps, missing, extra, mismatches = self.validator.validate(
            ["col1"], ["col1"],
            source_types={"col1": "String"},
            target_types={"col1": "IntegerType"},
        )
        assert status == ValidationStatus.FAIL
        assert len(mismatches) == 1

    def test_compatible_integer_types(self):
        """Int32 and Int64 are in the same compatibility group."""
        status, comps, missing, extra, mismatches = self.validator.validate(
            ["col1"], ["col1"],
            source_types={"col1": "Int32"},
            target_types={"col1": "LongType"},
        )
        assert status == ValidationStatus.PASS

    def test_compatible_float_types(self):
        """Float and Double are in the same compatibility group."""
        status, comps, missing, extra, mismatches = self.validator.validate(
            ["col1"], ["col1"],
            source_types={"col1": "Float"},
            target_types={"col1": "DoubleType"},
        )
        assert status == ValidationStatus.PASS

    def test_empty_schemas(self):
        status, comps, missing, extra, mismatches = self.validator.validate([], [])
        assert status == ValidationStatus.PASS


# ── Row Count Validator ───────────────────────────────────────────

class TestRowCountValidator:
    def setup_method(self):
        self.validator = RowCountValidator()

    def test_exact_match(self):
        status, diff, pct = self.validator.validate(1000, 1000)
        assert status == ValidationStatus.PASS
        assert diff == 0
        assert pct == 0.0

    def test_within_tolerance(self):
        status, diff, pct = self.validator.validate(1000, 1005, tolerance_pct=1.0)
        assert status == ValidationStatus.PASS

    def test_fail_over_tolerance(self):
        status, diff, pct = self.validator.validate(1000, 1200, tolerance_pct=1.0)
        assert status == ValidationStatus.FAIL

    def test_zero_source(self):
        status, diff, pct = self.validator.validate(0, 100)
        assert status == ValidationStatus.FAIL
        assert pct == 100.0

    def test_both_zero(self):
        status, diff, pct = self.validator.validate(0, 0)
        assert status == ValidationStatus.PASS

    def test_target_fewer_rows(self):
        status, diff, pct = self.validator.validate(1000, 800)
        assert diff == -200

    def test_warning_near_tolerance(self):
        """Slightly over tolerance gets WARNING."""
        status, diff, pct = self.validator.validate(1000, 1015, tolerance_pct=1.0)
        assert status == ValidationStatus.WARNING


# ── Data Comparator ───────────────────────────────────────────────

class TestDataComparator:
    def setup_method(self):
        self.comparator = DataComparator()

    def test_exact_match(self):
        rows = [{"a": "1", "b": "2"}, {"a": "3", "b": "4"}]
        status, pct, mismatches = self.comparator.compare(rows, rows)
        assert status == ValidationStatus.PASS
        assert pct == 100.0
        assert len(mismatches) == 0

    def test_value_mismatch(self):
        src = [{"a": "1", "b": "2"}]
        tgt = [{"a": "1", "b": "999"}]
        status, pct, mismatches = self.comparator.compare(src, tgt)
        assert pct < 100.0
        assert len(mismatches) > 0
        assert mismatches[0].column_name == "b"

    def test_float_tolerance(self):
        src = [{"val": 1.0001}]
        tgt = [{"val": 1.0002}]
        status, pct, mismatches = self.comparator.compare(src, tgt, float_tolerance=0.001)
        assert pct == 100.0

    def test_float_tolerance_exceeded(self):
        src = [{"val": 1.0}]
        tgt = [{"val": 2.0}]
        status, pct, mismatches = self.comparator.compare(src, tgt, float_tolerance=0.001)
        assert pct < 100.0

    def test_null_comparison_both_null(self):
        src = [{"a": None}]
        tgt = [{"a": None}]
        status, pct, mismatches = self.comparator.compare(src, tgt)
        assert pct == 100.0

    def test_null_comparison_one_null(self):
        src = [{"a": None}]
        tgt = [{"a": "value"}]
        status, pct, mismatches = self.comparator.compare(src, tgt)
        assert pct < 100.0

    def test_case_insensitive_string(self):
        src = [{"name": "ALICE"}]
        tgt = [{"name": "alice"}]
        status, pct, mismatches = self.comparator.compare(src, tgt)
        assert pct == 100.0

    def test_whitespace_trimmed(self):
        src = [{"name": "  alice  "}]
        tgt = [{"name": "alice"}]
        status, pct, mismatches = self.comparator.compare(src, tgt)
        assert pct == 100.0

    def test_both_empty(self):
        status, pct, mismatches = self.comparator.compare([], [])
        assert status == ValidationStatus.PASS

    def test_one_empty(self):
        status, pct, mismatches = self.comparator.compare([{"a": 1}], [])
        assert status == ValidationStatus.FAIL

    def test_key_column_sort(self):
        src = [{"id": "2", "val": "b"}, {"id": "1", "val": "a"}]
        tgt = [{"id": "1", "val": "a"}, {"id": "2", "val": "b"}]
        status, pct, mismatches = self.comparator.compare(src, tgt, key_columns=["id"])
        assert pct == 100.0

    def test_max_mismatches_limit(self):
        src = [{"a": str(i)} for i in range(100)]
        tgt = [{"a": str(i + 1000)} for i in range(100)]
        status, pct, mismatches = self.comparator.compare(src, tgt, max_mismatches=10)
        assert len(mismatches) == 10


# ── Aggregate Validator ───────────────────────────────────────────

class TestAggregateValidator:
    def setup_method(self):
        self.validator = AggregateValidator()

    def test_exact_match(self):
        source = {"col1": {"sum": 100, "count": 10}}
        target = {"col1": {"sum": 100, "count": 10}}
        status, comps = self.validator.validate(source, target)
        assert status == ValidationStatus.PASS
        assert all(c.match for c in comps)

    def test_within_tolerance(self):
        source = {"col1": {"sum": 100.0}}
        target = {"col1": {"sum": 100.5}}
        status, comps = self.validator.validate(source, target, tolerance=0.01)
        assert status == ValidationStatus.PASS

    def test_outside_tolerance(self):
        source = {"col1": {"sum": 100.0}}
        target = {"col1": {"sum": 200.0}}
        status, comps = self.validator.validate(source, target, tolerance=0.01)
        assert status == ValidationStatus.FAIL

    def test_missing_column_in_target(self):
        source = {"col1": {"sum": 100}}
        target = {"col2": {"sum": 200}}
        status, comps = self.validator.validate(source, target)
        # col1 only in source, col2 only in target
        assert any(not c.match for c in comps)

    def test_zero_source_value(self):
        source = {"col1": {"sum": 0}}
        target = {"col1": {"sum": 0}}
        status, comps = self.validator.validate(source, target)
        assert status == ValidationStatus.PASS

    def test_string_comparison(self):
        source = {"col1": {"mode": "Active"}}
        target = {"col1": {"mode": "Active"}}
        status, comps = self.validator.validate(source, target)
        assert status == ValidationStatus.PASS


# ── Reconciliation Reporter ───────────────────────────────────────

class TestReconciliationReporter:
    def setup_method(self):
        self.reporter = ReconciliationReporter()

    def test_build_report_all_pass(self):
        schema_result = (ValidationStatus.PASS, [], [], [], [])
        row_count_result = (ValidationStatus.PASS, 0, 0.0)
        report = self.reporter.build_report(
            workflow_name="test",
            schema_result=schema_result,
            row_count_result=row_count_result,
            source_row_count=100,
            target_row_count=100,
        )
        assert report.status == ValidationStatus.PASS
        assert report.workflow_name == "test"

    def test_build_report_with_failures(self):
        schema_result = (ValidationStatus.FAIL, [], ["missing_col"], [], [])
        report = self.reporter.build_report(
            workflow_name="test",
            schema_result=schema_result,
        )
        assert report.status == ValidationStatus.FAIL
        assert "missing_col" in report.missing_columns

    def test_to_json(self):
        report = ValidationReport(
            status=ValidationStatus.PASS,
            workflow_name="test",
            source_row_count=100,
            target_row_count=100,
        )
        json_str = self.reporter.to_json(report)
        data = json.loads(json_str)
        assert data["status"] == "PASS"
        assert data["workflow_name"] == "test"

    def test_to_html(self):
        report = ValidationReport(
            status=ValidationStatus.PASS,
            workflow_name="test",
        )
        html = self.reporter.to_html(report)
        assert "<!DOCTYPE html>" in html
        assert "test" in html
        assert "PASS" in html

    def test_to_html_with_mismatches(self):
        report = ValidationReport(
            status=ValidationStatus.FAIL,
            workflow_name="test",
            missing_columns=["col_x"],
            type_mismatches=[ColumnComparison("col_y", "String", "IntegerType", False)],
            sample_mismatches=[SampleMismatch(0, "col_z", "a", "b")],
            aggregate_comparisons=[AggregateComparison("col_w", "sum", 100, 200, False, 100, 100.0)],
        )
        html = self.reporter.to_html(report)
        assert "col_x" in html
        assert "col_y" in html
        assert "col_z" in html
        assert "col_w" in html

    def test_to_dict(self):
        report = ValidationReport(status=ValidationStatus.PASS, workflow_name="test")
        d = self.reporter.to_dict(report)
        assert isinstance(d, dict)
        assert d["status"] == "PASS"

    def test_recommendations_missing_columns(self):
        report = ValidationReport(missing_columns=["col_a"])
        recs = self.reporter._generate_recommendations(report)
        assert any("Missing" in r for r in recs)

    def test_recommendations_extra_columns(self):
        report = ValidationReport(extra_columns=["col_b"])
        recs = self.reporter._generate_recommendations(report)
        assert any("Extra" in r for r in recs)

    def test_recommendations_type_mismatches(self):
        report = ValidationReport(
            type_mismatches=[ColumnComparison("col_c", "String", "Int", False)]
        )
        recs = self.reporter._generate_recommendations(report)
        assert any("mismatch" in r.lower() for r in recs)

    def test_recommendations_all_pass(self):
        report = ValidationReport()
        recs = self.reporter._generate_recommendations(report)
        assert any("passed" in r.lower() for r in recs)


# ── WorkflowValidator (Orchestrator) ──────────────────────────────

class TestWorkflowValidator:
    def setup_method(self):
        self.validator = WorkflowValidator(workflow_name="test_workflow")

    def test_full_validation_all_pass(self):
        report = self.validator.validate(
            source_columns=["id", "name"],
            target_columns=["id", "name"],
            source_row_count=1000,
            target_row_count=1000,
            source_rows=[{"id": "1", "name": "Alice"}],
            target_rows=[{"id": "1", "name": "Alice"}],
            source_aggregates={"id": {"count": 1000}},
            target_aggregates={"id": {"count": 1000}},
        )
        assert report.status == ValidationStatus.PASS

    def test_schema_only_validation(self):
        report = self.validator.validate(
            source_columns=["id", "name", "extra"],
            target_columns=["id", "name"],
        )
        assert report.schema_status == ValidationStatus.FAIL

    def test_row_count_only_validation(self):
        report = self.validator.validate(
            source_row_count=1000,
            target_row_count=500,
        )
        assert report.row_count_status == ValidationStatus.FAIL

    def test_no_data_provided(self):
        """When no data is provided, all sub-validators are skipped."""
        report = self.validator.validate()
        assert report.status == ValidationStatus.PASS

    def test_to_json(self):
        report = self.validator.validate(
            source_columns=["id"],
            target_columns=["id"],
        )
        json_str = self.validator.to_json(report)
        data = json.loads(json_str)
        assert "schema" in data

    def test_to_html(self):
        report = self.validator.validate(
            source_columns=["id"],
            target_columns=["id"],
        )
        html = self.validator.to_html(report)
        assert "<html>" in html

    def test_to_dict(self):
        report = self.validator.validate(
            source_row_count=100,
            target_row_count=100,
        )
        d = self.validator.to_dict(report)
        assert isinstance(d, dict)

    def test_combined_warnings(self):
        report = self.validator.validate(
            source_columns=["id"],
            target_columns=["id", "extra_col"],
            source_row_count=1000,
            target_row_count=1000,
        )
        # Extra column -> WARNING for schema, row count is PASS
        assert report.schema_status == ValidationStatus.WARNING
        assert report.row_count_status == ValidationStatus.PASS
        assert report.status == ValidationStatus.WARNING


# ── Type Mapping ──────────────────────────────────────────────────

class TestTypeMapping:
    def test_all_alteryx_types_mapped(self):
        expected_types = [
            "Byte", "Int16", "Int32", "Int64",
            "Float", "Double", "FixedDecimal",
            "String", "V_String", "V_WString", "WString",
            "Bool", "Date", "DateTime",
        ]
        for t in expected_types:
            assert t in ALTERYX_TO_SPARK_TYPE, f"Missing mapping for {t}"

    def test_integer_types_map_correctly(self):
        assert ALTERYX_TO_SPARK_TYPE["Int32"] == "IntegerType"
        assert ALTERYX_TO_SPARK_TYPE["Int64"] == "LongType"

    def test_string_types_map_correctly(self):
        assert ALTERYX_TO_SPARK_TYPE["String"] == "StringType"
        assert ALTERYX_TO_SPARK_TYPE["V_String"] == "StringType"
        assert ALTERYX_TO_SPARK_TYPE["V_WString"] == "StringType"

    def test_date_types_map_correctly(self):
        assert ALTERYX_TO_SPARK_TYPE["Date"] == "DateType"
        assert ALTERYX_TO_SPARK_TYPE["DateTime"] == "TimestampType"
