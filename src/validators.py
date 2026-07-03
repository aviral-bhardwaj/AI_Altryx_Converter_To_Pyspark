"""
Validation Framework
=====================
Compares the output of original Alteryx workflow with generated PySpark pipeline
to ensure correctness.

Provides:
1. Schema validation (column names, types)
2. Row count validation
3. Data comparison (sample-based)
4. Aggregate validation (sum, count, min, max, avg per column)
5. Reconciliation report (JSON, HTML, DataFrame)
"""

import json
import datetime
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


# ═══════════════════════════════════════════════════════════════════
# Type Mapping: Alteryx -> Spark
# ═══════════════════════════════════════════════════════════════════

ALTERYX_TO_SPARK_TYPE = {
    # Integer types
    "Byte": "ByteType",
    "Int16": "ShortType",
    "Int32": "IntegerType",
    "Int64": "LongType",
    # Float types
    "Float": "FloatType",
    "Double": "DoubleType",
    "FixedDecimal": "DecimalType",
    # String types
    "String": "StringType",
    "V_String": "StringType",
    "V_WString": "StringType",
    "WString": "StringType",
    # Boolean
    "Bool": "BooleanType",
    # Date/Time
    "Date": "DateType",
    "DateTime": "TimestampType",
}

# Type compatibility groups (types that are compatible for comparison)
TYPE_COMPAT_GROUPS = {
    "integer": {"ByteType", "ShortType", "IntegerType", "LongType"},
    "float": {"FloatType", "DoubleType", "DecimalType"},
    "string": {"StringType"},
    "boolean": {"BooleanType"},
    "date": {"DateType"},
    "timestamp": {"TimestampType"},
}


class ValidationStatus(Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    WARNING = "WARNING"


@dataclass
class ColumnComparison:
    """Result of comparing a single column between source and target."""
    column_name: str
    source_type: str = ""
    target_type: str = ""
    type_match: bool = True
    present_in_source: bool = True
    present_in_target: bool = True
    notes: str = ""


@dataclass
class AggregateComparison:
    """Result of comparing aggregates for a single column."""
    column_name: str
    metric: str  # sum, count, min, max, avg, distinct_count, null_count
    source_value: Optional[float] = None
    target_value: Optional[float] = None
    match: bool = True
    difference: Optional[float] = None
    pct_difference: Optional[float] = None


@dataclass
class SampleMismatch:
    """A single mismatched row/cell in sample comparison."""
    row_index: int
    column_name: str
    source_value: str
    target_value: str


@dataclass
class ValidationReport:
    """Complete validation report."""
    status: ValidationStatus = ValidationStatus.PASS
    timestamp: str = ""
    workflow_name: str = ""

    # Schema validation
    schema_status: ValidationStatus = ValidationStatus.PASS
    column_comparisons: list = field(default_factory=list)
    missing_columns: list = field(default_factory=list)
    extra_columns: list = field(default_factory=list)
    type_mismatches: list = field(default_factory=list)

    # Row count validation
    row_count_status: ValidationStatus = ValidationStatus.PASS
    source_row_count: int = 0
    target_row_count: int = 0
    row_count_difference: int = 0
    row_count_pct_difference: float = 0.0

    # Aggregate validation
    aggregate_status: ValidationStatus = ValidationStatus.PASS
    aggregate_comparisons: list = field(default_factory=list)

    # Sample data comparison
    sample_status: ValidationStatus = ValidationStatus.PASS
    sample_size: int = 0
    matching_rows_pct: float = 100.0
    sample_mismatches: list = field(default_factory=list)

    # Recommendations
    recommendations: list = field(default_factory=list)


# ═══════════════════════════════════════════════════════════════════
# Schema Validator
# ═══════════════════════════════════════════════════════════════════

class SchemaValidator:
    """Compare column names and data types between source and target schemas."""

    def validate(
        self,
        source_columns: list,
        target_columns: list,
        source_types: Optional[dict] = None,
        target_types: Optional[dict] = None,
    ) -> tuple:
        """
        Compare schemas.

        Args:
            source_columns: List of column names from source.
            target_columns: List of column names from target.
            source_types: Optional dict of {column_name: type_string} for source.
            target_types: Optional dict of {column_name: type_string} for target.

        Returns:
            (status, column_comparisons, missing, extra, type_mismatches)
        """
        source_types = source_types or {}
        target_types = target_types or {}

        # Case-insensitive comparison
        source_lower = {c.lower(): c for c in source_columns}
        target_lower = {c.lower(): c for c in target_columns}

        comparisons = []
        missing = []
        extra = []
        type_mismatches = []

        # Check source columns against target
        for col_lower, col_name in source_lower.items():
            comp = ColumnComparison(column_name=col_name)

            if col_lower not in target_lower:
                comp.present_in_target = False
                comp.notes = "Missing in target"
                missing.append(col_name)
            else:
                target_col = target_lower[col_lower]
                src_type = source_types.get(col_name, "")
                tgt_type = target_types.get(target_col, "")

                if src_type and tgt_type:
                    comp.source_type = src_type
                    comp.target_type = tgt_type
                    comp.type_match = self._types_compatible(src_type, tgt_type)
                    if not comp.type_match:
                        comp.notes = f"Type mismatch: {src_type} vs {tgt_type}"
                        type_mismatches.append(comp)

            comparisons.append(comp)

        # Check for extra columns in target
        for col_lower, col_name in target_lower.items():
            if col_lower not in source_lower:
                comp = ColumnComparison(
                    column_name=col_name,
                    present_in_source=False,
                    notes="Extra in target",
                )
                comparisons.append(comp)
                extra.append(col_name)

        # Determine status
        if missing or type_mismatches:
            status = ValidationStatus.FAIL
        elif extra:
            status = ValidationStatus.WARNING
        else:
            status = ValidationStatus.PASS

        return status, comparisons, missing, extra, type_mismatches

    def _types_compatible(self, source_type: str, target_type: str) -> bool:
        """Check if two types are compatible (within the same group)."""
        # Normalize Alteryx types to Spark types
        spark_source = ALTERYX_TO_SPARK_TYPE.get(source_type, source_type)
        spark_target = ALTERYX_TO_SPARK_TYPE.get(target_type, target_type)

        if spark_source == spark_target:
            return True

        # Check compatibility groups
        for group_types in TYPE_COMPAT_GROUPS.values():
            if spark_source in group_types and spark_target in group_types:
                return True

        return False


# ═══════════════════════════════════════════════════════════════════
# Row Count Validator
# ═══════════════════════════════════════════════════════════════════

class RowCountValidator:
    """Compare row counts between source and target."""

    def validate(
        self,
        source_count: int,
        target_count: int,
        tolerance_pct: float = 0.0,
    ) -> tuple:
        """
        Compare row counts.

        Args:
            source_count: Number of rows in source.
            target_count: Number of rows in target.
            tolerance_pct: Acceptable percentage difference (0.0 = exact match).

        Returns:
            (status, difference, pct_difference)
        """
        difference = target_count - source_count
        if source_count > 0:
            pct_difference = abs(difference) / source_count * 100
        else:
            pct_difference = 100.0 if target_count > 0 else 0.0

        if pct_difference <= tolerance_pct:
            status = ValidationStatus.PASS
        elif pct_difference <= tolerance_pct + 1.0:
            status = ValidationStatus.WARNING
        else:
            status = ValidationStatus.FAIL

        return status, difference, pct_difference


# ═══════════════════════════════════════════════════════════════════
# Data Comparator (Sample-Based)
# ═══════════════════════════════════════════════════════════════════

class DataComparator:
    """Compare actual data values between source and target (sample-based)."""

    def compare(
        self,
        source_rows: list,
        target_rows: list,
        key_columns: Optional[list] = None,
        float_tolerance: float = 0.0001,
        max_mismatches: int = 50,
    ) -> tuple:
        """
        Compare rows of data.

        Args:
            source_rows: List of dicts [{col: value, ...}, ...]
            target_rows: List of dicts [{col: value, ...}, ...]
            key_columns: Columns to sort by before comparison.
            float_tolerance: Tolerance for floating point comparison.
            max_mismatches: Maximum mismatches to collect.

        Returns:
            (status, matching_pct, mismatches)
        """
        if not source_rows or not target_rows:
            if not source_rows and not target_rows:
                return ValidationStatus.PASS, 100.0, []
            return ValidationStatus.FAIL, 0.0, []

        # Sort both by key columns if provided
        if key_columns:
            source_rows = sorted(source_rows, key=lambda r: tuple(str(r.get(k, "")) for k in key_columns))
            target_rows = sorted(target_rows, key=lambda r: tuple(str(r.get(k, "")) for k in key_columns))

        mismatches = []
        total_cells = 0
        matching_cells = 0

        compare_length = min(len(source_rows), len(target_rows))

        for i in range(compare_length):
            src_row = source_rows[i]
            tgt_row = target_rows[i]

            # Compare all columns present in source
            for col in src_row:
                total_cells += 1
                src_val = src_row.get(col)
                tgt_val = tgt_row.get(col)

                if self._values_match(src_val, tgt_val, float_tolerance):
                    matching_cells += 1
                elif len(mismatches) < max_mismatches:
                    mismatches.append(SampleMismatch(
                        row_index=i,
                        column_name=col,
                        source_value=str(src_val),
                        target_value=str(tgt_val),
                    ))

        matching_pct = (matching_cells / total_cells * 100) if total_cells > 0 else 100.0

        if matching_pct >= 99.9:
            status = ValidationStatus.PASS
        elif matching_pct >= 95.0:
            status = ValidationStatus.WARNING
        else:
            status = ValidationStatus.FAIL

        return status, matching_pct, mismatches

    def _values_match(self, source, target, tolerance: float) -> bool:
        """Compare two values with type-aware logic."""
        # Both None/NULL
        if source is None and target is None:
            return True
        if source is None or target is None:
            return False

        # Float comparison
        try:
            src_float = float(source)
            tgt_float = float(target)
            return abs(src_float - tgt_float) <= tolerance
        except (ValueError, TypeError):
            pass

        # String comparison (case-insensitive, whitespace-stripped)
        return str(source).strip().lower() == str(target).strip().lower()


# ═══════════════════════════════════════════════════════════════════
# Aggregate Validator
# ═══════════════════════════════════════════════════════════════════

class AggregateValidator:
    """Compare aggregate statistics between source and target."""

    def validate(
        self,
        source_aggregates: dict,
        target_aggregates: dict,
        tolerance: float = 0.01,
    ) -> tuple:
        """
        Compare aggregate values.

        Args:
            source_aggregates: {column: {metric: value, ...}, ...}
            target_aggregates: {column: {metric: value, ...}, ...}
            tolerance: Relative tolerance for numeric comparison.

        Returns:
            (status, comparisons)
        """
        comparisons = []
        has_failure = False
        has_warning = False

        all_columns = set(source_aggregates.keys()) | set(target_aggregates.keys())

        for col in sorted(all_columns):
            src_metrics = source_aggregates.get(col, {})
            tgt_metrics = target_aggregates.get(col, {})

            all_metrics = set(src_metrics.keys()) | set(tgt_metrics.keys())

            for metric in sorted(all_metrics):
                src_val = src_metrics.get(metric)
                tgt_val = tgt_metrics.get(metric)

                comp = AggregateComparison(
                    column_name=col,
                    metric=metric,
                    source_value=src_val,
                    target_value=tgt_val,
                )

                if src_val is not None and tgt_val is not None:
                    try:
                        src_f = float(src_val)
                        tgt_f = float(tgt_val)
                        comp.difference = tgt_f - src_f
                        if src_f != 0:
                            comp.pct_difference = abs(comp.difference) / abs(src_f) * 100
                        else:
                            comp.pct_difference = 0.0 if tgt_f == 0 else 100.0

                        if comp.pct_difference <= tolerance * 100:
                            comp.match = True
                        else:
                            comp.match = False
                            has_failure = True
                    except (ValueError, TypeError):
                        # String comparison
                        comp.match = str(src_val) == str(tgt_val)
                        if not comp.match:
                            has_warning = True
                elif src_val is None and tgt_val is None:
                    comp.match = True
                else:
                    comp.match = False
                    has_warning = True

                comparisons.append(comp)

        if has_failure:
            status = ValidationStatus.FAIL
        elif has_warning:
            status = ValidationStatus.WARNING
        else:
            status = ValidationStatus.PASS

        return status, comparisons


# ═══════════════════════════════════════════════════════════════════
# Reconciliation Report Generator
# ═══════════════════════════════════════════════════════════════════

class ReconciliationReporter:
    """Generate reconciliation reports in multiple formats."""

    def build_report(
        self,
        workflow_name: str,
        schema_result: Optional[tuple] = None,
        row_count_result: Optional[tuple] = None,
        sample_result: Optional[tuple] = None,
        aggregate_result: Optional[tuple] = None,
        source_row_count: int = 0,
        target_row_count: int = 0,
        sample_size: int = 0,
    ) -> ValidationReport:
        """Build a complete ValidationReport from individual validation results."""
        report = ValidationReport(
            timestamp=datetime.datetime.now().isoformat(),
            workflow_name=workflow_name,
        )

        # Schema
        if schema_result:
            status, comparisons, missing, extra, type_mismatches = schema_result
            report.schema_status = status
            report.column_comparisons = comparisons
            report.missing_columns = missing
            report.extra_columns = extra
            report.type_mismatches = type_mismatches

        # Row count
        if row_count_result:
            status, difference, pct_difference = row_count_result
            report.row_count_status = status
            report.source_row_count = source_row_count
            report.target_row_count = target_row_count
            report.row_count_difference = difference
            report.row_count_pct_difference = pct_difference

        # Sample comparison
        if sample_result:
            status, matching_pct, mismatches = sample_result
            report.sample_status = status
            report.sample_size = sample_size
            report.matching_rows_pct = matching_pct
            report.sample_mismatches = mismatches

        # Aggregates
        if aggregate_result:
            status, comparisons = aggregate_result
            report.aggregate_status = status
            report.aggregate_comparisons = comparisons

        # Overall status
        statuses = [
            report.schema_status,
            report.row_count_status,
            report.sample_status,
            report.aggregate_status,
        ]
        if ValidationStatus.FAIL in statuses:
            report.status = ValidationStatus.FAIL
        elif ValidationStatus.WARNING in statuses:
            report.status = ValidationStatus.WARNING
        else:
            report.status = ValidationStatus.PASS

        # Generate recommendations
        report.recommendations = self._generate_recommendations(report)

        return report

    def _generate_recommendations(self, report: ValidationReport) -> list:
        """Generate actionable recommendations based on validation results."""
        recs = []

        if report.missing_columns:
            recs.append(
                f"Missing columns in target: {', '.join(report.missing_columns)}. "
                "Check if column names are mapped correctly in the PySpark code."
            )

        if report.extra_columns:
            recs.append(
                f"Extra columns in target: {', '.join(report.extra_columns)}. "
                "Verify these are intentional additions."
            )

        if report.type_mismatches:
            cols = [tc.column_name for tc in report.type_mismatches]
            recs.append(
                f"Type mismatches in columns: {', '.join(cols)}. "
                "Add .cast() operations to match expected types."
            )

        if report.row_count_status == ValidationStatus.FAIL:
            recs.append(
                f"Row count mismatch: source={report.source_row_count:,}, "
                f"target={report.target_row_count:,} "
                f"(diff={report.row_count_difference:+,}, {report.row_count_pct_difference:.1f}%). "
                "Check filter conditions and join types."
            )

        if report.sample_status == ValidationStatus.FAIL:
            recs.append(
                f"Sample data match rate: {report.matching_rows_pct:.1f}%. "
                "Review mismatched columns for expression conversion errors."
            )

        failed_aggs = [
            c for c in report.aggregate_comparisons if not c.match
        ]
        if failed_aggs:
            cols = set(c.column_name for c in failed_aggs)
            recs.append(
                f"Aggregate mismatches in columns: {', '.join(cols)}. "
                "Check formula conversions and null handling."
            )

        if not recs:
            recs.append("All validations passed. No issues found.")

        return recs

    def to_json(self, report: ValidationReport) -> str:
        """Export report as JSON string."""
        data = {
            "status": report.status.value,
            "timestamp": report.timestamp,
            "workflow_name": report.workflow_name,
            "schema": {
                "status": report.schema_status.value,
                "missing_columns": report.missing_columns,
                "extra_columns": report.extra_columns,
                "type_mismatches": [
                    {
                        "column": tc.column_name,
                        "source_type": tc.source_type,
                        "target_type": tc.target_type,
                    }
                    for tc in report.type_mismatches
                ],
            },
            "row_count": {
                "status": report.row_count_status.value,
                "source": report.source_row_count,
                "target": report.target_row_count,
                "difference": report.row_count_difference,
                "pct_difference": report.row_count_pct_difference,
            },
            "sample_comparison": {
                "status": report.sample_status.value,
                "sample_size": report.sample_size,
                "matching_pct": report.matching_rows_pct,
                "mismatches": [
                    {
                        "row": m.row_index,
                        "column": m.column_name,
                        "source": m.source_value,
                        "target": m.target_value,
                    }
                    for m in report.sample_mismatches[:20]
                ],
            },
            "aggregates": {
                "status": report.aggregate_status.value,
                "comparisons": [
                    {
                        "column": c.column_name,
                        "metric": c.metric,
                        "source": c.source_value,
                        "target": c.target_value,
                        "match": c.match,
                        "difference": c.difference,
                        "pct_difference": c.pct_difference,
                    }
                    for c in report.aggregate_comparisons
                ],
            },
            "recommendations": report.recommendations,
        }
        return json.dumps(data, indent=2, default=str)

    def to_html(self, report: ValidationReport) -> str:
        """Export report as HTML string for human review."""
        status_colors = {
            ValidationStatus.PASS: "#28a745",
            ValidationStatus.FAIL: "#dc3545",
            ValidationStatus.WARNING: "#ffc107",
        }

        def badge(status: ValidationStatus) -> str:
            color = status_colors[status]
            return (
                f'<span style="background:{color};color:white;'
                f'padding:2px 8px;border-radius:4px;font-weight:bold">'
                f'{status.value}</span>'
            )

        from html import escape as _esc

        html = []
        html.append("<!DOCTYPE html>")
        html.append('<html><head><meta charset="utf-8">')
        html.append(f"<title>Validation Report - {_esc(str(report.workflow_name))}</title>")
        html.append("<style>")
        html.append("body { font-family: -apple-system, sans-serif; max-width: 900px; margin: 40px auto; padding: 0 20px; }")
        html.append("h1, h2, h3 { color: #333; }")
        html.append("table { border-collapse: collapse; width: 100%; margin: 16px 0; }")
        html.append("th, td { border: 1px solid #ddd; padding: 8px 12px; text-align: left; }")
        html.append("th { background: #f5f5f5; }")
        html.append("tr:nth-child(even) { background: #fafafa; }")
        html.append(".pass { color: #28a745; } .fail { color: #dc3545; } .warning { color: #ffc107; }")
        html.append("</style></head><body>")

        html.append(f"<h1>Validation Report: {_esc(str(report.workflow_name))}</h1>")
        html.append(f"<p>Generated: {_esc(str(report.timestamp))}</p>")
        html.append(f"<h2>Overall Status: {badge(report.status)}</h2>")

        # Schema section
        html.append(f"<h3>Schema Validation {badge(report.schema_status)}</h3>")
        if report.missing_columns:
            html.append(f"<p><strong>Missing columns:</strong> {_esc(', '.join(report.missing_columns))}</p>")
        if report.extra_columns:
            html.append(f"<p><strong>Extra columns:</strong> {_esc(', '.join(report.extra_columns))}</p>")
        if report.type_mismatches:
            html.append("<table><tr><th>Column</th><th>Source Type</th><th>Target Type</th></tr>")
            for tc in report.type_mismatches:
                html.append(f"<tr><td>{_esc(str(tc.column_name))}</td><td>{_esc(str(tc.source_type))}</td><td>{_esc(str(tc.target_type))}</td></tr>")
            html.append("</table>")

        # Row count section
        html.append(f"<h3>Row Count {badge(report.row_count_status)}</h3>")
        html.append(f"<p>Source: {report.source_row_count:,} | Target: {report.target_row_count:,} | ")
        html.append(f"Difference: {report.row_count_difference:+,} ({report.row_count_pct_difference:.1f}%)</p>")

        # Sample comparison section
        html.append(f"<h3>Sample Comparison {badge(report.sample_status)}</h3>")
        html.append(f"<p>Sample size: {report.sample_size:,} | Match rate: {report.matching_rows_pct:.1f}%</p>")
        if report.sample_mismatches:
            html.append("<table><tr><th>Row</th><th>Column</th><th>Source</th><th>Target</th></tr>")
            for m in report.sample_mismatches[:20]:
                html.append(f"<tr><td>{m.row_index}</td><td>{_esc(str(m.column_name))}</td>"
                            f"<td>{_esc(str(m.source_value))}</td><td>{_esc(str(m.target_value))}</td></tr>")
            html.append("</table>")

        # Aggregate section
        html.append(f"<h3>Aggregate Validation {badge(report.aggregate_status)}</h3>")
        if report.aggregate_comparisons:
            html.append("<table><tr><th>Column</th><th>Metric</th><th>Source</th><th>Target</th><th>Match</th></tr>")
            for c in report.aggregate_comparisons:
                match_cls = "pass" if c.match else "fail"
                html.append(
                    f'<tr><td>{_esc(str(c.column_name))}</td><td>{_esc(str(c.metric))}</td>'
                    f'<td>{_esc(str(c.source_value))}</td><td>{_esc(str(c.target_value))}</td>'
                    f'<td class="{match_cls}">{"PASS" if c.match else "FAIL"}</td></tr>'
                )
            html.append("</table>")

        # Recommendations
        html.append("<h3>Recommendations</h3>")
        html.append("<ul>")
        for rec in report.recommendations:
            html.append(f"<li>{_esc(str(rec))}</li>")
        html.append("</ul>")

        html.append("</body></html>")
        return "\n".join(html)

    def to_dict(self, report: ValidationReport) -> dict:
        """Export report as a dict (for DataFrame creation in Databricks)."""
        return json.loads(self.to_json(report))


# ═══════════════════════════════════════════════════════════════════
# Main Validation Orchestrator
# ═══════════════════════════════════════════════════════════════════

class WorkflowValidator:
    """
    Orchestrates all validation checks and produces a reconciliation report.

    Usage:
        validator = WorkflowValidator(workflow_name="my_workflow")
        report = validator.validate(
            source_columns=["col1", "col2"],
            target_columns=["col1", "col2"],
            source_row_count=1000,
            target_row_count=1000,
        )
        print(validator.to_json(report))
        print(validator.to_html(report))
    """

    def __init__(self, workflow_name: str = "workflow"):
        self.workflow_name = workflow_name
        self._schema_validator = SchemaValidator()
        self._row_count_validator = RowCountValidator()
        self._data_comparator = DataComparator()
        self._aggregate_validator = AggregateValidator()
        self._reporter = ReconciliationReporter()

    def validate(
        self,
        source_columns: Optional[list] = None,
        target_columns: Optional[list] = None,
        source_types: Optional[dict] = None,
        target_types: Optional[dict] = None,
        source_row_count: Optional[int] = None,
        target_row_count: Optional[int] = None,
        row_count_tolerance_pct: float = 0.0,
        source_rows: Optional[list] = None,
        target_rows: Optional[list] = None,
        key_columns: Optional[list] = None,
        float_tolerance: float = 0.0001,
        source_aggregates: Optional[dict] = None,
        target_aggregates: Optional[dict] = None,
        aggregate_tolerance: float = 0.01,
    ) -> ValidationReport:
        """
        Run all applicable validations and produce a report.

        All parameters are optional - only validations with sufficient data will run.
        """
        schema_result = None
        row_count_result = None
        sample_result = None
        aggregate_result = None

        # Schema validation
        if source_columns is not None and target_columns is not None:
            schema_result = self._schema_validator.validate(
                source_columns, target_columns, source_types, target_types
            )

        # Row count validation
        if source_row_count is not None and target_row_count is not None:
            row_count_result = self._row_count_validator.validate(
                source_row_count, target_row_count, row_count_tolerance_pct
            )

        # Sample data comparison
        sample_size = 0
        if source_rows is not None and target_rows is not None:
            sample_size = min(len(source_rows), len(target_rows))
            sample_result = self._data_comparator.compare(
                source_rows, target_rows, key_columns, float_tolerance
            )

        # Aggregate validation
        if source_aggregates is not None and target_aggregates is not None:
            aggregate_result = self._aggregate_validator.validate(
                source_aggregates, target_aggregates, aggregate_tolerance
            )

        return self._reporter.build_report(
            workflow_name=self.workflow_name,
            schema_result=schema_result,
            row_count_result=row_count_result,
            sample_result=sample_result,
            aggregate_result=aggregate_result,
            source_row_count=source_row_count or 0,
            target_row_count=target_row_count or 0,
            sample_size=sample_size,
        )

    def to_json(self, report: ValidationReport) -> str:
        """Export report as JSON."""
        return self._reporter.to_json(report)

    def to_html(self, report: ValidationReport) -> str:
        """Export report as HTML."""
        return self._reporter.to_html(report)

    def to_dict(self, report: ValidationReport) -> dict:
        """Export report as dict."""
        return self._reporter.to_dict(report)


# ═══════════════════════════════════════════════════════════════════
# Structural Validation (AST-based)
# ═══════════════════════════════════════════════════════════════════
#
# Used by the self-correction loop: parses the *generated* PySpark code
# with Python's `ast` module and compares the logical operations found
# against the operations expected from the original .yxmd DAG (tool
# sequence, join types/keys, filter conditions, aggregations, column
# mappings). Also lints for unresolved DataFrame references, syntax
# errors, TODO placeholders, and missed Databricks optimizations.

import ast
import re as _re

# Alteryx tool type -> DataFrame method names, any of which satisfies the tool.
EXPECTED_OPERATIONS = {
    "Filter": {"filter", "where"},
    "LockInFilter": {"filter", "where"},
    "Join": {"join"},
    "LockInJoin": {"join"},
    "Formula": {"withColumn"},
    "LockInFormula": {"withColumn"},
    "Summarize": {"groupBy", "agg"},
    "Sort": {"orderBy", "sort"},
    "Union": {"unionByName", "union"},
    "Unique": {"dropDuplicates", "distinct"},
    "Sample": {"limit", "sample"},
    "CrossTab": {"pivot"},
    "Transpose": {"select"},  # stack() expression inside a select
    "RecordID": {"withColumn"},
    "MultiRowFormula": {"withColumn"},
    "AppendFields": {"crossJoin"},
    "InputData": {"table", "csv", "parquet", "load", "createDataFrame"},
    "LockInStreamIn": {"table", "csv", "parquet", "load", "createDataFrame"},
    "DynamicInput": {"table", "csv", "parquet", "load", "createDataFrame"},
    "TextInput": {"createDataFrame"},
    "OutputData": {"createOrReplaceTempView", "saveAsTable", "write"},
    "LockInStreamOut": {"createOrReplaceTempView", "saveAsTable", "write"},
}

# Tool types with no runtime footprint — never flagged as missing.
_STRUCTURAL_IGNORE = {
    "Container", "Comment", "Browse", "BrowseV2", "BlockUntilDone",
    "RunCommand", "Select", "LockInSelect",  # Select may be a pure passthrough
}

_KNOWN_GLOBALS = {
    "spark", "F", "Window", "reduce", "dbutils", "display", "displayHTML",
    "True", "False", "None",
}


@dataclass
class StructuralIssue:
    """One mismatch between the .yxmd DAG and the generated PySpark."""
    severity: str            # "FAILED" or "WARNING"
    tool_id: Optional[int]
    tool_type: str
    message: str

    def to_dict(self) -> dict:
        return {
            "severity": self.severity,
            "tool_id": self.tool_id,
            "tool_type": self.tool_type,
            "message": self.message,
        }


@dataclass
class StructuralReport:
    """Result of structurally validating generated code against the DAG."""
    status: ValidationStatus = ValidationStatus.PASS
    syntax_ok: bool = True
    syntax_error: str = ""
    issues: list = field(default_factory=list)
    methods_found: dict = field(default_factory=dict)   # method name -> count
    tools_checked: int = 0

    @property
    def failures(self) -> list:
        return [i for i in self.issues if i.severity == "FAILED"]

    @property
    def warnings(self) -> list:
        return [i for i in self.issues if i.severity == "WARNING"]

    def to_dict(self) -> dict:
        return {
            "status": self.status.value,
            "syntax_ok": self.syntax_ok,
            "syntax_error": self.syntax_error,
            "tools_checked": self.tools_checked,
            "failures": [i.to_dict() for i in self.failures],
            "warnings": [i.to_dict() for i in self.warnings],
        }


def strip_databricks_magics(code: str) -> str:
    """Convert Databricks source-format notebook text into plain Python."""
    lines = []
    for line in code.splitlines():
        stripped = line.strip()
        if stripped.startswith("# MAGIC") or stripped == "# Databricks notebook source":
            continue
        if stripped == "# COMMAND ----------":
            continue
        lines.append(line)
    return "\n".join(lines)


class _CodeInspector(ast.NodeVisitor):
    """Collects method calls, assignments and name usages from generated code."""

    def __init__(self):
        self.method_calls: dict = {}      # method name -> count
        self.assigned_names: set = set()
        self.used_receivers: list = []    # (name, lineno) of x in x.method(...)
        self.join_types: list = []        # string literals passed as 3rd arg to .join
        self.string_args: set = set()     # all string literals seen in calls

    def visit_Assign(self, node):
        for target in node.targets:
            if isinstance(target, ast.Name):
                self.assigned_names.add(target.id)
            elif isinstance(target, ast.Tuple):
                for elt in target.elts:
                    if isinstance(elt, ast.Name):
                        self.assigned_names.add(elt.id)
        self.generic_visit(node)

    def visit_Import(self, node):
        for alias in node.names:
            self.assigned_names.add(alias.asname or alias.name.split(".")[0])
        self.generic_visit(node)

    def _add_arg_names(self, args_node):
        for a in list(args_node.posonlyargs) + list(args_node.args) + list(args_node.kwonlyargs):
            self.assigned_names.add(a.arg)
        if args_node.vararg:
            self.assigned_names.add(args_node.vararg.arg)
        if args_node.kwarg:
            self.assigned_names.add(args_node.kwarg.arg)

    def visit_Lambda(self, node):
        self._add_arg_names(node.args)
        self.generic_visit(node)

    def visit_FunctionDef(self, node):
        self.assigned_names.add(node.name)
        self._add_arg_names(node.args)
        self.generic_visit(node)

    def visit_For(self, node):
        for target in ast.walk(node.target):
            if isinstance(target, ast.Name):
                self.assigned_names.add(target.id)
        self.generic_visit(node)

    def visit_comprehension(self, node):
        for target in ast.walk(node.target):
            if isinstance(target, ast.Name):
                self.assigned_names.add(target.id)
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        for alias in node.names:
            self.assigned_names.add(alias.asname or alias.name)
        self.generic_visit(node)

    def visit_Call(self, node):
        func = node.func
        if isinstance(func, ast.Attribute):
            name = func.attr
            self.method_calls[name] = self.method_calls.get(name, 0) + 1
            # Track the base receiver variable: df_x.filter(...) -> df_x
            base = func.value
            while isinstance(base, (ast.Attribute, ast.Call)):
                base = base.value if isinstance(base, ast.Attribute) else base.func
            if isinstance(base, ast.Name):
                self.used_receivers.append((base.id, node.lineno))
            if name == "join" and len(node.args) >= 3:
                third = node.args[2]
                if isinstance(third, ast.Constant) and isinstance(third.value, str):
                    self.join_types.append(third.value)
        for arg in ast.walk(node):
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                self.string_args.add(arg.value)
        self.generic_visit(node)


class StructuralValidator:
    """
    Compares generated PySpark code against the original Alteryx DAG.

    FAILED  -> broken or missing logic (syntax error, missing tool operation,
               unresolved DataFrame reference, missing join keys)
    WARNING -> logic present but suboptimal or needing review (TODO markers,
               unsupported-tool passthroughs, missing broadcast/Delta hints)
    """

    def validate(self, code: str, workflow, context: Optional[dict] = None) -> StructuralReport:
        report = StructuralReport()
        python_src = strip_databricks_magics(code)

        # 1. Syntax gate — non-parseable code is an automatic FAIL.
        try:
            tree = ast.parse(python_src)
        except SyntaxError as exc:
            report.syntax_ok = False
            report.syntax_error = f"line {exc.lineno}: {exc.msg}"
            report.status = ValidationStatus.FAIL
            report.issues.append(StructuralIssue(
                "FAILED", None, "", f"Generated code has a syntax error: {report.syntax_error}"))
            return report

        inspector = _CodeInspector()
        inspector.visit(tree)
        report.methods_found = inspector.method_calls

        if context is None:
            context = workflow.get_unified_context()
        tools = context.get("tools", [])

        # 2. Tool coverage — every DAG tool must map to at least one operation.
        for tool in tools:
            tt = tool.tool_type
            if tt in _STRUCTURAL_IGNORE:
                continue
            report.tools_checked += 1
            expected = EXPECTED_OPERATIONS.get(tt)
            if expected is None:
                report.issues.append(StructuralIssue(
                    "WARNING", tool.tool_id, tt,
                    f"No structural expectation defined for tool type '{tt}'; "
                    f"verify manually."))
                continue
            if not expected & set(inspector.method_calls):
                report.issues.append(StructuralIssue(
                    "FAILED", tool.tool_id, tt,
                    f"Expected one of {sorted(expected)} for Alteryx {tt} "
                    f"(tool {tool.tool_id}) but none was found in the generated code."))

        # 3. Join fidelity — configured keys must appear in the code.
        for tool in tools:
            if tool.tool_type not in ("Join", "LockInJoin"):
                continue
            pc = tool.parsed_config or {}
            for key in set(pc.get("left_keys", []) + pc.get("right_keys", [])):
                if key and key not in inspector.string_args and key not in python_src:
                    report.issues.append(StructuralIssue(
                        "FAILED", tool.tool_id, tool.tool_type,
                        f"Join key '{key}' from the .yxmd configuration is missing "
                        f"from the generated join."))

        # 4. Filter fidelity — referenced fields must survive conversion.
        for tool in tools:
            if tool.tool_type not in ("Filter", "LockInFilter"):
                continue
            expr = (tool.parsed_config or {}).get("expression", "")
            for fld in _re.findall(r"\[([^\]]+)\]", expr):
                if fld not in python_src:
                    report.issues.append(StructuralIssue(
                        "FAILED", tool.tool_id, tool.tool_type,
                        f"Filter field '{fld}' from expression {expr!r} is missing "
                        f"from the generated code."))

        # 5. Unresolved DataFrame references (e.g. df_5_input placeholders).
        for name, lineno in inspector.used_receivers:
            if name in inspector.assigned_names or name in _KNOWN_GLOBALS:
                continue
            report.issues.append(StructuralIssue(
                "FAILED", None, "",
                f"Unresolved DataFrame reference '{name}' at line {lineno} — "
                f"an upstream tool likely failed to convert."))

        # 6. Warnings: TODOs, unsupported passthroughs, optimization hints.
        for m in _re.finditer(r"#.*TODO[:\s](.*)", code):
            report.issues.append(StructuralIssue(
                "WARNING", None, "", f"TODO placeholder: {m.group(1).strip()[:100]}"))
        if "UNSUPPORTED TOOL" in code:
            report.issues.append(StructuralIssue(
                "WARNING", None, "",
                "Notebook contains unsupported-tool passthroughs; review flagged cells."))
        if inspector.method_calls.get("join") and "F.broadcast(" not in python_src:
            small_inputs = any(t.tool_type == "TextInput" for t in tools)
            if small_inputs:
                report.issues.append(StructuralIssue(
                    "WARNING", None, "Join",
                    "Join with a small inline (TextInput) side could use "
                    "F.broadcast() for better performance."))
        if ".write.csv" in python_src:
            report.issues.append(StructuralIssue(
                "WARNING", None, "OutputData",
                "CSV write detected — consider Delta Lake format for Databricks."))

        # Final status
        if report.failures:
            report.status = ValidationStatus.FAIL
        elif report.warnings:
            report.status = ValidationStatus.WARNING
        else:
            report.status = ValidationStatus.PASS
        return report


def run_generated_code(code: str, spark, extra_globals: Optional[dict] = None) -> dict:
    """
    Execute generated PySpark against a live Spark session (sample-data
    validation). Returns {"ok": bool, "error": str, "dataframes": {name: {rows, columns}}}.

    SECURITY: this exec()s code derived from the input .yxmd with full driver
    privileges. It never runs implicitly — a caller must opt in by passing a
    Spark session (CLI never does; the notebook gates it behind the
    run_sample_validation widget). Only enable it for workflows you trust, and
    prefer a disposable cluster/sandbox for third-party files.

    Only DataFrame-producing assignments are inspected; write operations in
    generated notebooks are commented out by default so this is side-effect free.
    """
    from pyspark.sql import functions as F  # noqa: F401
    from pyspark.sql.window import Window  # noqa: F401
    from functools import reduce  # noqa: F401

    namespace = {"spark": spark, "F": F, "Window": Window, "reduce": reduce}
    namespace.update(extra_globals or {})
    python_src = strip_databricks_magics(code)

    result: dict = {"ok": True, "error": "", "dataframes": {}}
    try:
        exec(compile(python_src, "<generated_notebook>", "exec"), namespace)
    except Exception as exc:
        result["ok"] = False
        result["error"] = f"{type(exc).__name__}: {exc}"
        return result

    for name, value in namespace.items():
        if name.startswith("df_") and hasattr(value, "columns") and hasattr(value, "count"):
            try:
                result["dataframes"][name] = {
                    "rows": value.count(),
                    "columns": list(value.columns),
                }
            except Exception as exc:
                result["ok"] = False
                if not result["error"]:  # keep the FIRST failure, not the last
                    result["error"] = f"DataFrame '{name}' failed to evaluate: {exc}"
    return result
