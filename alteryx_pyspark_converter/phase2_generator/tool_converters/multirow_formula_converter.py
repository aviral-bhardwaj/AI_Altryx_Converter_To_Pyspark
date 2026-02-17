"""Convert Alteryx MultiRowFormula tool to PySpark window functions."""

from .base_converter import BaseToolConverter


class MultiRowFormulaConverter(BaseToolConverter):
    """Convert Alteryx MultiRowFormula to PySpark Window functions."""

    def generate_code(
        self,
        input_df_name: str,
        output_df_name: str,
        **kwargs,
    ) -> str:
        """
        Generate PySpark code using Window functions for multi-row formulas.

        Alteryx MultiRowFormula accesses rows above/below the current row.
        In PySpark this translates to Window functions with lag/lead.
        """
        field = self.configuration.get("field", "")
        expression = self.configuration.get("expression", "")
        num_rows = self.configuration.get("num_rows", 1)
        group_fields = self.configuration.get("group_fields", [])

        mapped_field = self.map_column(field)

        if not field or not expression:
            return (
                f"{self.get_comment()}\n"
                f"# WARNING: MultiRowFormula with empty field/expression\n"
                f"{output_df_name} = {input_df_name}\n"
            )

        lines = [self.get_comment()]

        # Build the window specification
        if group_fields:
            mapped_groups = self.map_columns(group_fields)
            partition_str = ", ".join(f'F.col("{g}")' for g in mapped_groups)
            lines.append(
                f"_window = Window.partitionBy({partition_str})"
                f".orderBy(F.monotonically_increasing_id())"
                f".rowsBetween(-{num_rows}, {num_rows})"
            )
        else:
            lines.append(
                f"_window = Window.orderBy(F.monotonically_increasing_id())"
                f".rowsBetween(-{num_rows}, {num_rows})"
            )

        # Convert the expression - replace Row-N/Row+N references
        pyspark_expr = self._convert_multirow_expression(expression, num_rows)

        lines.append(
            f'{output_df_name} = {input_df_name}.withColumn('
            f'"{mapped_field}", {pyspark_expr})'
        )

        return "\n".join(lines) + "\n"

    def _convert_multirow_expression(
        self, expression: str, num_rows: int
    ) -> str:
        """Convert Alteryx multi-row expression to PySpark.

        Handles Row-1:[field], Row+1:[field] references by converting
        them to F.lag() / F.lead() calls.
        """
        import re

        result = expression

        # Replace Row-N:[field] with lag
        def replace_prev_row(match):
            offset = int(match.group(1))
            col_name = match.group(2)
            mapped = self.map_column(col_name)
            return f'F.lag(F.col("{mapped}"), {offset}).over(_window)'

        result = re.sub(
            r'Row-(\d+):\[([^\]]+)\]',
            replace_prev_row,
            result,
        )

        # Replace Row+N:[field] with lead
        def replace_next_row(match):
            offset = int(match.group(1))
            col_name = match.group(2)
            mapped = self.map_column(col_name)
            return f'F.lead(F.col("{mapped}"), {offset}).over(_window)'

        result = re.sub(
            r'Row\+(\d+):\[([^\]]+)\]',
            replace_next_row,
            result,
        )

        # Convert remaining [field] references
        result = self.expr_converter.convert(result) if result else 'F.lit(None)'

        return result
