"""Convert Alteryx Transpose tool to PySpark."""

from .base_converter import BaseToolConverter


class TransposeConverter(BaseToolConverter):
    """Convert Alteryx Transpose tool to PySpark stack (unpivot) operations."""

    def generate_code(
        self,
        input_df_name: str,
        output_df_name: str,
        **kwargs,
    ) -> str:
        """
        Generate PySpark unpivot/stack code.

        Converts wide data to long format by transposing data columns
        into rows, keeping key columns fixed.

        Alteryx Transpose output columns: [key_cols..., Name, Value]
        """
        key_fields = self.configuration.get("key_fields", [])
        data_fields = self.configuration.get("data_fields", [])

        mapped_keys = self.map_columns(key_fields) if key_fields else []
        mapped_data = self.map_columns(data_fields) if data_fields else []

        if not mapped_data:
            return (
                f"{self.get_comment()}\n"
                f"# WARNING: No data fields to transpose\n"
                f"{output_df_name} = {input_df_name}\n"
            )

        # Build the stack expression
        # stack(n, 'col_name_1', col_1, 'col_name_2', col_2, ...)
        n = len(mapped_data)
        stack_args = []
        for col in mapped_data:
            stack_args.append(f"'{col}', `{col}`")
        stack_expr = f"stack({n}, {', '.join(stack_args)}) as (Name, Value)"

        lines = [self.get_comment()]

        if mapped_keys:
            key_cols_str = ", ".join(f'"{k}"' for k in mapped_keys)
            lines.append(
                f"{output_df_name} = {input_df_name}"
                f".select({key_cols_str}, "
                f'F.expr("{stack_expr}"))'
            )
        else:
            lines.append(
                f'{output_df_name} = {input_df_name}.select('
                f'F.expr("{stack_expr}"))'
            )

        # Filter out null values (Alteryx Transpose skips nulls by default)
        lines.append(
            f'{output_df_name} = {output_df_name}.filter(F.col("Value").isNotNull())'
        )

        return "\n".join(lines) + "\n"
