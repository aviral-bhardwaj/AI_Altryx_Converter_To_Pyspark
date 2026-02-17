"""Convert Alteryx AppendFields tool to PySpark."""

from .base_converter import BaseToolConverter


class AppendFieldsConverter(BaseToolConverter):
    """Convert Alteryx AppendFields tool to PySpark crossJoin."""

    def generate_code(
        self,
        input_df_name: str,
        output_df_name: str,
        **kwargs,
    ) -> str:
        """
        Generate PySpark crossJoin code.

        AppendFields in Alteryx appends every row from the Source (Right)
        to every row in the Target (Left), essentially a cross join.
        When Source has a single row, it broadcasts that row to all Target rows.
        """
        right_df_name = kwargs.get("right_df_name", f"{input_df_name}_right")

        lines = [self.get_comment()]

        # AppendFields is a cross join (typically source has 1 row)
        lines.append(
            f"{output_df_name} = {input_df_name}.crossJoin({right_df_name})"
        )

        # Handle column selection if specified
        select_fields = self.configuration.get("select_fields", [])
        if select_fields:
            for field_cfg in select_fields:
                name = field_cfg.get("name", "")
                if name == "*Unknown":
                    continue
                if not field_cfg.get("selected", True):
                    mapped = self.map_column(name)
                    lines.append(
                        f'{output_df_name} = {output_df_name}.drop("{mapped}")'
                    )
                elif field_cfg.get("rename"):
                    old_name = self.map_column(name)
                    new_name = self.map_column(field_cfg["rename"])
                    lines.append(
                        f'{output_df_name} = {output_df_name}'
                        f'.withColumnRenamed("{old_name}", "{new_name}")'
                    )

        return "\n".join(lines) + "\n"
