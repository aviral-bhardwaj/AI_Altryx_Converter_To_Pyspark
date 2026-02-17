"""Convert Alteryx RecordID tool to PySpark."""

from .base_converter import BaseToolConverter


class RecordIDConverter(BaseToolConverter):
    """Convert Alteryx RecordID tool to PySpark monotonically_increasing_id."""

    def generate_code(
        self,
        input_df_name: str,
        output_df_name: str,
        **kwargs,
    ) -> str:
        """Generate PySpark code to add a sequential record ID column."""
        field_name = self.configuration.get("field_name", "RecordID")
        starting_value = self.configuration.get("starting_value", 1)
        mapped_field = self.map_column(field_name)

        if starting_value == 1:
            return (
                f"{self.get_comment()}\n"
                f"{output_df_name} = {input_df_name}.withColumn(\n"
                f'    "{mapped_field}",\n'
                f"    F.monotonically_increasing_id() + 1\n"
                f")\n"
            )
        else:
            return (
                f"{self.get_comment()}\n"
                f"{output_df_name} = {input_df_name}.withColumn(\n"
                f'    "{mapped_field}",\n'
                f"    F.monotonically_increasing_id() + {starting_value}\n"
                f")\n"
            )
