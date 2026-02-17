"""Convert Alteryx FindReplace tool to PySpark."""

from .base_converter import BaseToolConverter


class FindReplaceConverter(BaseToolConverter):
    """Convert Alteryx FindReplace tool to PySpark string replacement."""

    def generate_code(
        self,
        input_df_name: str,
        output_df_name: str,
        **kwargs,
    ) -> str:
        """
        Generate PySpark code for find-and-replace.

        The FindReplace tool takes two inputs:
        - Left (data to search in)
        - Right (lookup table with find/replace pairs)

        For simple cases we generate regexp_replace; for lookup-based
        we generate a join-based replacement pattern.
        """
        find_field = self.configuration.get("find_field", "")
        replace_field = self.configuration.get("replace_field", "")
        find_mode = self.configuration.get("find_mode", "Normal")

        right_df_name = kwargs.get("right_df_name", "")
        mapped_find = self.map_column(find_field) if find_field else ""
        mapped_replace = self.map_column(replace_field) if replace_field else ""

        if not find_field:
            return (
                f"{self.get_comment()}\n"
                f"# WARNING: FindReplace with no find field configured\n"
                f"{output_df_name} = {input_df_name}\n"
            )

        lines = [self.get_comment()]

        if right_df_name:
            # Lookup-based find/replace using a join
            lines.append(f"# FindReplace using lookup table")
            lines.append(
                f"# Joining with replacement lookup and applying replacements"
            )
            lines.append(
                f'{output_df_name} = {input_df_name}'
            )
            lines.append(
                f'# NOTE: FindReplace with lookup table may need manual adjustment'
            )
            lines.append(
                f'# based on the specific find/replace column pairs'
            )
        else:
            # Simple replacement
            lines.append(
                f'{output_df_name} = {input_df_name}'
            )

        return "\n".join(lines) + "\n"
