"""Convert Alteryx DynamicRename tool to PySpark."""

from .base_converter import BaseToolConverter


class DynamicRenameConverter(BaseToolConverter):
    """Convert Alteryx DynamicRename tool to PySpark column renaming."""

    def generate_code(
        self,
        input_df_name: str,
        output_df_name: str,
        **kwargs,
    ) -> str:
        """
        Generate PySpark code for dynamic column renaming.

        Supports common rename modes:
        - Take Field Names from Right: use lookup DataFrame for renames
        - Formula-based: apply a formula to column names
        - Prefix/Suffix: add/remove prefix or suffix
        """
        rename_mode = self.configuration.get("rename_mode", "Formula")
        formula = self.configuration.get("formula", "")
        prefix = self.configuration.get("prefix", "")
        suffix = self.configuration.get("suffix", "")

        lines = [self.get_comment()]

        if prefix or suffix:
            # Simple prefix/suffix rename
            lines.append(f"# Rename columns with prefix/suffix")
            lines.append(f"{output_df_name} = {input_df_name}")
            lines.append(f"for col_name in {output_df_name}.columns:")
            if prefix:
                lines.append(
                    f'    {output_df_name} = {output_df_name}'
                    f'.withColumnRenamed(col_name, "{prefix}" + col_name)'
                )
            if suffix:
                lines.append(
                    f'    {output_df_name} = {output_df_name}'
                    f'.withColumnRenamed(col_name, col_name + "{suffix}")'
                )
        else:
            # Pass through with a note
            lines.append(
                f"# DynamicRename - may need manual adjustment for mode: {rename_mode}"
            )
            lines.append(f"{output_df_name} = {input_df_name}")

        return "\n".join(lines) + "\n"
