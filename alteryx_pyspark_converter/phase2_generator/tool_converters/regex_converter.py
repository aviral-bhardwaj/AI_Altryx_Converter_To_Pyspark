"""Convert Alteryx RegEx tool to PySpark."""

from .base_converter import BaseToolConverter


class RegExConverter(BaseToolConverter):
    """Convert Alteryx RegEx tool to PySpark regexp operations."""

    def generate_code(
        self,
        input_df_name: str,
        output_df_name: str,
        **kwargs,
    ) -> str:
        """
        Generate PySpark code for regex operations.

        Handles three output methods:
        - Replace: regexp_replace
        - Parse: regexp_extract (with capture groups)
        - Match: filter with rlike
        """
        field = self.configuration.get("field", "")
        expression = self.configuration.get("expression", "")
        output_method = self.configuration.get("output_method", "Replace")
        replace_expr = self.configuration.get("replace_expression", "")

        mapped_field = self.map_column(field)

        if not field or not expression:
            return (
                f"{self.get_comment()}\n"
                f"# WARNING: RegEx with empty field/expression\n"
                f"{output_df_name} = {input_df_name}\n"
            )

        lines = [self.get_comment()]

        if output_method == "Replace":
            # Replace matched pattern
            replace_with = replace_expr if replace_expr else ""
            lines.append(
                f'{output_df_name} = {input_df_name}.withColumn(\n'
                f'    "{mapped_field}",\n'
                f'    F.regexp_replace(F.col("{mapped_field}"), '
                f'r"{expression}", "{replace_with}")\n'
                f')'
            )

        elif output_method == "Parse":
            # Extract capture groups into new columns
            # Count capture groups in the regex
            import re
            groups = re.findall(r'\((?!\?)', expression)
            num_groups = len(groups)

            if num_groups == 0:
                # No capture groups - extract whole match
                lines.append(
                    f'{output_df_name} = {input_df_name}.withColumn(\n'
                    f'    "{mapped_field}_parsed",\n'
                    f'    F.regexp_extract(F.col("{mapped_field}"), '
                    f'r"{expression}", 0)\n'
                    f')'
                )
            else:
                # Extract each capture group
                current = input_df_name
                for i in range(1, num_groups + 1):
                    out_col = f"{mapped_field}_group{i}"
                    target = output_df_name if i == num_groups else current
                    lines.append(
                        f'{target} = {current}.withColumn(\n'
                        f'    "{out_col}",\n'
                        f'    F.regexp_extract(F.col("{mapped_field}"), '
                        f'r"{expression}", {i})\n'
                        f')'
                    )
                    current = target

        elif output_method == "Match":
            # Filter rows matching the pattern
            lines.append(
                f'{output_df_name} = {input_df_name}.filter(\n'
                f'    F.col("{mapped_field}").rlike(r"{expression}")\n'
                f')'
            )

        elif output_method == "Tokenize":
            # Split into tokens based on regex
            lines.append(
                f'{output_df_name} = {input_df_name}.withColumn(\n'
                f'    "{mapped_field}_tokens",\n'
                f'    F.split(F.col("{mapped_field}"), r"{expression}")\n'
                f')'
            )

        else:
            # Unknown method - default to replace
            lines.append(
                f'# WARNING: Unknown RegEx output method "{output_method}"\n'
                f'{output_df_name} = {input_df_name}'
            )

        return "\n".join(lines) + "\n"
