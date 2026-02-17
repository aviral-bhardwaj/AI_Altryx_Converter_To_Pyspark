"""Convert Alteryx formula expressions to PySpark expressions."""

import re
import logging
from typing import Optional

from ..config.column_mapping import ColumnMapper
from .function_mapper import FUNCTION_MAP, convert_date_format
from .operator_mapper import COMPARISON_OPERATORS, LOGICAL_OPERATORS

logger = logging.getLogger(__name__)


class AlteryxExpressionConverter:
    """
    Convert Alteryx formula expressions to PySpark expressions.

    Handles:
    - Column references: [col_name] -> F.col("mapped_name")
    - Comparison operators: = -> ==, <> -> !=
    - Logical operators: AND -> &, OR -> |, NOT -> ~
    - Function calls: Contains([col], "x") -> F.col("col").contains("x")
    - IF/ELSEIF/ELSE/ENDIF -> F.when().when().otherwise()
    - String literals: "text" -> "text"
    - Numeric literals: 123 -> 123
    - Null references: Null() -> F.lit(None)
    """

    # Regex patterns
    COLUMN_REF = re.compile(r'\[([^\]]+)\]')
    STRING_LITERAL = re.compile(r'"([^"]*)"')
    FUNCTION_CALL = re.compile(r'(\w+)\s*\(')
    IF_PATTERN = re.compile(
        r'\bIF\b', re.IGNORECASE
    )
    ELSEIF_PATTERN = re.compile(r'\bELSEIF\b', re.IGNORECASE)
    ELSE_PATTERN = re.compile(r'\bELSE\b', re.IGNORECASE)
    ENDIF_PATTERN = re.compile(r'\bENDIF\b', re.IGNORECASE)
    THEN_PATTERN = re.compile(r'\bTHEN\b', re.IGNORECASE)

    def __init__(self, column_mapper: ColumnMapper):
        self.column_mapper = column_mapper

    def convert(self, alteryx_expr: str) -> str:
        """
        Convert an Alteryx expression to PySpark.

        Args:
            alteryx_expr: The Alteryx formula expression.

        Returns:
            Equivalent PySpark expression string.
        """
        if not alteryx_expr or not alteryx_expr.strip():
            return ""

        expr = alteryx_expr.strip()

        # Normalize whitespace
        expr = re.sub(r'\s+', ' ', expr)

        # Check if this is an IF expression
        if self.IF_PATTERN.search(expr):
            return self.convert_if_expression(expr)

        # Check if this is a function call
        func_match = self.FUNCTION_CALL.match(expr.strip())
        if func_match:
            func_name = func_match.group(1)
            if func_name in FUNCTION_MAP and func_name not in ('IF', 'THEN', 'ELSE', 'ENDIF'):
                return self._convert_function_call(expr)

        # Convert simple expression
        return self._convert_simple_expression(expr)

    def convert_if_expression(self, alteryx_expr: str) -> str:
        """
        Convert IF/ELSEIF/ELSE/ENDIF to F.when().when().otherwise().

        Example:
            IF [val] > 10 THEN "High"
            ELSEIF [val] > 5 THEN "Medium"
            ELSE "Low"
            ENDIF
            ->
            F.when(F.col("val") > 10, F.lit("High"))
             .when(F.col("val") > 5, F.lit("Medium"))
             .otherwise(F.lit("Low"))
        """
        expr = alteryx_expr.strip()

        # Parse the IF expression into branches
        branches = self._parse_if_branches(expr)

        if not branches:
            logger.warning("Could not parse IF expression: %s", expr)
            return f'F.expr("""{alteryx_expr}""")'

        parts = []
        otherwise_value = None

        for branch in branches:
            condition = branch.get("condition")
            value = branch.get("value", "")

            if condition is None:
                # This is the ELSE branch
                otherwise_value = self._convert_value_expr(value)
            else:
                pyspark_condition = self._convert_simple_expression(condition)
                pyspark_value = self._convert_value_expr(value)

                if not parts:
                    parts.append(f"F.when({pyspark_condition}, {pyspark_value})")
                else:
                    parts.append(f".when({pyspark_condition}, {pyspark_value})")

        result = "".join(parts)

        if otherwise_value is not None:
            result += f".otherwise({otherwise_value})"

        return result

    def convert_column_reference(self, alteryx_expr: str) -> str:
        """
        Convert [column_name] references to F.col("mapped_name").

        Args:
            alteryx_expr: Expression containing [column] references.

        Returns:
            Expression with F.col() references.
        """
        def replace_ref(match: re.Match) -> str:
            col_name = match.group(1)
            mapped = self.column_mapper.to_databricks(col_name)
            return f'F.col("{mapped}")'

        return self.COLUMN_REF.sub(replace_ref, alteryx_expr)

    def convert_filter_expression(self, alteryx_expr: str) -> str:
        """
        Convert a filter expression for use in .filter().

        This is similar to convert() but optimized for filter contexts.
        """
        return self.convert(alteryx_expr)

    def _convert_simple_expression(self, expr: str) -> str:
        """Convert a simple (non-IF) expression to PySpark."""
        result = expr

        # Protect string literals from modification
        literals: list[str] = []
        def save_literal(match: re.Match) -> str:
            literals.append(match.group(0))
            return f"__STR_{len(literals) - 1}__"

        result = self.STRING_LITERAL.sub(save_literal, result)

        # Convert column references
        result = self._replace_column_refs(result)

        # Convert operators
        result = self._replace_operators(result)

        # Convert Null() function
        result = re.sub(r'\bNull\(\)', 'F.lit(None)', result, flags=re.IGNORECASE)

        # Restore string literals
        for i, lit in enumerate(literals):
            result = result.replace(f"__STR_{i}__", lit)

        # Convert string concatenation with + to F.concat()
        result = self._convert_string_concat(result)

        return result

    def _convert_string_concat(self, expr: str) -> str:
        """Convert Alteryx string '+' concatenation to F.concat().

        In Alteryx, strings are concatenated with +:
            [FirstName] + " " + [LastName]
        becomes:
            F.concat(F.col("FirstName"), F.lit(" "), F.col("LastName"))
        """
        # Only convert if there's a + between string/col expressions
        # Check if expression contains + with string or column operands
        if '+' not in expr:
            return expr

        # Split by + while respecting parentheses and strings
        parts = self._split_by_plus(expr)
        if len(parts) <= 1:
            return expr

        # Check if this looks like string concatenation (not arithmetic)
        has_string_or_col = any(
            'F.col(' in p or 'F.lit(' in p or p.strip().startswith('"')
            for p in parts
        )
        has_numeric = all(
            self._is_numeric_expr(p.strip()) for p in parts
        )

        if has_string_or_col and not has_numeric:
            converted = []
            for part in parts:
                part = part.strip()
                if part.startswith('"') and part.endswith('"'):
                    converted.append(f'F.lit({part})')
                else:
                    converted.append(part)
            return f"F.concat({', '.join(converted)})"

        return expr

    def _split_by_plus(self, expr: str) -> list[str]:
        """Split expression by + operator, respecting parentheses and strings."""
        parts = []
        current = []
        depth = 0
        in_string = False

        for char in expr:
            if char == '"' and (not current or current[-1] != '\\'):
                in_string = not in_string
                current.append(char)
            elif in_string:
                current.append(char)
            elif char == '(':
                depth += 1
                current.append(char)
            elif char == ')':
                depth -= 1
                current.append(char)
            elif char == '+' and depth == 0:
                parts.append(''.join(current).strip())
                current = []
            else:
                current.append(char)

        if current:
            remaining = ''.join(current).strip()
            if remaining:
                parts.append(remaining)

        return parts

    def _is_numeric_expr(self, expr: str) -> bool:
        """Check if an expression is purely numeric."""
        try:
            float(expr)
            return True
        except ValueError:
            return False

    def _convert_function_call(self, expr: str) -> str:
        """Convert an Alteryx function call to PySpark."""
        # Extract function name and arguments
        func_match = re.match(r'(\w+)\s*\((.*)\)$', expr.strip(), re.DOTALL)
        if not func_match:
            return self.convert_column_reference(expr)

        func_name = func_match.group(1)
        args_str = func_match.group(2).strip()

        # Parse arguments (handling nested parentheses and string literals)
        args = self._parse_function_args(args_str)

        if func_name in FUNCTION_MAP:
            _, template = FUNCTION_MAP[func_name]
            return self._apply_function_template(template, args)

        # Unknown function: wrap in F.expr()
        converted_args = [self.convert(a.strip()) for a in args]
        return f'F.expr("{func_name}({", ".join(converted_args)})")'

    def _apply_function_template(
        self, template: str, args: list[str]
    ) -> str:
        """Apply a function template with converted arguments."""
        converted_args = []
        for arg in args:
            arg = arg.strip()

            # Check if the arg is a column reference
            col_match = self.COLUMN_REF.fullmatch(arg)
            if col_match:
                col_name = col_match.group(1)
                mapped = self.column_mapper.to_databricks(col_name)
                converted_args.append(mapped)
            else:
                # Convert the argument as an expression
                converted_args.append(self._convert_value_expr(arg))

        # Fill template
        result = template
        for i, arg in enumerate(converted_args):
            result = result.replace(f"{{{i}}}", str(arg))

        return result

    def _convert_value_expr(self, expr: str) -> str:
        """Convert a value expression (used in THEN/ELSE clauses)."""
        expr = expr.strip()

        # String literal
        if expr.startswith('"') and expr.endswith('"'):
            return f'F.lit({expr})'

        # Numeric literal
        try:
            float(expr)
            return f"F.lit({expr})"
        except ValueError:
            pass

        # Boolean
        if expr.lower() in ("true", "false"):
            return f"F.lit({expr.capitalize()})"

        # Null
        if expr.lower() == "null()" or expr.lower() == "null":
            return "F.lit(None)"

        # Column reference
        col_match = self.COLUMN_REF.fullmatch(expr)
        if col_match:
            mapped = self.column_mapper.to_databricks(col_match.group(1))
            return f'F.col("{mapped}")'

        # Complex expression - recursively convert
        return self.convert(expr)

    def _replace_column_refs(self, expr: str) -> str:
        """Replace [column_name] with F.col("mapped_name")."""
        def replace_ref(match: re.Match) -> str:
            col_name = match.group(1)
            mapped = self.column_mapper.to_databricks(col_name)
            return f'F.col("{mapped}")'

        return self.COLUMN_REF.sub(replace_ref, expr)

    def _replace_operators(self, expr: str) -> str:
        """Replace Alteryx operators with PySpark equivalents."""
        result = expr

        # Replace comparison operators (careful with = vs ==)
        # Only replace standalone = (not == or !=)
        result = re.sub(r'(?<![!<>=])=(?!=)', ' == ', result)
        result = result.replace('<>', ' != ')

        # Replace logical operators (word boundary matching)
        result = re.sub(r'\bAND\b', '&', result, flags=re.IGNORECASE)
        result = re.sub(r'\bOR\b', '|', result, flags=re.IGNORECASE)
        result = re.sub(r'\bNOT\s+', '~', result, flags=re.IGNORECASE)

        # Clean up extra whitespace
        result = re.sub(r'\s+', ' ', result).strip()

        return result

    def _parse_if_branches(self, expr: str) -> list[dict]:
        """Parse IF/ELSEIF/ELSE/ENDIF expression into branches."""
        branches = []

        # Tokenize by keywords
        tokens = re.split(
            r'\b(IF|THEN|ELSEIF|ELSE|ENDIF)\b',
            expr,
            flags=re.IGNORECASE,
        )

        current_condition: Optional[str] = None
        state = "start"

        for token in tokens:
            token = token.strip()
            if not token:
                continue

            upper = token.upper()

            if upper == "IF":
                state = "condition"
            elif upper == "ELSEIF":
                state = "condition"
            elif upper == "THEN":
                state = "value"
            elif upper == "ELSE":
                current_condition = None
                state = "else_value"
            elif upper == "ENDIF":
                break
            elif state == "condition" and token:
                current_condition = token
            elif state == "value" and token:
                branches.append({
                    "condition": current_condition,
                    "value": token,
                })
                current_condition = None
                state = "between"
            elif state == "else_value" and token:
                branches.append({
                    "condition": None,
                    "value": token,
                })

        return branches

    def _parse_function_args(self, args_str: str) -> list[str]:
        """
        Parse function arguments, handling nested parens and string literals.
        """
        args = []
        current = []
        depth = 0
        in_string = False

        for char in args_str:
            if char == '"' and (not current or current[-1] != '\\'):
                in_string = not in_string
                current.append(char)
            elif in_string:
                current.append(char)
            elif char == '(':
                depth += 1
                current.append(char)
            elif char == ')':
                depth -= 1
                current.append(char)
            elif char == ',' and depth == 0:
                args.append(''.join(current).strip())
                current = []
            else:
                current.append(char)

        if current:
            remaining = ''.join(current).strip()
            if remaining:
                args.append(remaining)

        return args
