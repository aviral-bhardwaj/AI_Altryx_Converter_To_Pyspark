"""
Alteryx Expression Parser
==========================
Parses Alteryx formula expressions and converts them to PySpark equivalents.

Handles:
- Field references: [FieldName] -> F.col("FieldName")
- Function calls: Trim(Left([Field], 5)) -> F.trim(F.substring(F.col("Field"), 1, 5))
- IF/ELSEIF/ELSE/ENDIF blocks -> F.when(...).when(...).otherwise(...)
- IIF(cond, true_val, false_val) -> F.when(cond, true_val).otherwise(false_val)
- Operators: +, -, *, /, =, !=, >, <, >=, <=, AND, OR, NOT
- Nested expressions
- String literals, numeric literals, NULL
"""

import re
from typing import Optional


# Alteryx function -> PySpark expression mapping
# Each entry: (alteryx_func, pyspark_template, arg_count_or_range)
FUNCTION_MAP = {
    # String functions
    "Trim": ("F.trim({0})", 1),
    "TRIM": ("F.trim({0})", 1),
    "LTrim": ("F.ltrim({0})", 1),
    "RTrim": ("F.rtrim({0})", 1),
    "Left": ("F.substring({0}, 1, {1})", 2),
    "Right": ("F.substring({0}, -{1}, {1})", 2),
    "Substring": ("F.substring({0}, {1} + 1, {2})", 3),  # Alteryx 0-based -> Spark 1-based
    "Length": ("F.length({0})", 1),
    "UPPERCASE": ("F.upper({0})", 1),
    "LOWERCASE": ("F.lower({0})", 1),
    "Upper": ("F.upper({0})", 1),
    "Lower": ("F.lower({0})", 1),
    "PadLeft": ("F.lpad({0}, {1}, {2})", 3),
    "PadRight": ("F.rpad({0}, {1}, {2})", 3),
    "Contains": ("{0}.contains({1})", 2),
    "StartsWith": ("{0}.startswith({1})", 2),
    "EndsWith": ("{0}.endswith({1})", 2),
    "FindString": ("(F.locate({1}, {0}) - 1)", 2),  # Returns -1 if not found
    "ReplaceString": ("F.regexp_replace({0}, {1}, {2})", 3),
    "GetWord": ("F.split({0}, \" \")[{1}]", 2),
    "ToString": ("{0}.cast(\"string\")", 1),
    "ToNumber": ("{0}.cast(\"double\")", 1),
    "ToInteger": ("{0}.cast(\"int\")", 1),

    # Null functions
    "IsNull": ("{0}.isNull()", 1),
    "IsEmpty": ("(({0}.isNull()) | (F.trim({0}) == F.lit(\"\")))", 1),
    "IFNULL": ("F.coalesce({0}, {1})", 2),
    "Null": ("F.lit(None)", 0),

    # Math functions
    "Abs": ("F.abs({0})", 1),
    "ABS": ("F.abs({0})", 1),
    "Ceil": ("F.ceil({0})", 1),
    "Floor": ("F.floor({0})", 1),
    "Round": ("F.round({0}, {1})", 2),
    "Pow": ("F.pow({0}, {1})", 2),
    "Log": ("F.log({0})", 1),
    "Log10": ("F.log10({0})", 1),
    "Sqrt": ("F.sqrt({0})", 1),
    "Mod": ("({0} % {1})", 2),
    "MIN": ("F.least({0}, {1})", 2),
    "MAX": ("F.greatest({0}, {1})", 2),
    "RandInt": ("(F.rand() * {0}).cast(\"int\")", 1),

    # Date functions
    "DateTimeParse": ("F.to_timestamp({0}, {1})", 2),
    "DateTimeFormat": ("F.date_format({0}, {1})", 2),
    "DateTimeAdd": ("F.date_add({0}, {1})", 2),
    "DateTimeDiff": ("F.datediff({0}, {1})", 2),
    "DateTimeNow": ("F.current_timestamp()", 0),
    "DateTimeToday": ("F.current_date()", 0),
    "DateTimeYear": ("F.year({0})", 1),
    "DateTimeMonth": ("F.month({0})", 1),
    "DateTimeDay": ("F.dayofmonth({0})", 1),

    # Regex functions
    "REGEX_Match": ("{0}.rlike({1})", 2),
    "REGEX_Replace": ("F.regexp_replace({0}, {1}, {2})", 3),
    "REGEX_CountMatches": ("F.size(F.split({0}, {1})) - 1", 2),

    # Type functions
    "Concat": ("F.concat({0}, {1})", 2),
}

# Alteryx date format tokens -> Spark date format tokens
DATE_FORMAT_MAP = {
    "yyyy": "yyyy",
    "yy": "yy",
    "MMMM": "MMMM",
    "MMM": "MMM",
    "MM": "MM",
    "M": "M",
    "dd": "dd",
    "d": "d",
    "HH": "HH",
    "hh": "hh",
    "mm": "mm",
    "ss": "ss",
    "Month": "MMMM",
    "Mon": "MMM",
    "Day": "EEEE",
    "Dy": "EEE",
}


def convert_date_format(alteryx_fmt: str) -> str:
    """Convert an Alteryx date format string to a Spark date format string."""
    result = alteryx_fmt
    # Sort by length descending to replace longer tokens first
    for alt_token, spark_token in sorted(DATE_FORMAT_MAP.items(), key=lambda x: -len(x[0])):
        result = result.replace(alt_token, spark_token)
    return result


class Token:
    """A lexer token."""
    __slots__ = ("type", "value")

    def __init__(self, type_: str, value: str):
        self.type = type_
        self.value = value

    def __repr__(self):
        return f"Token({self.type}, {self.value!r})"


class ExpressionLexer:
    """Tokenize an Alteryx expression string."""

    # Token patterns in priority order
    PATTERNS = [
        ("WHITESPACE", r"\s+"),
        ("STRING", r"'(?:[^'\\]|\\.)*'"),
        ("STRING_DQ", r'"(?:[^"\\]|\\.)*"'),
        ("NUMBER", r"\d+(?:\.\d+)?"),
        ("FIELD_REF", r"\[([^\]]+)\]"),
        ("LPAREN", r"\("),
        ("RPAREN", r"\)"),
        ("COMMA", r","),
        ("OP_NEQ", r"!=|<>"),
        ("OP_GTE", r">="),
        ("OP_LTE", r"<="),
        ("OP_EQ", r"==?"),  # Both = and ==
        ("OP_GT", r">"),
        ("OP_LT", r"<"),
        ("OP_PLUS", r"\+"),
        ("OP_MINUS", r"-"),
        ("OP_MUL", r"\*"),
        ("OP_DIV", r"/"),
        ("KEYWORD", r"\b(?:IF|THEN|ELSEIF|ELSE|ENDIF|AND|OR|NOT|IN|IIF|True|False|NULL|Null)\b"),
        ("IDENT", r"[A-Za-z_][A-Za-z0-9_]*"),
    ]

    def __init__(self):
        self._regex = re.compile(
            "|".join(f"(?P<{name}>{pattern})" for name, pattern in self.PATTERNS)
        )

    def tokenize(self, expression: str) -> list:
        """Tokenize an expression string into a list of Tokens."""
        tokens = []
        pos = 0
        for match in self._regex.finditer(expression):
            if match.start() != pos:
                # Skip unexpected characters
                pass
            pos = match.end()
            for name, _ in self.PATTERNS:
                value = match.group(name)
                if value is not None:
                    if name == "WHITESPACE":
                        break
                    if name == "STRING_DQ":
                        # Normalize double-quoted strings to single-quoted
                        name = "STRING"
                        value = "'" + value[1:-1].replace("'", "\\'") + "'"
                    tokens.append(Token(name, value))
                    break
        return tokens


class ExpressionParser:
    """
    Recursive descent parser for Alteryx expressions.
    Converts to PySpark expression strings.
    """

    def __init__(self):
        self._lexer = ExpressionLexer()
        self._tokens = []
        self._pos = 0

    def parse(self, expression: str) -> str:
        """
        Parse an Alteryx expression and return a PySpark expression string.

        Examples:
            "[Status] = 'Active'" -> "F.col(\"Status\") == F.lit(\"Active\")"
            "Trim(Left([Name], 5))" -> "F.trim(F.substring(F.col(\"Name\"), 1, 5))"
            "IF [Score] > 10 THEN 'High' ELSE 'Low' ENDIF"
                -> "F.when(F.col(\"Score\") > F.lit(10), F.lit(\"High\")).otherwise(F.lit(\"Low\"))"
        """
        if not expression or not expression.strip():
            return ""

        self._tokens = self._lexer.tokenize(expression)
        self._pos = 0

        if not self._tokens:
            return ""

        result = self._parse_or_expr()
        return result

    # --- Token helpers ---

    def _peek(self) -> Optional[Token]:
        if self._pos < len(self._tokens):
            return self._tokens[self._pos]
        return None

    def _advance(self) -> Token:
        tok = self._tokens[self._pos]
        self._pos += 1
        return tok

    def _expect(self, type_: str, value: str = None) -> Token:
        tok = self._peek()
        if tok is None:
            raise SyntaxError(f"Expected {type_} but got end of expression")
        if tok.type != type_ or (value and tok.value != value):
            raise SyntaxError(f"Expected {type_}({value}) but got {tok}")
        return self._advance()

    def _match_keyword(self, keyword: str) -> bool:
        tok = self._peek()
        if tok and tok.type == "KEYWORD" and tok.value.upper() == keyword.upper():
            self._advance()
            return True
        return False

    # --- Grammar rules ---

    def _parse_or_expr(self) -> str:
        """or_expr: and_expr (OR and_expr)*"""
        left = self._parse_and_expr()
        while self._match_keyword("OR"):
            right = self._parse_and_expr()
            left = f"({left}) | ({right})"
        return left

    def _parse_and_expr(self) -> str:
        """and_expr: not_expr (AND not_expr)*"""
        left = self._parse_not_expr()
        while self._match_keyword("AND"):
            right = self._parse_not_expr()
            left = f"({left}) & ({right})"
        return left

    def _parse_not_expr(self) -> str:
        """not_expr: NOT? comparison"""
        if self._match_keyword("NOT"):
            operand = self._parse_comparison()
            return f"~({operand})"
        return self._parse_comparison()

    def _parse_comparison(self) -> str:
        """comparison: addition ((=|!=|<>|>|<|>=|<=|IN) addition)?"""
        left = self._parse_addition()
        tok = self._peek()
        if tok and tok.type.startswith("OP_"):
            op_map = {
                "OP_EQ": "==",
                "OP_NEQ": "!=",
                "OP_GT": ">",
                "OP_LT": "<",
                "OP_GTE": ">=",
                "OP_LTE": "<=",
            }
            if tok.type in op_map:
                self._advance()
                right = self._parse_addition()
                return f"{left} {op_map[tok.type]} {right}"
        # Handle IN keyword
        if tok and tok.type == "KEYWORD" and tok.value.upper() == "IN":
            self._advance()
            self._expect("LPAREN")
            values = [self._parse_addition()]
            while self._peek() and self._peek().type == "COMMA":
                self._advance()
                values.append(self._parse_addition())
            self._expect("RPAREN")
            return f"{left}.isin([{', '.join(values)}])"
        return left

    def _parse_addition(self) -> str:
        """addition: multiplication ((+|-) multiplication)*"""
        left = self._parse_multiplication()
        while True:
            tok = self._peek()
            if tok and tok.type == "OP_PLUS":
                self._advance()
                right = self._parse_multiplication()
                left = f"{left} + {right}"
            elif tok and tok.type == "OP_MINUS":
                self._advance()
                right = self._parse_multiplication()
                left = f"{left} - {right}"
            else:
                break
        return left

    def _parse_multiplication(self) -> str:
        """multiplication: unary ((*|/) unary)*"""
        left = self._parse_unary()
        while True:
            tok = self._peek()
            if tok and tok.type == "OP_MUL":
                self._advance()
                right = self._parse_unary()
                left = f"{left} * {right}"
            elif tok and tok.type == "OP_DIV":
                self._advance()
                right = self._parse_unary()
                left = f"{left} / {right}"
            else:
                break
        return left

    def _parse_unary(self) -> str:
        """unary: -atom | atom"""
        tok = self._peek()
        if tok and tok.type == "OP_MINUS":
            self._advance()
            operand = self._parse_atom()
            return f"-{operand}"
        return self._parse_atom()

    def _parse_atom(self) -> str:
        """atom: field_ref | string | number | function_call | if_expr | iif_expr | (expr) | keyword"""
        tok = self._peek()
        if tok is None:
            return ""

        # Field reference: [FieldName]
        if tok.type == "FIELD_REF":
            self._advance()
            field_name = tok.value[1:-1]  # Strip brackets
            return f'F.col("{field_name}")'

        # String literal
        if tok.type == "STRING":
            self._advance()
            # Convert to F.lit("...")
            inner = tok.value[1:-1]  # Strip quotes
            return f'F.lit("{inner}")'

        # Number literal
        if tok.type == "NUMBER":
            self._advance()
            return f"F.lit({tok.value})"

        # Parenthesized expression
        if tok.type == "LPAREN":
            self._advance()
            expr = self._parse_or_expr()
            self._expect("RPAREN")
            return f"({expr})"

        # Keywords
        if tok.type == "KEYWORD":
            kw = tok.value.upper()

            if kw == "IF":
                return self._parse_if_expr()

            if kw == "IIF":
                return self._parse_iif_expr()

            if kw in ("TRUE", "FALSE"):
                self._advance()
                return f"F.lit({kw == 'TRUE'})"

            if kw == "NULL":
                self._advance()
                return "F.lit(None)"

            # Fall through to IDENT-like handling
            self._advance()
            return tok.value

        # Identifier (function call or bare name)
        if tok.type == "IDENT":
            return self._parse_function_or_ident()

        # Fallback: skip token
        self._advance()
        return tok.value

    def _parse_function_or_ident(self) -> str:
        """Parse a function call like Trim(...) or a bare identifier."""
        tok = self._advance()
        func_name = tok.value

        # Check if followed by '(' => function call
        next_tok = self._peek()
        if next_tok and next_tok.type == "LPAREN":
            self._advance()  # consume '('
            args = self._parse_arg_list()
            self._expect("RPAREN")
            return self._convert_function(func_name, args)

        # Bare identifier - could be a column name without brackets
        return f'F.col("{func_name}")'

    def _parse_arg_list(self) -> list:
        """Parse a comma-separated list of arguments."""
        args = []
        if self._peek() and self._peek().type == "RPAREN":
            return args  # Empty arg list

        args.append(self._parse_or_expr())
        while self._peek() and self._peek().type == "COMMA":
            self._advance()  # consume ','
            args.append(self._parse_or_expr())
        return args

    def _parse_if_expr(self) -> str:
        """
        Parse IF/ELSEIF/ELSE/ENDIF block:
            IF cond THEN value [ELSEIF cond THEN value]* [ELSE value] ENDIF
        """
        self._expect("KEYWORD", "IF")

        parts = []  # List of (condition, value)

        # First IF
        condition = self._parse_or_expr()
        self._expect("KEYWORD", "THEN")
        value = self._parse_or_expr()
        parts.append((condition, value))

        # ELSEIF blocks
        while self._match_keyword("ELSEIF"):
            condition = self._parse_or_expr()
            self._expect("KEYWORD", "THEN")
            value = self._parse_or_expr()
            parts.append((condition, value))

        # ELSE block
        else_value = None
        if self._match_keyword("ELSE"):
            else_value = self._parse_or_expr()

        self._expect("KEYWORD", "ENDIF")

        # Build PySpark when/otherwise chain
        result = f"F.when({parts[0][0]}, {parts[0][1]})"
        for cond, val in parts[1:]:
            result += f".when({cond}, {val})"
        if else_value:
            result += f".otherwise({else_value})"
        else:
            result += ".otherwise(F.lit(None))"

        return result

    def _parse_iif_expr(self) -> str:
        """Parse IIF(condition, true_value, false_value)."""
        self._expect("KEYWORD", "IIF")
        self._expect("LPAREN")

        condition = self._parse_or_expr()
        self._expect("COMMA")
        true_val = self._parse_or_expr()
        self._expect("COMMA")
        false_val = self._parse_or_expr()

        self._expect("RPAREN")
        return f"F.when({condition}, {true_val}).otherwise({false_val})"

    def _convert_function(self, func_name: str, args: list) -> str:
        """Convert an Alteryx function call to PySpark."""
        # Look up in the function map
        if func_name in FUNCTION_MAP:
            template, expected_args = FUNCTION_MAP[func_name]
            if expected_args == 0:
                return template
            # Pad args if needed
            while len(args) < expected_args:
                args.append("F.lit(None)")
            return template.format(*args[:expected_args])

        # Special handling for functions not in the map
        # Generic fallback: treat as a PySpark function
        arg_str = ", ".join(args)
        return f"F.{func_name}({arg_str})"


def convert_expression(expression: str) -> str:
    """
    Convenience function: convert a single Alteryx expression to PySpark.

    Args:
        expression: Alteryx formula expression string.

    Returns:
        PySpark expression string.
    """
    parser = ExpressionParser()
    return parser.parse(expression)


def convert_filter_expression(expression: str) -> str:
    """
    Convert an Alteryx filter expression to a PySpark filter condition.
    Same as convert_expression but with context that this is a filter.
    """
    return convert_expression(expression)


def convert_formula_expression(expression: str, target_field: str) -> str:
    """
    Convert an Alteryx formula expression for use in withColumn.

    Args:
        expression: The Alteryx formula expression.
        target_field: The target column name.

    Returns:
        A complete .withColumn(...) code snippet.
    """
    pyspark_expr = convert_expression(expression)
    return f'.withColumn("{target_field}", {pyspark_expr})'
