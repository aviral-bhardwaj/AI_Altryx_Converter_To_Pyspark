"""
Tests for the Alteryx expression parser.
"""

import pytest
from src.expression_parser import (
    ExpressionParser,
    ExpressionLexer,
    convert_expression,
    convert_filter_expression,
    convert_formula_expression,
    convert_date_format,
    Token,
)


class TestExpressionLexer:
    """Tests for the lexer/tokenizer."""

    def setup_method(self):
        self.lexer = ExpressionLexer()

    def test_tokenize_field_reference(self):
        tokens = self.lexer.tokenize("[FieldName]")
        assert len(tokens) == 1
        assert tokens[0].type == "FIELD_REF"
        assert tokens[0].value == "[FieldName]"

    def test_tokenize_string_literal(self):
        tokens = self.lexer.tokenize("'hello world'")
        assert len(tokens) == 1
        assert tokens[0].type == "STRING"

    def test_tokenize_number(self):
        tokens = self.lexer.tokenize("42")
        assert len(tokens) == 1
        assert tokens[0].type == "NUMBER"
        assert tokens[0].value == "42"

    def test_tokenize_decimal_number(self):
        tokens = self.lexer.tokenize("3.14")
        assert len(tokens) == 1
        assert tokens[0].type == "NUMBER"
        assert tokens[0].value == "3.14"

    def test_tokenize_operators(self):
        tokens = self.lexer.tokenize("+ - * / = != > < >= <=")
        types = [t.type for t in tokens]
        assert "OP_PLUS" in types
        assert "OP_MINUS" in types
        assert "OP_MUL" in types
        assert "OP_DIV" in types
        assert "OP_EQ" in types
        assert "OP_NEQ" in types
        assert "OP_GT" in types
        assert "OP_LT" in types
        assert "OP_GTE" in types
        assert "OP_LTE" in types

    def test_tokenize_keywords(self):
        tokens = self.lexer.tokenize("IF THEN ELSE ENDIF AND OR NOT")
        assert all(t.type == "KEYWORD" for t in tokens)

    def test_tokenize_function_call(self):
        tokens = self.lexer.tokenize("Trim([Name])")
        assert tokens[0].type == "IDENT"
        assert tokens[0].value == "Trim"
        assert tokens[1].type == "LPAREN"
        assert tokens[2].type == "FIELD_REF"
        assert tokens[3].type == "RPAREN"

    def test_tokenize_complex_expression(self):
        tokens = self.lexer.tokenize('[Status] = "Active" AND [Score] > 10')
        assert len(tokens) > 0
        types = [t.type for t in tokens]
        assert "FIELD_REF" in types
        assert "KEYWORD" in types  # AND

    def test_tokenize_whitespace_ignored(self):
        tokens = self.lexer.tokenize("  [A]  +  [B]  ")
        assert all(t.type != "WHITESPACE" for t in tokens)

    def test_tokenize_double_quoted_string(self):
        tokens = self.lexer.tokenize('"hello"')
        assert len(tokens) == 1
        assert tokens[0].type == "STRING"


class TestExpressionParser:
    """Tests for the expression parser."""

    def setup_method(self):
        self.parser = ExpressionParser()

    def test_parse_empty(self):
        assert self.parser.parse("") == ""
        assert self.parser.parse("   ") == ""

    def test_parse_field_reference(self):
        result = self.parser.parse("[Status]")
        assert 'F.col("Status")' in result

    def test_parse_string_literal(self):
        result = self.parser.parse("'Active'")
        assert 'F.lit("Active")' in result

    def test_parse_number_literal(self):
        result = self.parser.parse("42")
        assert "F.lit(42)" in result

    def test_parse_equality(self):
        result = self.parser.parse('[Status] = "Active"')
        assert "==" in result
        assert 'F.col("Status")' in result
        assert 'F.lit("Active")' in result

    def test_parse_inequality(self):
        result = self.parser.parse("[Score] != 0")
        assert "!=" in result

    def test_parse_greater_than(self):
        result = self.parser.parse("[Score] > 10")
        assert ">" in result
        assert 'F.col("Score")' in result
        assert "F.lit(10)" in result

    def test_parse_less_than(self):
        result = self.parser.parse("[Score] < 5")
        assert "<" in result

    def test_parse_and_operator(self):
        result = self.parser.parse('[Status] = "Active" AND [Score] > 5')
        assert "&" in result

    def test_parse_or_operator(self):
        result = self.parser.parse('[Status] = "A" OR [Status] = "B"')
        assert "|" in result

    def test_parse_not_operator(self):
        result = self.parser.parse('NOT [Active]')
        assert "~" in result

    def test_parse_arithmetic_plus(self):
        result = self.parser.parse("[A] + [B]")
        assert "+" in result

    def test_parse_arithmetic_minus(self):
        result = self.parser.parse("[A] - [B]")
        assert "-" in result

    def test_parse_arithmetic_multiply(self):
        result = self.parser.parse("[A] * [B]")
        assert "*" in result

    def test_parse_arithmetic_divide(self):
        result = self.parser.parse("[A] / [B]")
        assert "/" in result

    def test_parse_parenthesized_expression(self):
        result = self.parser.parse("([A] + [B]) * [C]")
        assert "(" in result
        assert ")" in result
        assert "*" in result

    def test_parse_negative_number(self):
        result = self.parser.parse("-5")
        assert "-" in result

    def test_parse_null_keyword(self):
        result = self.parser.parse("NULL")
        assert "F.lit(None)" in result

    def test_parse_true_keyword(self):
        result = self.parser.parse("True")
        assert "F.lit(True)" in result

    def test_parse_false_keyword(self):
        result = self.parser.parse("False")
        assert "F.lit(False)" in result


class TestFunctionConversion:
    """Tests for Alteryx function -> PySpark conversion."""

    def setup_method(self):
        self.parser = ExpressionParser()

    def test_trim(self):
        result = self.parser.parse("Trim([Name])")
        assert "F.trim" in result
        assert 'F.col("Name")' in result

    def test_left(self):
        result = self.parser.parse("Left([Name], 5)")
        assert "F.substring" in result
        assert "1" in result  # Spark is 1-based

    def test_right(self):
        result = self.parser.parse("Right([Name], 3)")
        assert "F.substring" in result

    def test_uppercase(self):
        result = self.parser.parse("UPPERCASE([Name])")
        assert "F.upper" in result

    def test_lowercase(self):
        result = self.parser.parse("LOWERCASE([Name])")
        assert "F.lower" in result

    def test_length(self):
        result = self.parser.parse("Length([Name])")
        assert "F.length" in result

    def test_tostring(self):
        result = self.parser.parse("ToString([ID])")
        assert 'cast("string")' in result

    def test_tonumber(self):
        result = self.parser.parse("ToNumber([Price])")
        assert 'cast("double")' in result

    def test_tointeger(self):
        result = self.parser.parse("ToInteger([Count])")
        assert 'cast("int")' in result

    def test_isnull(self):
        result = self.parser.parse("IsNull([Field])")
        assert "isNull()" in result

    def test_isempty(self):
        result = self.parser.parse("IsEmpty([Field])")
        assert "isNull()" in result
        assert "F.trim" in result

    def test_ifnull(self):
        result = self.parser.parse("IFNULL([Field], 0)")
        assert "F.coalesce" in result

    def test_null_function(self):
        result = self.parser.parse("Null()")
        assert "F.lit(None)" in result

    def test_abs(self):
        result = self.parser.parse("Abs([Value])")
        assert "F.abs" in result

    def test_ceil(self):
        result = self.parser.parse("Ceil([Value])")
        assert "F.ceil" in result

    def test_floor(self):
        result = self.parser.parse("Floor([Value])")
        assert "F.floor" in result

    def test_round(self):
        result = self.parser.parse("Round([Value], 2)")
        assert "F.round" in result

    def test_contains(self):
        result = self.parser.parse("Contains([Name], 'John')")
        assert "contains" in result

    def test_startswith(self):
        result = self.parser.parse("StartsWith([Name], 'Dr')")
        assert "startswith" in result

    def test_findstring(self):
        result = self.parser.parse("FindString([Name], 'x')")
        assert "F.locate" in result
        assert "- 1" in result  # Adjusts for 0-based Alteryx

    def test_padleft(self):
        result = self.parser.parse("PadLeft([Code], 5, '0')")
        assert "F.lpad" in result

    def test_padright(self):
        result = self.parser.parse("PadRight([Code], 5, '0')")
        assert "F.rpad" in result

    def test_getword(self):
        result = self.parser.parse("GetWord([Text], 0)")
        assert "F.split" in result

    def test_regex_match(self):
        result = self.parser.parse("REGEX_Match([Email], '^[a-z]+@')")
        assert "rlike" in result

    def test_regex_replace(self):
        result = self.parser.parse("REGEX_Replace([Text], '[^a-z]', '')")
        assert "F.regexp_replace" in result

    def test_datetime_now(self):
        result = self.parser.parse("DateTimeNow()")
        assert "F.current_timestamp()" in result

    def test_datetime_today(self):
        result = self.parser.parse("DateTimeToday()")
        assert "F.current_date()" in result

    def test_nested_functions(self):
        result = self.parser.parse("Trim(Left([Name], 5))")
        assert "F.trim" in result
        assert "F.substring" in result

    def test_deeply_nested_functions(self):
        result = self.parser.parse("UPPERCASE(Trim([Name]))")
        assert "F.upper" in result
        assert "F.trim" in result


class TestIfExpressions:
    """Tests for IF/ELSEIF/ELSE/ENDIF conversion."""

    def setup_method(self):
        self.parser = ExpressionParser()

    def test_simple_if_else(self):
        result = self.parser.parse('IF [Score] > 8 THEN "High" ELSE "Low" ENDIF')
        assert "F.when" in result
        assert "otherwise" in result
        assert 'F.lit("High")' in result
        assert 'F.lit("Low")' in result

    def test_if_elseif_else(self):
        result = self.parser.parse(
            'IF [Score] > 8 THEN "High" ELSEIF [Score] > 5 THEN "Med" ELSE "Low" ENDIF'
        )
        assert "F.when" in result
        assert ".when(" in result
        assert ".otherwise(" in result

    def test_if_without_else(self):
        result = self.parser.parse('IF [Active] THEN "Yes" ENDIF')
        assert "F.when" in result
        assert "otherwise(F.lit(None))" in result

    def test_iif_function(self):
        result = self.parser.parse('IIF([Score] > 5, "Pass", "Fail")')
        assert "F.when" in result
        assert "otherwise" in result


class TestConvenienceFunctions:
    """Tests for the convenience wrapper functions."""

    def test_convert_expression(self):
        result = convert_expression("[A] + [B]")
        assert "F.col" in result
        assert "+" in result

    def test_convert_filter_expression(self):
        result = convert_filter_expression('[Status] = "Active"')
        assert "==" in result

    def test_convert_formula_expression(self):
        result = convert_formula_expression("Trim([Name])", "clean_name")
        assert '.withColumn("clean_name"' in result
        assert "F.trim" in result


class TestDateFormatConversion:
    """Tests for date format string conversion."""

    def test_basic_date(self):
        assert "yyyy" in convert_date_format("yyyy-MM-dd")
        assert "MM" in convert_date_format("yyyy-MM-dd")
        assert "dd" in convert_date_format("yyyy-MM-dd")

    def test_month_name(self):
        result = convert_date_format("Month dd, yyyy")
        assert "MMMM" in result

    def test_abbreviated_month(self):
        result = convert_date_format("Mon dd, yyyy")
        assert "MMM" in result

    def test_day_name(self):
        result = convert_date_format("Day")
        assert "EEEE" in result

    def test_abbreviated_day(self):
        result = convert_date_format("Dy")
        assert "EEE" in result


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def setup_method(self):
        self.parser = ExpressionParser()

    def test_empty_string(self):
        assert self.parser.parse("") == ""

    def test_whitespace_only(self):
        assert self.parser.parse("   ") == ""

    def test_single_field(self):
        result = self.parser.parse("[Name]")
        assert 'F.col("Name")' in result

    def test_field_with_spaces(self):
        result = self.parser.parse("[First Name]")
        assert 'F.col("First Name")' in result

    def test_complex_nested_expression(self):
        """Test a realistic complex expression."""
        expr = 'IF Contains([Name], "Corp") THEN "Corporate" ELSEIF IsNull([Name]) THEN "Unknown" ELSE [Name] ENDIF'
        result = self.parser.parse(expr)
        assert "F.when" in result
        assert "contains" in result
        assert "isNull()" in result

    def test_multiple_and_conditions(self):
        result = self.parser.parse("[A] > 1 AND [B] > 2 AND [C] > 3")
        assert "&" in result

    def test_in_operator(self):
        result = self.parser.parse("[Status] IN ('A', 'B', 'C')")
        assert "isin" in result
