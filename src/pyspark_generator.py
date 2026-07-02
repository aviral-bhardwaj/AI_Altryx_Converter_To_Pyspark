"""
Backwards-compatibility shim.

The deterministic generator was split into:
- ``src/tools/``            — plugin-style tool converter registry
- ``src/converter_engine.py`` — YAML-driven conversion engine

This module re-exports the original public API so existing imports
(`from src.pyspark_generator import ...`) keep working.
"""

from .converter_engine import ConverterEngine, ConversionResult, PySparkCodeGenerator, ToolStatus
from .tools import (
    CONVERTER_REGISTRY,
    GeneratorContext,
    ToolConverter,
    get_converter,
    topological_sort,
)
from .tools.builtin import (
    AppendFieldsConverter,
    BrowseConverter,
    CrossTabConverter,
    DateTimeConverter,
    DynamicRenameConverter,
    FilterConverter,
    FindReplaceConverter,
    FormulaConverter,
    GenerateRowsConverter,
    InputDataConverter,
    JoinConverter,
    MultiFieldFormulaConverter,
    MultiRowFormulaConverter,
    OutputDataConverter,
    PassthroughConverter,
    RecordIDConverter,
    RegExConverter,
    RunningTotalConverter,
    SampleConverter,
    SelectConverter,
    SortConverter,
    SummarizeConverter,
    TextInputConverter,
    TextToColumnsConverter,
    TransposeConverter,
    UnionConverter,
    UniqueConverter,
)

__all__ = [
    "PySparkCodeGenerator",
    "ConverterEngine",
    "ConversionResult",
    "ToolStatus",
    "GeneratorContext",
    "ToolConverter",
    "topological_sort",
    "get_converter",
    "CONVERTER_REGISTRY",
    "InputDataConverter",
    "TextInputConverter",
    "OutputDataConverter",
    "FilterConverter",
    "FormulaConverter",
    "SelectConverter",
    "JoinConverter",
    "UnionConverter",
    "SummarizeConverter",
    "SortConverter",
    "UniqueConverter",
    "SampleConverter",
    "CrossTabConverter",
    "TransposeConverter",
    "MultiRowFormulaConverter",
    "RegExConverter",
    "RecordIDConverter",
    "AppendFieldsConverter",
    "RunningTotalConverter",
    "BrowseConverter",
    "TextToColumnsConverter",
    "DateTimeConverter",
    "DynamicRenameConverter",
    "GenerateRowsConverter",
    "MultiFieldFormulaConverter",
    "FindReplaceConverter",
    "PassthroughConverter",
]
