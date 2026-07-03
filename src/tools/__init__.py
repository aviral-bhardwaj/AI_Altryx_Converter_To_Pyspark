"""
src.tools — plugin-style Alteryx tool converter registry.

Import order matters: importing this package loads the built-in converters,
which self-register via the @register decorator.
"""

from .base import GeneratorContext, ToolConverter, topological_sort
from .registry import (
    build_converter_map,
    get_converter,
    register,
    registered_tool_types,
    resolve_converter_class,
)
from .builtin import *  # noqa: F401,F403 — self-registers all built-in converters
from . import builtin as _builtin

# Backwards-compatible instance registry (used by legacy code/tests).
CONVERTER_REGISTRY: dict = build_converter_map()

__all__ = [
    "ToolConverter",
    "GeneratorContext",
    "topological_sort",
    "register",
    "get_converter",
    "registered_tool_types",
    "resolve_converter_class",
    "build_converter_map",
    "CONVERTER_REGISTRY",
] + [name for name in dir(_builtin) if name.endswith("Converter")]
