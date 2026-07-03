"""
Plugin-Style Tool Registry
===========================
Central registry mapping Alteryx tool types to ``ToolConverter`` classes.

Two extension mechanisms, both requiring zero changes to the core engine:

1. **Python plugin**: subclass ``ToolConverter`` anywhere under ``src/tools/``
   and decorate with ``@register("MyToolType")``.
2. **YAML mapping** (``config/tool_mapping.yaml``): declare
   ``tools.<ToolType>.converter: dotted.path.ToClass`` to add or override a
   mapping without editing code — the class is imported lazily.
"""

import importlib
import logging
from typing import Optional

from .base import ToolConverter

logger = logging.getLogger(__name__)

# tool_type -> ToolConverter subclass
_REGISTRY: dict = {}


def register(*tool_types: str):
    """Class decorator registering a ToolConverter for one or more tool types."""
    def decorator(cls):
        cls.tool_types = tuple(tool_types)
        for tt in tool_types:
            _REGISTRY[tt] = cls
        return cls
    return decorator


def registered_tool_types() -> list:
    """All tool types with a registered converter."""
    return sorted(_REGISTRY.keys())


def resolve_converter_class(dotted_path: str):
    """Import a converter class from a dotted path like ``src.tools.builtin.FilterConverter``."""
    module_path, _, class_name = dotted_path.rpartition(".")
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def build_converter_map(tool_mapping: Optional[dict] = None) -> dict:
    """
    Build a ``{tool_type: ToolConverter instance}`` map from the built-in
    registry, optionally overridden/extended by a parsed tool_mapping.yaml.
    """
    converter_map = {tt: cls() for tt, cls in _REGISTRY.items()}

    if tool_mapping:
        for tool_type, spec in (tool_mapping.get("tools") or {}).items():
            dotted = (spec or {}).get("converter")
            if not dotted:
                continue
            try:
                converter_map[tool_type] = resolve_converter_class(dotted)()
            except (ImportError, AttributeError) as exc:
                logger.warning(
                    "tool_mapping.yaml: cannot load converter %r for tool %s (%s); "
                    "falling back to built-in", dotted, tool_type, exc,
                )
    return converter_map


def get_converter(tool_type: str, converter_map: Optional[dict] = None) -> ToolConverter:
    """Get the converter for a tool type, falling back to passthrough."""
    source = converter_map if converter_map is not None else None
    if source and tool_type in source:
        return source[tool_type]
    cls = _REGISTRY.get(tool_type)
    if cls is not None:
        return cls()
    from .builtin import PassthroughConverter
    return PassthroughConverter()
