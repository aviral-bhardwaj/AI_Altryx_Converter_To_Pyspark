"""
Backwards-compatibility shim: the validation framework now lives in
``src/validators.py`` (row/schema/aggregate validation plus the AST-based
structural validator used by the self-correction loop).
"""

from .validators import *  # noqa: F401,F403
