# Alteryx → Databricks Converter (Skill Mode)
#
# Modules:
#   parser              - Alteryx .yxmd XML parser (lxml) -> workflow DAG
#   models              - Data models (Workflow, Tool, Container, Connection)
#   converter_engine    - YAML-driven core conversion engine
#   tools/              - Plugin-style tool converter registry
#   expression_parser   - Alteryx expression -> PySpark expression converter
#   databricks_exporter - .ipynb / .py / .dbc notebook generation + batch mode
#   validators          - Schema/row/aggregate + AST structural validation
#   self_correction     - Generate -> validate -> correct loop (max 3 passes)
#   context_builder     - AI prompt context builder (optional AI mode)
#   ai_generator        - Claude AI-powered code generator (optional AI mode)
#   utils               - CLI utilities

__version__ = "1.0.0"
