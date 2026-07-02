# Changelog

## [1.0.0] — 2026-07-02

### Added — "Skill Mode" release
- **Single Skill Mode notebook** (`notebooks/01_Skill_Mode_Alteryx_to_Databricks.ipynb`)
  replacing the previous fragmented per-stage notebooks: end-to-end
  upload → parse → visualize DAG → convert → self-correct → export → bundle deploy.
- **Plugin-style tool registry** (`src/tools/`): new Alteryx tools are added as
  Python classes with `@register("ToolType")` — no core changes required.
- **YAML-driven engine** (`config/tool_mapping.yaml` + `src/converter_engine.py`)
  with per-tool conversion status tracking and graceful fallbacks for
  unsupported tools.
- **Self-correcting validation loop** (`src/self_correction.py`): AST-based
  structural validation of generated PySpark against the original `.yxmd` DAG,
  strict-mode regeneration on FAILED, optimization rules (broadcast/Delta/ZORDER)
  on WARNING, up to 3 iterations, validation report embedded as the notebook's
  top `%md` cell.
- **Databricks exporter** (`src/databricks_exporter.py`): `.ipynb` (nbformat),
  Databricks source `.py`, and `.dbc` archive output, plus recursive batch mode
  mirroring folder structures.
- **Databricks Asset Bundle** packaging: root `databricks.yml` with
  `dev`/`staging`/`prod` targets, `resources/alteryx_migration_job.yml`
  multi-task migration workflow, and a validation gate (`src/job_validate.py`).
- **CI/CD**: `.github/workflows/bundle-deploy.yml` — tests + `bundle validate`
  on every push, deploy on `main`/manual dispatch.
- **Marketplace readiness**: `marketplace/` listing metadata and publishing
  guide, MIT license, `pyproject.toml` packaging, sample workflows under
  `tests/sample_workflows/`.

### Changed
- Parser now prefers `lxml` (stdlib `xml.etree` fallback).
- Input converters emit escape-safe string literals and flag local
  `.yxdb`/Windows-path sources as Unity Catalog mapping TODOs instead of
  generating broken code.
- `convert.py` gained `--self-correct`, `--format py|ipynb|dbc`, and `--batch`.

### Compatibility
- `src.pyspark_generator` and `src.validation` remain as import shims; the
  full 271-test suite passes unchanged.
