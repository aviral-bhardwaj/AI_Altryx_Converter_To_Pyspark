# Publishing to Databricks Marketplace

This runbook takes the accelerator from this repo to a live listing on
[Databricks Marketplace](https://marketplace.databricks.com). Values to paste
into the Provider Console live in [`listing.yaml`](listing.yaml).

## 1. Become a Marketplace provider

1. You need a Databricks account on **Unity Catalog-enabled** workspaces and
   the **Marketplace admin** entitlement (account admin grants this in the
   account console).
2. Apply via **Partner Portal → Become a Marketplace provider** (or ask your
   Databricks account team). Legal onboarding requires:
   - Provider display name and logo (512×512 PNG)
   - Company description
   - Support contact (email or URL)
   - Privacy policy URL and terms of service
3. Once approved, the **Provider Console** appears under
   **Marketplace → Provider console** in your workspace.

## 2. Prepare the assets

Marketplace listings for solution accelerators are **notebook-based**; data
products are shared via Delta Sharing. This accelerator ships everything you
need:

| Asset | Where |
|---|---|
| Skill Mode notebook | `notebooks/01_Skill_Mode_Alteryx_to_Databricks.ipynb` |
| Library code (synced by the bundle) | `src/`, `config/` |
| Sample workflows for the demo | `tests/sample_workflows/*.yxmd` |
| License / terms | `LICENSE` (MIT) |
| Long-form listing copy | `marketplace/listing.yaml` → `description` |

Before submitting:

```bash
pytest tests/ -q                      # 271 tests must pass
databricks bundle validate -t dev     # bundle must validate cleanly
python convert.py tests/sample_workflows --batch --self-correct \
    --format ipynb --output-dir ./demo_output
```

Import `demo_output/*.ipynb` into a clean workspace and Run All — this is
exactly what a Marketplace reviewer (and your first customer) will do.

Add the media files referenced by `listing.yaml`:

- `marketplace/assets/icon.png` — 512×512 product icon
- Screenshots of §3 (DAG visualization) and §6 (self-correction report)
  from the Skill Mode notebook

## 3. Create the listing

In **Provider console → Listings → Create listing**:

1. **Listing type**: choose the notebook/solution-accelerator flow (not a
   Delta Sharing data product).
2. Copy `title`, `subtitle`, `description`, `categories` from `listing.yaml`.
3. **Assets**: attach the Skill Mode notebook. Recommended: also link the Git
   repo (customers clone it as a Git folder and get the full bundle — the
   notebook auto-detects the bundle root either way).
4. **Access**: `Instantly available` (free) is the fastest path to publish;
   switch to `Request access` if you want lead capture.
5. Attach the license/terms (MIT) and support contact.
6. Submit for review. Databricks reviews new listings before they go public —
   typical review checks: the notebook runs top-to-bottom on a fresh cluster
   (DBR 13.3+), no hardcoded credentials/hosts, description matches behavior.

## 4. Versioning & updates

- Bump `version` in `pyproject.toml` and `src/__init__.py`, update
  `CHANGELOG.md`, tag the release in Git.
- Re-attach the refreshed notebook (or let the linked Git repo carry the
  update) and edit the listing in the Provider Console; edits go through the
  same lightweight review.
- Keep `main` deployable at all times — `.github/workflows/bundle-deploy.yml`
  gates every push with the test suite + `databricks bundle validate`.

## 5. Review checklist (pre-submission)

- [ ] `pytest` green and CI passing on `main`
- [ ] `databricks bundle validate` passes for `dev`, `staging`, `prod`
- [ ] Skill Mode notebook runs end-to-end on DBR 13.3+ with default widgets
      (uses the bundled sample workflows — no external data needed)
- [ ] No secrets, tokens, or workspace hostnames anywhere in the repo
- [ ] LICENSE, support contact, and privacy policy URL current
- [ ] Icon + at least 2 screenshots uploaded
- [ ] Listing description spell-checked and matches actual capabilities
