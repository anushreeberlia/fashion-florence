# Fashion Florence

Standalone training workspace for Florence-2 fashion tagging.

## Current scope

- Build training data that maps iMaterialist labels into Fashion Florence output tags.
- Keep training schema and mapping logic local to this repo.
- Avoid runtime dependencies on `loom`.

## Training modules

- `training/schema.py`: source of truth for allowed training output values.
- `training/label_mapping.py`: iMaterialist label ID to training target conversion.

## Quick sanity run

From repo root:

```bash
python3 -m training.label_mapping
```