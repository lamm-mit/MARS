# MaterialDB Generation

Builds `data/MARS_Data/MaterialDB_paper/internal_material_database.json` from the markdown files in `data/MARS_Data/MDs_paper/InternalMaterials/`.

Each subdirectory of `InternalMaterials/` is treated as one material. The script calls a local LLM server to extract structured properties from the markdown content, producing a JSON database keyed by stable material ID slugs.

## Requirements

- Python 3.8+
- A running local LLM server (gpt-oss-20b or compatible) at `http://localhost:8081/v1`

```bash
pip install -r requirements.txt
```

## Usage

```bash
python data/MaterialDB_Generation/build_materialdb.py
```

The script reads `config.yaml` from the same directory. All paths in the config are relative to `data/MaterialDB_Generation/`.

To use a custom config:

```bash
python data/MaterialDB_Generation/build_materialdb.py path/to/config.yaml
```

## Configuration

Edit `config.yaml` to change paths or LLM settings:


| Key             | Default                                                         | Description                               |
| --------------- | --------------------------------------------------------------- | ----------------------------------------- |
| `input_root`    | `../MARS_Data/MDs_paper/InternalMaterials`                      | Input markdown directory                  |
| `database_file` | `../MARS_Data/MaterialDB_paper/internal_material_database.json` | Output JSON                               |
| `state_file`    | `processing_state.json`                                         | Resumability state (local, not committed) |
| `llm.base_url`  | `http://localhost:8081/v1`                                      | Local LLM server endpoint                 |
| `llm.model`     | `gpt-oss-20b`                                                   | Model name                                |


## Output format

```json
{
  "supplier-material-slug": {
    "id": "supplier-material-slug",
    "name": "Material Name",
    "supplier": "Supplier Name",
    "extracted": { ... },
    "raw_markdown_text": "...",
    "source_files": ["relative/path/file.mmd"],
    "created_at": "2025-01-15T10:00:00Z",
    "updated_at": "2025-01-15T10:00:00Z"
  }
}
```

## Resumability

The script tracks processing state in `processing_state.json`. On re-runs, unchanged materials are skipped. Delete `processing_state.json` to force a full rebuild.