# ChromaDB Generation

Builds ChromaDB persistent collections from directories of markdown files.

Each subdirectory of the configured input root is treated as a **source group**. All `.md` and `.mmd` files within it are chunked and embedded using `nomic-ai/nomic-embed-text-v1.5`, then stored in a ChromaDB collection.

The included `config.yaml` targets the MaterialDB collection (spec sheets in `data/MARS_Data/MDs_paper/InternalMaterials/`). Any other corpus can be built by writing a new config file and passing it as an argument.

## Requirements

- Python 3.8+

```bash
pip install -r requirements.txt
```

## Usage

```bash
# Build with the default config (MaterialDB)
python data/ChromaDB_Generation/build_chromadb.py

# Build with a custom config
python data/ChromaDB_Generation/build_chromadb.py path/to/config.yaml
```

## Configuration

| Key | Description |
|---|---|
| `input_root` | Directory containing source-group subdirectories (path relative to this file) |
| `output_dir` | ChromaDB persistence directory (path relative to this file) |
| `collection_name` | ChromaDB collection name |
| `chunk_size` | Target chunk size in tokens (default: `500`) |
| `chunk_overlap` | Overlap between chunks in tokens (default: `50`) |
| `embeddings.model_name` | Sentence-transformers model |
| `embeddings.device` | `auto` picks CUDA if available, else CPU; or set `cuda:0` / `cpu` explicitly |
| `embeddings.batch_size` | Embedding batch size — reduce if OOM (default: `16`) |

## Expected input structure

```
{input_root}/
└── {SourceGroup}/
    ├── file_a.mmd
    └── file_b.md
```

## Metadata stored per chunk

| Field | Description |
|---|---|
| `source_group` | Subdirectory name (e.g. `PEEK`) |
| `source_file` | Filename |
| `chunk_index` | Index of this chunk within the file |
