# KG Generation

Builds knowledge graphs for the MARS pipeline from a corpus of markdown files.

For each document, an LLM extracts a per-document knowledge graph (nodes + edges as structured JSON). The individual graphs are then incrementally merged into a single global graph, with similar nodes (cosine similarity ≥ 0.9) deduplicated using `nomic-ai/nomic-embed-text-v1.5` embeddings.

Adapted from `make_KG_PFAS.ipynb` in [GraphAgents.](https://github.com/lamm-mit/GraphAgents)

## Requirements

- Python 3.8+
- A running local LLM server at `http://localhost:8081/v1`
- [GraphReasoning](https://github.com/lamm-mit/GraphReasoning) installed separately

```bash
pip install -r requirements.txt
pip install git+https://github.com/lamm-mit/GraphReasoning
```

## Usage

```bash
python data/KG_Generation/build_kg.py                      # uses config.yaml
python data/KG_Generation/build_kg.py path/to/config.yaml  # custom config
```

## Configuration

The included `config.yaml` targets the Patents corpus. Copy and edit it for other corpora (PFAS, Textbooks).


| Key                          | Description                                                    |
| ---------------------------- | -------------------------------------------------------------- |
| `corpus`                     | Label for this corpus — used in log messages                   |
| `paths.input_dir`            | Markdown corpus (see [Input structure](#input-structure))      |
| `paths.graph_dir`            | Per-document `.graphml` files (intermediate, not committed)    |
| `paths.merged_dir`           | Checkpoint merged graphs (intermediate, not committed)         |
| `paths.output_dir`           | Final graph and embeddings output                              |
| `paths.graph_output_file`    | Filename for the final merged `.graphml`                       |
| `paths.embedding_file`       | Filename for the final node embeddings `.pkl`                  |
| `llm.provider`               | `local` (llama.cpp) or `openai` (OpenAI API)                   |
| `llm.base_url`               | Local LLM server endpoint — ignored when `provider: openai`    |
| `llm.model_name`             | Model name (e.g. `gpt-oss-20b` for local, `gpt-4o` for OpenAI) |
| `llm.max_tokens`             | Max tokens per LLM call                                        |
| `llm.temperature`            | Sampling temperature                                           |
| `embeddings.model_name`      | Sentence-transformers model for node embeddings                |
| `embeddings.device`          | `auto`, `cuda:0`, or `cpu`                                     |
| `kg.chunk_size`              | Max characters per text chunk sent to LLM                      |
| `kg.similarity_threshold`    | Cosine similarity above which two nodes are merged             |
| `kg.simplify_every_n`        | Run graph simplification every N documents                     |
| `kg.simplify_size_threshold` | Remove disconnected components smaller than this               |


## Relation to original notebook

`build_kg.py` is adapted from `make_KG_PFAS.ipynb`, a notebook from the [GraphAgents](https://github.com/lamm-mit/GraphAgents) repository (LAMM-MIT, Stewart et al. 2026). The core algorithm — LLM-based KG extraction via `make_graph_from_text`, incremental merging via `add_new_subgraph_from_text`, and nomic embedding of nodes — is unchanged. The following differences apply:


| Aspect                       | Notebook                                                                                      | `build_kg.py`                                                                                                                                                                                                                                        |
| ---------------------------- | --------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **LLM provider**             | Together API (Llama 4 Maverick)                                                               | Local llama.cpp or OpenAI API, switchable via config                                                                                                                                                                                                 |
| **Structured output**        | `instructor` + pydantic                                                                       | OpenAI JSON mode + pydantic (no `instructor`)                                                                                                                                                                                                        |
| **Multi-threading**          | Supported (`sys.argv` thread index/total)                                                     | Single-threaded for reproducibility                                                                                                                                                                                                                  |
| **Configuration**            | Hardcoded paths and parameters                                                                | `config.yaml`                                                                                                                                                                                                                                        |
| **Corpus**                   | PFAS papers only                                                                              | Any corpus via config                                                                                                                                                                                                                                |
| **Markdown layout**          | Only `{doc_data_dir}/{doc_id}/{doc_id}.md` (one subfolder per paper, filename matches folder) | Same nested layout **or** flat `input_dir/*.md` files; `collect_documents()` merges both (deduplicated)                                                                                                                                              |
| **GraphReasoning / Louvain** | Relies on whatever version you had when the notebook last ran                                 | Defines `colors2Community` on `GraphReasoning.graph_tools` at import if missing — current PyPI GraphReasoning calls it from `graph_Louvain` but leaves the helper commented out, which would raise `NameError` during merge when simplification runs |
| **Checkpoint finding**       | Fragile `split('_')[2]` path parsing                                                          | Regex on filename                                                                                                                                                                                                                                    |
| **Image nodes**              | Vision LLM call                                                                               | Stub returning empty graph (not used)                                                                                                                                                                                                                |


## Resumability

The script checkpoints after every document:

- If a per-document `.graphml` already exists in `graph_dir/`, that document is skipped.
- If merged checkpoints exist in `merged_dir/`, the script resumes from the last valid one.

Intermediate files in `working/` are excluded from git (covered by `.gitignore`).

## Input structure

Use **either** layout (or both); each path is one document. The stem of the file (e.g. `foo` from `foo.md`) is used as the document title in the pipeline.

**Flat (only in `build_kg.py`):** drop `.md` files directly under `input_dir`:

```
{input_dir}/
├── first_patent.md
└── second_patent.md
```

**Nested (notebook and `build_kg.py`):** one folder per document, markdown name matches the folder name (same as `make_KG_PFAS.ipynb` `doc_list` construction):

```
{input_dir}/
└── {doc_id}/
    └── {doc_id}.md
```

## Inspecting a graph

`[show_kg.ipynb](show_kg.ipynb)` loads a built `.graphml` and provides:

- **Summary** — node/edge counts, connected components, node type distribution, top edge relations
- **Example paths** — randomly sampled multi-hop paths printed as `(node)-[relation]-(node)-[relation]-(node)`
- **Visualisation** — spring-layout plot rendered inline; node labels shown when the graph is small enough

By default it reads the graph path from `config.yaml`. Set `GRAPH_FILE` in the Config cell to point at any `.graphml` directly.

The notebook also documents the included dummy graph `Patent_dummy_KG.graphml`: it was built from `data/MARS_Data/MDs_paper/Patents/dummy_example_material_specification_sheet.md` using `build_kg.py`, and serves as a minimal end-to-end example of the same process used to produce the three real KGs in the paper.

`[show_kg.py](show_kg.py)` provides the same summary and optional PNG export as a command-line tool:

```bash
python data/KG_Generation/show_kg.py                                         # reads from config.yaml
python data/KG_Generation/show_kg.py path/to/graph.graphml --plot kg.png
```

## Output

```
data/MARS_Data/KGs/
├── {graph_output_file}     # final merged GraphML
└── {embedding_file}        # node embeddings (.pkl)
```

