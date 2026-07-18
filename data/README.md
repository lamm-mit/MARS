# MARS Data & Models

This directory contains all knowledge resources required to run the MARS pipeline: knowledge graphs, vector databases, a material inventory, model weights, and tooling to convert your own PDF documents into the markdown format the pipeline expects. A minimal dummy dataset is included so the pipeline can be run end-to-end out of the box. For best results, replace the dummy data with your own domain-specific documents.

---

## Repository Structure

```
data/
├── README.md
├── download_data.py               # Downloads full KGs from HuggingFace
├── Markdown_Generation/           # PDF → markdown pipeline (see Markdown_Generation/README.md)
│   └── batch_process_pdfs.py
├── ChromaDB_Generation/           # ChromaDB building pipeline (see ChromaDB_Generation/README.md)
├── MaterialDB_Generation/         # MaterialDB building pipeline (see MaterialDB_Generation/README.md)
├── KG_Generation/                 # Knowledge graph building pipeline (see KG_Generation/README.md)
└── MARS_Data/
    ├── KGs/                       # Knowledge graphs — dummy included; full KGs on HuggingFace
    ├── ChromaDBs_paper/           # Vector databases — dummy MaterialDB included; stubs for others
    │   ├── MaterialDB/
    │   │   └── chromaDB_MaterialDB/
    │   ├── PFAS/                  # stub — build from your own data
    │   ├── Patents/               # stub — build from your own data
    │   └── Textbooks/             # stub — build from your own data
    ├── MaterialDB_paper/          # Material inventory — dummy entry included
    │   └── internal_material_database.json
    ├── Models/
    │   └── DeepSeek-OCR/          # OCR inference code; all weights on HuggingFace
    ├── PDFs_paper/                # Example PDF structure
    │   ├── InternalMaterials/     # one dummy example included
    │   ├── PFAS/
    │   ├── Patents/
    │   └── Textbooks/
    └── MDs_paper/                 # Markdown from PDFs — dummy InternalMaterials included
        ├── InternalMaterials/
        ├── PFAS/
        ├── Patents/
        └── Textbooks/
```

---

## Getting Started

**0. Set up environment**

We suggest using a different environment than that created to run the repo for creating data. The reason being that the various requirements files to be run may cause pip dependencies issues with the environment to be used for running the MARS pipelines.

**1. Download knowledge graphs** (optional — dummy KG included)

The three full KGs (~2 GB) are hosted on HuggingFace and not included in the git repository. Download them with:

```bash
pip install huggingface_hub   # if not already installed
python data/download_data.py
```

This places the files in `data/MARS_Data/KGs/` and skips any that already exist. The dummy KG (`Patent_dummy_KG`) is already included and is sufficient to run the pipeline end-to-end without the full data.

**2. Download model weights from HuggingFace** (optional — only needed to run inference or rebuild data)

All model weights are hosted on HuggingFace. Download and place them in `data/MARS_Data/Models/`:

```bash
# From the project HuggingFace collection (https://huggingface.co/collections/lamm-mit/mars-69fb3118ccf8c28d06643c56):
#   gpt-oss-20b-mxfp4.gguf                        → data/MARS_Data/Models/
#   Llama-3.3-70B-Instruct-Q4_K_L.gguf            → data/MARS_Data/Models/
#   lamm-mit/DeepSeek-OCR weights                 → data/MARS_Data/Models/DeepSeek-OCR/
#   lamm-mit/nomic-embed-text-v1.5                → data/MARS_Data/Models/nomic-embed-text-v1.5-standalone/
#   (or set MODEL_PATH = 'lamm-mit/DeepSeek-OCR' in config.py — weights download at runtime)
```

**3. Run the pipeline**

See the main `README.md` for pipeline usage instructions. The dummy data works out of the box with no configuration changes.

---

## Data Resources

### KGs/ — Knowledge Graphs

The full KG files (~2 GB total) are hosted on HuggingFace at `[lamm-mit/MARS-KGs](https://huggingface.co/datasets/lamm-mit/MARS-KGs)`. A dummy Patent KG is included in the repository for end-to-end testing without downloading the full data. Run `python data/download_data.py` to fetch all three KGs.

Three knowledge graphs, each stored as a GraphML file plus a precomputed node-embedding pickle. Embeddings were generated with `nomic-ai/nomic-embed-text-v1.5` and are required for cosine-similarity-based node mapping at runtime.


| File pair                                                            | Source corpus                     | Nodes   | Edges   | GraphML | Embeddings | Used in  |
| -------------------------------------------------------------------- | --------------------------------- | ------- | ------- | ------- | ---------- | -------- |
| `PFAS_4612.graphml` + `PFAS_embeddings_nomic_v1_5.pkl`               | 4,612 PFAS research papers        | 144,335 | 459,248 | 137 MB  | 432 MB     | System 1 |
| `MatProp_62999.graphml` + `MatProp_62999_embeddings_nomic_v1_5.pkl`  | 62,999 material-science abstracts | 317,800 | 774,009 | 229 MB  | 481 MB     | System 2 |
| `Patent_13654_US.graphml` + `Patent_13654_embeddings_nomic_v1_5.pkl` | 13,654 polymer-science patents    | 199,008 | 578,361 | 154 MB  | 594 MB     | System 2 |


All three KGs were built extracting semantic triplets from text chunks using the [GraphReasoning](https://github.com/lamm-mit/GraphReasoning) framework. The PFAS and Material Properties KGs were introduced in prior work; the Patents KG was newly constructed for this study.

---

### ChromaDBs_paper/ — Vector Databases

Four ChromaDB persistent collections for text retrieval across the three pipeline stages. A dummy `MaterialDB` collection built from the included example spec sheet is provided out of the box. The remaining three collections are empty stubs — populate them by adding your own source documents and running `data/ChromaDB_Generation/build_chromadb.py`.


| Directory                        | Source corpus             | Included        | Used in      |
| -------------------------------- | ------------------------- | --------------- | ------------ |
| `MaterialDB/chromaDB_MaterialDB` | Supplier spec sheets      | Dummy (1 entry) | Systems 2, 3 |
| `PFAS/`                          | PFAS research papers      | Stub            | System 1     |
| `Patents/`                       | Polymer-science patents   | Stub            | Systems 2, 3 |
| `Textbooks/`                     | Polymer-science textbooks | Stub            | System 3     |


---

### MaterialDB_paper/ — Material Inventory

`internal_material_database.json` is a structured JSON dictionary of lab-available materials used in System 2 to ground candidate proposals in physically available materials. A dummy entry built from the included example spec sheet is provided.

Each entry contains:


| Field               | Type         | Description                                              |
| ------------------- | ------------ | -------------------------------------------------------- |
| `id`                | string       | Unique material identifier                               |
| `name`              | string       | Material name from the supplier datasheet                |
| `supplier`          | string       | Supplier or manufacturer                                 |
| `raw_markdown_text` | string       | Full machine-readable markdown from the source datasheet |
| `extracted`         | object       | Structured dictionary of extracted material properties   |
| `llm_prompt`        | string       | Prompt used for LLM-based property extraction            |
| `llm_response_raw`  | string       | Raw LLM output                                           |
| `llm_response_json` | object       | Parsed JSON of the LLM extraction output                 |
| `created_at`        | string       | Entry creation timestamp                                 |
| `updated_at`        | string       | Last update timestamp                                    |
| `source_files`      | list[string] | Source markdown file(s) used to build this entry         |


---

### Models/

All model weights are hosted on HuggingFace — nothing is committed to this repository. The `DeepSeek-OCR/` directory contains inference code only.


| Model                                | Format                      | Role                                                       | HuggingFace source                                                                      |
| ------------------------------------ | --------------------------- | ---------------------------------------------------------- | --------------------------------------------------------------------------------------- |
| `nomic-embed-text-v1.5`              | HuggingFace model           | Embedding model for all Chroma queries and KG node mapping | `[lamm-mit/nomic-embed-text-v1.5](https://huggingface.co/lamm-mit/nomic-embed-text-v1.5)`   |
| `DeepSeek-OCR/`                      | Inference code + HF weights | OCR model for converting spec sheet PDFs to markdown       | `[lamm-mit/DeepSeek-OCR](https://huggingface.co/lamm-mit/DeepSeek-OCR)`                     |
| `gpt-oss-20b-mxfp4.gguf`             | GGUF (MXFP4)                | Primary LLM backbone for all MARS agents                   | `[lamm-mit/gpt-oss-20b](https://huggingface.co/lamm-mit/gpt-oss-20b)`                       |
| `Llama-3.3-70B-Instruct-Q4_K_L.gguf` | GGUF (Q4_K_L)               | LLM for knowledge graph construction                       | `[tphage/Llama-3.3-70B-Instruct](https://huggingface.co/tphage/Llama-3.3-70B-Instruct)` |


The two GGUF models are served locally via `llama.cpp`. Download from `[lamm-mit/gpt-oss-20b](https://huggingface.co/lamm-mit/gpt-oss-20b)` and `[tphage/Llama-3.3-70B-Instruct](https://huggingface.co/tphage/Llama-3.3-70B-Instruct)`.

**DeepSeek-OCR model weights** are available on HuggingFace. Set `MODEL_PATH` in `data/MARS_Data/Models/DeepSeek-OCR/DeepSeek-OCR-master/DeepSeek-OCR-vllm/config.py` to one of:

```python
# Option 1 — official DeepSeek HuggingFace (downloads weights at runtime):
MODEL_PATH = 'deepseek-ai/DeepSeek-OCR'

# Option 2 — project HuggingFace repo (exact weights used in the paper):
MODEL_PATH = 'lamm-mit/DeepSeek-OCR'

# Option 3 — local weights (if downloaded):
MODEL_PATH = '/path/to/data/MARS_Data/Models/DeepSeek-OCR'
```

---

### PDFs_paper/ and MDs_paper/

`PDFs_paper/` holds the folder structure for source PDF documents. One dummy example spec sheet is provided in `InternalMaterials/` to illustrate the expected structure. The subfolders for PFAS papers, patents, and textbooks are empty stubs.

`MDs_paper/` holds the corresponding markdown documents converted from PDFs. A dummy `InternalMaterials/` set is included (generated from the dummy spec sheet). The other subdirectories are empty stubs.

To use your own documents, place PDFs in the appropriate `PDFs_paper/` subdirectory and follow the instructions in the sections below to convert them to markdown and build the databases.


| Directory                                      | Contents                      | Included      |
| ---------------------------------------------- | ----------------------------- | ------------- |
| `data/MARS_Data/PDFs_paper/InternalMaterials/` | Supplier spec sheet PDFs      | Dummy example |
| `data/MARS_Data/PDFs_paper/PFAS/`              | PFAS research paper PDFs      | Stub          |
| `data/MARS_Data/PDFs_paper/Patents/`           | Polymer-science patent PDFs   | Stub          |
| `data/MARS_Data/PDFs_paper/Textbooks/`         | Polymer-science textbook PDFs | Stub          |
| `data/MARS_Data/MDs_paper/InternalMaterials/`  | Spec sheet markdown           | Dummy example |
| `data/MARS_Data/MDs_paper/PFAS/`               | PFAS paper markdown           | Stub          |
| `data/MARS_Data/MDs_paper/Patents/`            | Patent markdown               | Stub          |
| `data/MARS_Data/MDs_paper/Textbooks/`          | Textbook markdown             | Stub          |


---

## Converting Supplier Spec Sheet PDFs to Markdown

See `**[Markdown_Generation/README.md](Markdown_Generation/README.md)`** for full setup and usage instructions.

The script `data/Markdown_Generation/batch_process_pdfs.py` converts PDFs in `data/MARS_Data/PDFs_paper/InternalMaterials/` to markdown using [DeepSeek-OCR](https://github.com/deepseek-ai/DeepSeek-OCR). Run from the repo root:

```bash
python data/Markdown_Generation/batch_process_pdfs.py
```

---

## Building Your Own Data

The general workflow for populating the pipeline with your own documents is:

1. Place your PDFs in `data/MARS_Data/PDFs_paper/{corpus}/` (one subfolder per supplier/source)
2. Convert PDFs to markdown (`data/Markdown_Generation/`)
3. Build the MaterialDB from markdown (`data/MaterialDB_Generation/`)
4. Build ChromaDB collections from markdown (`data/ChromaDB_Generation/`)
5. Build knowledge graphs (`data/KG_Generation/`)
6. Point the MARS pipeline config at the updated paths via `--override`

For the PFAS, patent, and textbook corpora, markdown can also be generated with `[marker-pdf](https://github.com/VikParuchuri/marker)`.

---

### MaterialDB Generation

See `**[MaterialDB_Generation/README.md](MaterialDB_Generation/README.md)**` for setup and usage instructions.

`data/MaterialDB_Generation/` contains the script for building `data/MARS_Data/MaterialDB_paper/internal_material_database.json` from markdown spec sheets in `data/MARS_Data/MDs_paper/InternalMaterials/`.

---

### ChromaDB Generation

See `**[ChromaDB_Generation/README.md](ChromaDB_Generation/README.md)**` for setup and usage instructions.

`data/ChromaDB_Generation/` contains the script for building the ChromaDB vector database collections in `data/MARS_Data/ChromaDBs_paper/` from markdown documents in `data/MARS_Data/MDs_paper/`.

---

### Knowledge Graph Generation

See `**[KG_Generation/README.md](KG_Generation/README.md)**` for setup and usage instructions.

`data/KG_Generation/` contains the pipeline for building the knowledge graphs in `data/MARS_Data/KGs/` from source documents using the [GraphReasoning](https://github.com/lamm-mit/GraphReasoning) framework.

The literature search results behind the PFAS and Material Properties KGs — paper metadata, abstracts, and per-search-term counts — are in [`paper_metadata/`](../paper_metadata/README.md) at the repo root.

---

## Experimental Setup

Experiments were run on a single node with 4× NVIDIA Tesla V100S (32 GB VRAM each), Python 3.11, CentOS Linux 7, CUDA 12.6. The LLM backbone was `gpt-oss-20b` (natively quantized in MXFP4), served locally via `llama.cpp`. A fixed random seed (9527) was used for all components that expose a seed parameter; all LLM calls used temperature 0. Due to low-level nondeterminism in LLM inference backends, repeated runs yield similar but not bit-for-bit identical results.

---

## Citation

```bibtex
@misc{hage2026mars,
  title         = {MARS: Hierarchical Multi-Agent Reasoning Systems Enable Knowledge-Grounded Material Substitution},
  author        = {Tarjei Paule Hage and Yu-Chuan Hsu and Wei Lu and Gayla Lyon and Jiezhu Jin and Markus J. Buehler},
  year          = {2026},
  eprint        = {TODO},  % TODO: replace with arXiv eprint ID
  archivePrefix = {arXiv},
  primaryClass  = {cs.AI},
}
```

