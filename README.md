# MARS: Hierarchical Multi-Agent Reasoning Systems Enable Knowledge-Grounded Material Substitution

**Tarjei Paule Hage · Yu-Chuan Hsu · Wei Lu · Gayla Lyon · Jiezhu Jin · Markus J. Buehler** — Massachusetts Institute of Technology

MARS is a three-system LLM pipeline for knowledge-grounded material substitution. Given a query, System 1 extracts required material properties from a domain knowledge graph and RAG corpus. System 2 proposes a candidate substitute by reasoning over two knowledge graphs and two retrieval corpora. System 3 assesses lab-scale manufacturability against three additional corpora, feeding blocking constraints back to System 2 if the candidate fails and looping until a viable substitute is found or the iteration limit is reached.

<img width="1203" height="301" alt="image" src="https://github.com/user-attachments/assets/7dd562db-7763-4732-b97f-564051f92c6f" />

MARS system overview

---

## Supplementary Information

This repository serves as both a reproduction package and the supplementary material for the paper. All code, configuration, prompts, frozen results, and intermediate artifacts from the THV substitution case study are included here. For a detailed account of what this repository contains beyond what appears in the paper — including extended pipeline artifacts, full chat logs, and the evaluation breakdown — see **[SI.md](SI.md)**.

---

## Quick Start

> **Requirements:** Linux, NVIDIA GPU (CUDA 12.x), conda or mamba, an OpenAI API key.

```bash
git clone https://github.com/LAMM-MIT/MARS && cd MARS
conda env create -f environment.yml && conda activate MARS
pip install git+https://github.com/lamm-mit/GraphReasoning.git
export OPENAI_API_KEY="sk-..."
./run_experiments.sh -a -e
```

Results are written to `results/Query1/`. The frozen paper outputs are in `results_from_paper/`.

---

## Installation

**Conda environment** — `environment.yml` installs all Python dependencies with exact version pins, including PyTorch with the appropriate CUDA build for your driver. Create and activate it:

```bash
conda env create -f environment.yml
conda activate MARS
```

**GraphReasoning** — installed separately because it is not on PyPI and has its own dependency chain:

```bash
pip install git+https://github.com/lamm-mit/GraphReasoning.git
```

**LLM backend** — the default backend for this repository is the OpenAI API (`gpt-5-nano`), which requires an API key and allows anyone to run the pipeline without local infrastructure. The paper itself used `gpt-oss-20b` served locally via llama.cpp, and the repository fully supports locally-hosted models as well. Set your OpenAI key to use the default:

```bash
export OPENAI_API_KEY="sk-..."
```

To switch to a locally-hosted model, see [config/README.md](config/README.md).

**First run** — on the first call to `initialize()`, the embedding model (`nomic-ai/nomic-embed-text-v1.5`, ~270 MB) is downloaded automatically from HuggingFace.

---

## Data

**Dummy data** is included at `data/MARS_Data/` and loads out of the box — no download or configuration needed for an initial run. However, with dummy data MARS will not produce scientifically meaningful results. Meaningful outputs require the full knowledge graphs and ChromaDB retrieval corpora (see below).

**Full knowledge graphs** (~2 GB, three KG pairs) are hosted on HuggingFace at [lamm-mit/MARS-KGs](https://huggingface.co/datasets/lamm-mit/MARS-KGs). Download them and point the pipeline at them:

```bash
python data/download_data.py
python scripts/run_mars.py --override config/overrides/downloaded_KGs.yaml
```

**ChromaDB vector databases** — four RAG corpora (PFAS literature, patent corpus, material database, manufacturing textbooks) are used in the paper. The pipeline can be run with user-constructed ChromaDBs; `data/README.md` documents the full build pipeline. The downloaded knowledge graphs alone are not necessarily enough for producing comparable results.

**Paper metadata** — the literature search results behind the PFAS and Material-Properties knowledge graphs and RAG corpora — paper titles, authors, DOIs, journals, and abstracts (no full text) — are in [`paper_metadata/`](paper_metadata/README.md).

---

## Reproducing the Results

### Running the pipeline

The primary entry point runs the full MARS pipeline, all ablation conditions, and the LLM-as-judge evaluation in sequence:

```bash
./run_experiments.sh -a -e
```

The three stages can also be run individually:

```bash
python scripts/run_mars.py --queries Query1       # System 1 → System 2 ↔ System 3
python scripts/run_ablations.py --queries Query1  # 3-agent, 1-agent+RAG, 1-agent, 1-agent-GPT-5.4
python scripts/run_evaluation.py --queries Query1 # LLM-as-judge blind evaluation
```

### Ablation conditions


| Condition                        | Description                                                 |
| -------------------------------- | ----------------------------------------------------------- |
| **MARS (full pipeline)**         | System 1 → System 2 ↔ System 3 with RAG + dual-KG reasoning |
| **3-agent**                      | 3 sequential LLM calls, no RAG or KG                        |
| **1-agent + RAG/KG**             | Single LLM call with pre-retrieved RAG + KG context         |
| **1-agent (no RAG/KG)**          | Single LLM call, purely parametric                          |
| **1-agent (no RAG/KG, GPT-5.4)** | Same as above using a GPT-5.4 backend                       |


### LLM-judge evaluation

`run_evaluation.py` randomises system labels (A–E) for blind scoring across 12 subsystem criteria on a 1–5 ordinal scale. See `config/evaluation_rubric.yaml` for the full rubric and choice of judge model.

### Important notes on reproduction

**Data requirements.** The frozen results in `results_from_paper/` were produced using the full ChromaDB retrieval corpora. Users who build equivalent ChromaDBs from their own document collections (see `data/README.md`) can run the full pipeline under the same conditions.

**MARS candidate convergence.** MARS is iterative and may not find a manufacturable candidate within the configured maximum number of System 2 / System 2↔3 iterations. Before running the LLM-judge evaluation, verify that your MARS run produced an actual candidate by checking `results/Query1/mars.json`. Evaluating against a failed run will produce misleading scores.

For information on switching LLM backends, using override files, or pointing the pipeline at data stored on your own drives, see **[config/README.md](config/README.md)**.

---

## Frozen Results

`results_from_paper/` contains the exact outputs used in the paper:


| Path                                | Contents                                                                                            |
| ----------------------------------- | --------------------------------------------------------------------------------------------------- |
| `Query1/mars.json`                  | Final MARS pipeline output (candidate, properties, constraints)                                     |
| `Query1/ablation_*.json`            | Outputs from all four ablation conditions                                                           |
| `Query1/artifacts/`                 | Full intermediate artifacts: per-system outputs, agent chat logs, KG subgraphs, rejected candidates |
| `evaluation/eval_Query1.json`       | Per-query LLM-judge scores across all 12 criteria                                                   |
| `evaluation/aggregate_results.json` | Aggregate rankings across all systems                                                               |


For a description of how these files relate to specific figures and tables in the paper, see **[SI.md](SI.md)**.

---

## Notebooks

- `notebooks/walkthrough.ipynb` — interactive demo that loads a pre-computed result and visualises the full pipeline run: execution timeline, agent chat logs, KG subgraph, and per-system outputs.
- `notebooks/graph_viz.ipynb` — visualises the Material Informed Subgraph produced during System 2, at multiple levels of detail from ego-graphs around individual nodes to the full graph topology.
- `notebooks/mars_showcase_detailed.ipynb` — renders any MARS run end-to-end for a chosen query: requirements, the closed-loop System 2⇄3 search, the knowledge subgraph, the manufacturing route, agent reasoning traces, and (where available) the blind evaluation.

---

## Repository Structure

```
├── config/                      # Configuration — see config/README.md
│   ├── config.yaml              # Base config: LLM, embeddings, data paths, hyperparameters
│   ├── prompts.yaml             # All LLM system and user prompts
│   ├── queries.yaml             # Benchmark query definitions
│   ├── evaluation_rubric.yaml   # LLM-judge rubric
│   └── overrides/               # Drop-in override files (local LLM, full data, etc.)
├── src/                         # Pipeline source code
│   ├── runner.py                # Orchestrator (initialize + run_query)
│   ├── agents/                  # ResearchManager, ResearchScientist, …
│   ├── pipelines/               # System 1, 2, 3 pipeline logic
│   ├── config/                  # YAML loader with ${ENV_VAR} interpolation
│   └── utils/                   # LLM wrapper, embeddings, ChromaDB, KG tools, …
├── scripts/
│   ├── run_mars.py              # Full MARS pipeline
│   ├── run_ablations.py         # Ablation conditions
│   ├── run_evaluation.py        # LLM-as-judge evaluation
│   └── build_showcase.py        # Generates the mars_showcase notebook
├── notebooks/
│   ├── walkthrough.ipynb        # Interactive pipeline demo
│   ├── graph_viz.ipynb          # Material Informed Subgraph visualisation
│   └── mars_showcase_detailed.ipynb  # Full per-query showcase report
├── data/                        # Data and generation tooling — see data/README.md
│   ├── MARS_Data/               # KGs, ChromaDBs, MaterialDB (dummy data included)
│   └── download_data.py         # Download full KGs from HuggingFace
├── paper_metadata/              # Literature metadata/abstracts behind the KGs — see paper_metadata/README.md
├── environment.yml              # Conda environment (exact version pins)
├── run_experiments.sh           # One-command reproducer
├── SI.md                        # Supplementary information
└── results_from_paper/          # Frozen paper outputs
```
---

## Sample results

Visualization of the substitute material proposed by MARS for the THV case study: a three-layer composite film consisting of a polyolefin elastomer base, an EVOH gas-barrier layer, and an OTS oleophobic surface coating. 

<img width="1093" height="268" alt="image" src="https://github.com/user-attachments/assets/4b21489c-5572-4979-989a-b1e7e2a6dd9f" />

Left: Schematic cross-section of the proposed layer stack; layer thicknesses are not drawn to scale. Right: Illustrative AI-generated rendering of the proposed film, depicting oil droplets beading on the oleophobic surface of the flexed, transparent film (generated with Gemini 3.5 Flash [56]). No physical sample was fabricated; both panels visualize the proposed design.

<img width="1560" height="372" alt="image" src="https://github.com/user-attachments/assets/c0660a80-3376-4c82-b7a0-92656d395efe" />

---

## License

MIT — see [LICENSE](LICENSE).

## Citation

```bibtex
@misc{hage2026mars,
  title         = {MARS: Hierarchical Multi-Agent Reasoning Systems Enable Knowledge-Grounded Material Substitution},
  author        = {Tarjei Paule Hage and Yu-Chuan Hsu and Wei Lu and Gayla Lyon and Jiezhu Jin and Markus J. Buehler},
  year          = {2026},
  eprint        = {TODO},
  archivePrefix = {arXiv},
  primaryClass  = {cs.AI},
}
```
