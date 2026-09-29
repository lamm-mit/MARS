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
export OPENAI_API_KEY="sk-..."
./run_experiments.sh -a -e
```

Results are written to `results/Query1/`. The frozen paper outputs are in `results_from_paper/` (the THV case-study run) and `paper_data/` (figures and the CSV/JSON data behind the ablation study, run-time analysis and SI feedback experiments).

---

## Installation

**Conda environment** — `environment.yml` installs all Python dependencies with exact version pins, including PyTorch with the appropriate CUDA build for your driver. Create and activate it:

```bash
conda env create -f environment.yml
conda activate MARS
```

**GraphReasoning** — no separate install. A pinned copy of [lamm-mit/GraphReasoning](https://github.com/lamm-mit/GraphReasoning) (commit `f1d6d44`) is vendored at `src/vendor/graphreasoning/` and imported from there. It was previously pip-installed from unpinned `master`, so two people installing on different days got different code; the copy makes KG generation reproducible and lets the compatibility fixes live where they take effect rather than in import-time monkeypatches. Do **not** `pip install GraphReasoning` alongside it — the vendored copy takes precedence and the installed one would sit unused. All modifications are documented in [`src/vendor/graphreasoning/VENDORED.md`](src/vendor/graphreasoning/VENDORED.md).

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

### Human-in-the-loop review

System 1 extracts hard constraints and required properties from the query. `--human-review` pauses the run there and waits for a domain expert to approve or amend that output before System 2 begins:

```bash
python scripts/run_mars.py --queries Query1 --human-review
```

The run writes `results/<Query>/artifacts/human_review.md` and polls it. In that file the expert can uncheck a constraint to remove it, check a property to promote it to a hard constraint, delete a property line to drop it, edit any line's text in place, and add free-text constraints. Setting `Status: APPROVED` resumes the pipeline; the applied edits are recorded in `human_review_result.json`.

Promotion is copy-not-move: a promoted property stays in the property list, because System 2's candidate proposal and KG grounding read only properties, so an item that moved rather than copied would silently vanish from grounding. If no approval arrives within `timeout_seconds` (default 1800) the run continues with the original System 1 output rather than failing. Configure under `pipelines.human_review` in `config/config.yaml`; the gate is off by default.

A related flag, `--itemized-check`, makes System 2's feasibility validation return a SATISFIED / VIOLATED / NO_EVIDENCE verdict for every property and constraint, producing an auditable per-candidate table instead of the model selecting a few "critical" properties. Only VIOLATED rejects a candidate; NO_EVIDENCE never does.

### Ablation conditions


| Condition                            | Description                                                      |
| ------------------------------------ | ---------------------------------------------------------------- |
| **MARS (gpt-oss-20b)**               | System 1 → System 2 ↔ System 3 with RAG + dual-KG reasoning      |
| **MARS (GPT-5.6-sol)**               | Same framework with GPT-5.6-sol as the backbone of every agent   |
| **3 LLM calls (no RAG/KG)**          | Three sequential calls mirroring the subsystem decomposition     |
| **1 LLM call (w/ RAG/KG)**           | Single call with a static RAG + KG packet retrieved from the query |
| **1 LLM call (no RAG/KG)**           | Single call, purely parametric (gpt-oss-20b)                     |
| **1 LLM call (GPT-5.6-sol)**         | Single parametric call with the closed-source backbone           |

Each configuration is run over five fixed seeds (`config/overrides/seed_*.yaml` for gpt-oss-20b; `gpt56sol_paperdata.yaml` and `gpt56sol_seed_*.yaml` for GPT-5.6-sol).


### LLM-judge evaluation

The evaluation reported in the paper is the pooled, blind protocol in `evaluation_refined/`: each configuration's run is rendered into three standardized subsystem reports (`run_v2_reports.py`, using the builders in `evaluate_summaries.ipynb`), and for every seed and subsystem the six reports are anonymised, shuffled and scored in one judge call by GPT-6-astra against `rubric_s{1,2,3}_final.yaml` (four criteria per subsystem, 0–10 in steps of 0.5; capabilities a configuration lacks by construction score null and count as zero). The judge outputs are in `evaluation_trial/pooled_final6_astra/`; per-call scores of both judges (GPT-6-astra and the GPT-5.6-sol agreement run) and the aggregated tables are in `paper_data/ablation/`. `scripts/run_evaluation.py` with `config/evaluation_rubric.yaml` is the earlier single-run protocol and is kept for reference.

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


`paper_data/` holds the figures and the CSV/JSON data behind the ablation study (Table 2, Figures 5–7), the run-time analysis and the SI feedback-isolation experiments; `paper_data/README.md` maps every manuscript item to its files and `manifest.json` records a SHA-256 per file.

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
│   ├── evaluation_rubric.yaml   # Earlier LLM-judge rubric (see evaluation_refined/ for the paper's)
│   └── overrides/               # Drop-in overrides: local LLM, full data, per-seed and per-backbone runs,
│                                #   replay_feedback_* for the SI feedback experiment
├── src/                         # Pipeline source code
│   ├── runner.py                # Orchestrator (initialize + run_query)
│   ├── agents/                  # ResearchManager, ResearchScientist, …
│   ├── pipelines/               # System 1, 2, 3 pipeline logic
│   ├── config/                  # YAML loader with ${ENV_VAR} interpolation
│   ├── utils/                   # LLM wrapper, embeddings, ChromaDB, KG tools,
│   │                            #   human_review.py (System 1 review gate), …
│   └── vendor/graphreasoning/   # Pinned GraphReasoning copy — see VENDORED.md
├── scripts/
│   ├── run_mars.py              # Full MARS pipeline
│   ├── run_ablations.py         # Ablation conditions
│   ├── run_evaluation.py        # Earlier single-run LLM-as-judge evaluation
│   └── build_showcase.py        # Generates the mars_showcase notebook
├── evaluation_refined/          # Pooled blind judging used in the paper: report builders, final rubrics
├── evaluation_trial/pooled_final6_astra/  # Primary judge outputs (GPT-6-astra), 5 seeds x 3 subsystems
├── paper_data/                  # Figures + CSV/JSON data behind every figure/table — see paper_data/README.md
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
