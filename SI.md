# Supplementary Information

This repository is the supplementary material for the paper:

**MARS: Hierarchical Multi-Agent Reasoning Systems Enable Knowledge-Grounded Material Substitution**
Tarjei Paule Hage, Yu-Chuan Hsu, Wei Lu, Markus J. Buehler — Massachusetts Institute of Technology

The paper presents and evaluates the MARS framework using a single case study: replacing THV (an elastomeric terpolymer of tetrafluoroethylene, hexafluoropropylene, and vinylidene fluoride) in applications requiring broad chemical resistance. For brevity, the paper reports only the final outputs of each subsystem and aggregate evaluation scores, explicitly directing the reader to this repository for the full intermediate artifacts, complete evaluation results, and the reproducible evaluation framework. This document describes where each piece of supplementary material lives.

---

## Correspondence Between the Paper and This Repository


| Paper reference                  | What it contains                                                                             | Location in this repository                                        |
| -------------------------------- | -------------------------------------------------------------------------------------------- | ------------------------------------------------------------------ |
| Text Box 1                       | Evaluation query (THV substitution)                                                          | `config/queries.yaml` → `Query1`                                   |
| Text Box 2 (§2.3.1)              | Final System 1 output: extracted properties W and hard constraints H                         | `results_from_paper/Query1/artifacts/system1_2026041810_0.json`    |
| Text Box 3 (§2.4.1)              | Final System 2 output: proposed candidate (ENGAGE / EVOH / OTS composite) with justification | `results_from_paper/Query1/artifacts/system2_2026041810_1.json`    |
| Text Box 4 (§2.5.1)              | Final System 3 output: manufacturability judgment and 9-step process recipe                  | `results_from_paper/Query1/artifacts/system3_2026041810_1.json`    |
| Figure 6 (§2.6)                  | Closed-loop rejection-aware search trace (System 2 × System 3 iterations)                    | `results_from_paper/Query1/artifacts/pipeline_run_2026041810.json` |
| Table 3 / Figures 9–10 (§2.7.1)  | Subsystem and criterion-level average scores                                                 | `results_from_paper/evaluation/`                                   |
| §4.7 / full evaluation framework | Complete 12-criterion LLM-as-judge rubric with ordinal scale definitions                     | `config/evaluation_rubric.yaml`                                    |


---

## What This Repository Contains Beyond the Paper

### 1. Full Source Code

The complete Python implementation of the MARS framework: agent implementations (`src/agents/`), pipeline logic (`src/pipelines/`), utility modules (`src/utils/`), and experiment scripts (`scripts/`). All system prompts and agent hyperparameters are externalised in `config/prompts.yaml` and `config/config.yaml`.

### 2. Full System Configuration


| File                            | Description                                                                                                                |
| ------------------------------- | -------------------------------------------------------------------------------------------------------------------------- |
| `config/config.yaml`            | Agent hyperparameters, LLM endpoint, embedding model, and data paths                                                       |
| `config/prompts.yaml`           | All LLM system and user prompts across all agents and subsystems                                                           |
| `config/evaluation_rubric.yaml` | Complete 12-criterion LLM-as-judge rubric (the "complete evaluation framework" referenced in §2.7 and §2.7.1 of the paper) |
| `config/queries.yaml`           | Benchmark query definitions, including the THV substitution query (Text Box 1)                                             |


### 3. Full Intermediate Pipeline Artifacts (THV Substitution Run)

Sections 2.3.1, 2.4.1, and 2.5.1 of the paper each state that "the full model outputs are provided in the Supplementary Information." These are the intermediate artifacts stored throughout the run (`run ID: 2026041810`), which the paper does not report:


| File                                     | Contents                                                                                                                                                                  |
| ---------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `artifacts/system1_2026041810_0.json`    | Full System 1 output: retrieved documents, KG paths, all Q&A pairs, extracted properties W and constraints H                                                              |
| `artifacts/system2_2026041810_0.json`    | System 2, iteration 1: Material-Informed Subgraph query, first candidate (Engaged Elastomer / INFULEX Barrier / PDMS Topcoat), validation subqueries, blocking assessment |
| `artifacts/system2_2026041810_1.json`    | System 2, iteration 2: refined candidate (ENGAGE / EVOH / OTS composite), validation subqueries, feasibility assessment                                                   |
| `artifacts/system3_2026041810_0.json`    | System 3, iteration 1: manufacturability assessment of first candidate — blocked (IP constraints on PDMS processing)                                                      |
| `artifacts/system3_2026041810_1.json`    | System 3, iteration 2: manufacturability assessment of final candidate — approved, with 9-step process recipe                                                             |
| `artifacts/pipeline_run_2026041810.json` | Full pipeline metadata: timing, run IDs, iteration history, and final outcome                                                                                             |
| `artifacts/evaluation_2026041810.json`   | Evaluation export linking pipeline run to the LLM-judge evaluation                                                                                                        |
| `artifacts/rejected_candidates.json`     | All candidates internally rejected during System 2 self-evaluation, with associated constraints                                                                           |


All paths above are relative to `results_from_paper/Query1/`.

### 4. Material-Informed Knowledge Graph Subgraph

`results_from_paper/Query1/artifacts/subgraphs/2026041810_0_material_informed.json` contains the full dual-KG material-informed subgraph constructed by System 2 for the THV query. This graph encodes relational connections between the required material properties W and lab-available materials, drawn from both the Material Properties KG and the Patents KG. The graph structure corresponds to the `((W, M), E)` representation described in §2.4.

The `notebooks/graph_viz.ipynb` notebook visualises this subgraph at multiple levels of detail.

### 5. LLM Agent Chat Logs

`results_from_paper/Query1/artifacts/chats/` contains the complete turn-by-turn agent interaction logs for all subsystems and iterations:


| File                                                              | Contents                                                                           |
| ----------------------------------------------------------------- | ---------------------------------------------------------------------------------- |
| `system1_chat_log_material_requirements_2026041810_0.json`        | System 1: all agent exchanges during requirements extraction (§2.3)                |
| `system2_chat_log_material_discovery_2026041810_0.json`           | System 2, iteration 1: candidate proposal and validation loop (§2.4)               |
| `system2_chat_log_material_discovery_2026041810_1.json`           | System 2, iteration 2: refined candidate proposal after System 3 blocking feedback |
| `system3_chat_log_manufacturability_assessment_2026041810_0.json` | System 3, iteration 1: manufacturability assessment of first candidate (§2.5)      |
| `system3_chat_log_manufacturability_assessment_2026041810_1.json` | System 3, iteration 2: manufacturability assessment of final candidate             |


### 6. Complete Ablation Outputs

The paper reports aggregate evaluation scores across the five configurations in Table 3 and Figures 9–10. The full outputs, which the judge model received as input, are:


| File                                                           | Paper name (Table 1)                      | Description                                                 |
| -------------------------------------------------------------- | ----------------------------------------- | ----------------------------------------------------------- |
| `results_from_paper/Query1/mars.json`                          | MARS                                      | Full MARS pipeline output                                   |
| `results_from_paper/Query1/ablation_3agent.json`               | Ablation 3 — 3 LLM Calls (No RAG/KG)      | Three sequential LLM calls, no retrieval or graph grounding |
| `results_from_paper/Query1/ablation_1agent_rag.json`           | Ablation 2 — 1 LLM Call (w/ RAG/KG)       | Single call with all RAG and KG context concatenated        |
| `results_from_paper/Query1/ablation_1agent_no_rag.json`        | Ablation 1 — 1 LLM Call (No RAG/KG)       | Single call, purely parametric                              |
| `results_from_paper/Query1/ablation_1agent_no_rag_openai.json` | GPT-5.4 — 1 LLM Call (No RAG/KG, GPT-5.4) | Single call using GPT-5.4 as base model                     |


### 7. Full Criterion-Level Evaluation Results

Section 2.7.1 states that "complete results for all evaluated configurations together with the reproducible evaluation framework is provided in the Supplementary Information." The paper reports subsystem and total averages (Table 3). This repository contains the complete per-criterion scores across all 12 criteria for each configuration:


| File                                                   | Contents                                                                                     |
| ------------------------------------------------------ | -------------------------------------------------------------------------------------------- |
| `results_from_paper/evaluation/eval_Query1.json`       | Per-criterion scores (12 criteria × 5 configurations) with judge rationale and blind mapping |
| `results_from_paper/evaluation/aggregate_results.json` | Aggregated rankings and average scores across all criteria and configurations                |


The 12 evaluation criteria (4 per subsystem, Table 2 in the paper):


| System 1               | System 2                  | System 3                        |
| ---------------------- | ------------------------- | ------------------------------- |
| Completeness           | Alignment with properties | Plausibility of synthesis route |
| Relevance to query     | Novelty                   | Processing practicality         |
| Scientific correctness | Realism                   | Compatibility with material     |
| Clarity and structure  | Reasoning quality         | Industrial relevance            |


### 8. Interactive Notebooks


| Notebook                      | Description                                                                                                                                               |
| ----------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `notebooks/walkthrough.ipynb` | Loads the frozen paper results and visualises the full pipeline run: execution timeline (Gantt), agent chat logs, KG subgraph, and per-system outputs     |
| `notebooks/graph_viz.ipynb`   | Visualises the Material-Informed Subgraph at multiple levels of detail: ego-graphs around individual property nodes, and the full 507-node graph topology |
| `notebooks/mars_showcase_detailed.ipynb` | Full per-query showcase report: requirements, closed-loop System 2⇄3 search, knowledge subgraph, manufacturing route, agent reasoning traces, and blind evaluation (where available) |


---

## Note on LLM Backend

The experiments reported in the paper were conducted using `openai/gpt-oss-20b` (natively quantised in MXFP4 format) served locally via `llama.cpp` on a single node with 4 × NVIDIA Tesla V100S GPUs (32 GB VRAM each), as described in §4.1. The repository defaults to the OpenAI API (`gpt-5-nano`) to allow anyone to run the pipeline without local GPU infrastructure. Instructions for switching to a locally-hosted model are in `config/README.md`.