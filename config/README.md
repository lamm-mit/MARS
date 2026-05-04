# Configuration Reference

MARS is configured via YAML files in this directory. The base config (`config.yaml`) sets all defaults. Override files deep-merge on top of it — only the keys present in the override file change; everything else inherits.

---

## Files


| File                                      | Purpose                                                                                                                                                                                    |
| ----------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `config.yaml`                             | Base configuration — LLM backend, embeddings, data paths, agent hyperparameters                                                                                                            |
| `prompts.yaml`                            | All LLM system and user prompts                                                                                                                                                            |
| `queries.yaml`                            | Benchmark query definitions                                                                                                                                                                |
| `evaluation_rubric.yaml`                  | LLM-judge rubric (12 subsystem criteria, 1–5 scale, judge model)                                                                                                                           |
| `overrides/local_LLM.yaml`                | Switch to a locally-hosted LLM endpoint                                                                                                                                                    |
| `overrides/downloaded_KGs.yaml`           | Point at the full downloaded KGs instead of dummy data                                                                                                                                     |
| `overrides/local_LLM_downloaded_KGs.yaml` | Both: local LLM + full downloaded KGs                                                                                                                                                      |
| `overrides/paper_reproduction.yaml`       | Full paper reproduction: local LLM + all real data (KGs, ChromaDBs, MaterialDB). Requires `MARS_KG_DIR`, `MARS_CHROMA_DIR`, and `MARS_MATERIAL_DB` to be set to where you store your data. |


---

## LLM Backend

The default backend is the **OpenAI API** (`gpt-5-nano`). This requires `OPENAI_API_KEY` to be set:

```bash
export OPENAI_API_KEY="sk-..."
./run_experiments.sh -e
```

To switch to a **locally-hosted LLM** (e.g. `gpt-oss-20b` via vLLM at `localhost:8081`):

```bash
./run_experiments.sh -e --local
# or directly:
python scripts/run_mars.py --override config/overrides/local_LLM.yaml
```

`overrides/local_LLM.yaml` sets the endpoint, model name, and temperature. Edit it to match your local server if needed.

> **Note:** The LLM-as-judge evaluation step (`run_evaluation.py`) always calls the OpenAI API (`gpt-4.1`) via `OPENAI_API_KEY`, regardless of which backend is used for the main pipeline.

---

## Data Paths

Data paths are controlled by three environment variables with defaults pointing at the included dummy data:


| Variable           | Default                                                             | Points to                                                |
| ------------------ | ------------------------------------------------------------------- | -------------------------------------------------------- |
| `MARS_KG_DIR`      | `./data/MARS_Data/KGs`                                              | Directory with `.graphml` + `.pkl` knowledge graph files |
| `MARS_CHROMA_DIR`  | `./data/MARS_Data/ChromaDBs_paper`                                  | Directory with ChromaDB persist directories              |
| `MARS_MATERIAL_DB` | `./data/MARS_Data/MaterialDB_paper/internal_material_database.json` | Internal material database JSON                          |


The dummy data works out of the box — no changes needed for an initial run. To use the full downloaded KGs:

```bash
python data/download_data.py   # ~2 GB
python scripts/run_mars.py --override config/overrides/downloaded_KGs.yaml
```

To point at custom data, either export the env vars or edit `overrides/local_LLM.yaml`.

---

## Override System

Any script that accepts `--override` deep-merges the given file on top of `config.yaml`:

```bash
python scripts/run_mars.py --override config/overrides/local_LLM.yaml
python scripts/run_mars.py --override config/overrides/downloaded_KGs.yaml
```

Only keys present in the override file are changed. You can create your own override file with any subset of keys from `config.yaml`.