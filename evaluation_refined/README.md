# Pooled blind evaluation (the protocol reported in the paper)

Everything the paper's Table 2 and Figures 5 to 7 rest on is produced here: each run is
rendered into three standardized subsystem reports, and the reports of all six
configurations are judged together, blind, one call per seed and subsystem.

## Protocol

1. **Reports, not final exports.** `mars.json` carries no retrieval trail, so a judge
   cannot tell an evidenced number from an invented one. `build_subsystem_summaries.py`
   and `run_v2_reports.py` rebuild a System 1, System 2 and System 3 report from the raw
   artifacts (`system{1,2,3}_*.json`, `pipeline_run_*.json`, `chats/*.json`): every
   question, retrieved-document count, KG path count, answer and verdict, with the
   pipeline's own text quoted verbatim. Sections a reduced configuration cannot produce
   are kept and marked absent by design. Each System 2 report also embeds the
   internal-database record (identity and datasheet properties) of every proposed
   component matched to the database, so candidate anchoring is verifiable inside the
   report.
2. **Pooled, shuffled judging.** For each seed and subsystem the six reports are
   anonymised (Report A to F), shuffled with a fixed per-call seed, and scored in one
   judge call with a neutral configuration note (e.g. "one LLM call, no retrieval;
   sections marked absent are absent by design"). Scores are absolute, not ranks; the
   shuffle map is stored with the result.
3. **Rubrics.** `rubric_s{1,2,3}_final.yaml`: four criteria per subsystem on a 0 to 10
   scale in steps of 0.5.

   | Subsystem | Criteria |
   |---|---|
   | System 1 | completeness, scientific correctness, evidence grounding (null without retrieval), investigation quality (null without a trace) |
   | System 2 | requirement alignment, adversarial verification (null without a verification loop), candidate anchoring, adaptive reasoning (null without rejection-driven adaptation) |
   | System 3 | route coherence, grounded practicality, process compatibility, feasibility-verdict grounding (null without a feasibility analysis) |

   A null for a capability a configuration lacks by construction counts as zero in that
   configuration's subsystem mean. A null for a MARS capability that a particular run did
   not exercise (adaptive reasoning when the first candidate is accepted) is excluded from
   that run's mean. The judge-run JSONs record these files under their earlier names:
   `rubric_s1_v6`, `rubric_s2_v3`, `rubric_s3_v5`.
4. **Judge.** GPT-6-astra, chosen from a different model generation than either evaluated
   backbone. The same 15 calls were repeated with GPT-5.6-sol as a second judge (mean
   absolute per-criterion difference 0.18, no own-backbone preference); see
   `paper_data/ablation/judge_scores_gpt56_sol/`.

## Files

| File | Role |
|---|---|
| `build_subsystem_summaries.py` | Raw run artifacts to the three subsystem reports (Markdown for the judge, JSON with structured counts). Counting and string matching only; no LLM. |
| `run_v2_reports.py` | Build the reports for one run and judge them; the per-run entry point. |
| `run_refined_evaluation.py` | Judge machinery (`load_rubric`, `call_judge`) shared by every judge driver; stores the full prompt and raw response next to each result. |
| `evaluate_summaries.ipynb` | Notebook form of the builders and judges (cell 6 is the System 2 renderer with the embedded database records). Cell outputs contain absolute local paths. |
| `rubric_s{1,2,3}_final.yaml` | The rubrics above. |

## Where the outputs are

- `evaluation_trial/pooled_final6_astra/`: the primary judge outputs, one
  `eval_s<n>_seed<seed>_*.json` (scores, strengths, weaknesses, rationale, confidence,
  shuffle map) and one `judge_raw_*.json` per call, 15 calls in total.
- `paper_data/ablation/`: per-call scores of both judges, `scores_table.csv`,
  `per_criterion.csv` and `aggregates.json` (Table 2 and the figure data).

## Running it

```bash
conda activate MARS
python evaluation_refined/run_v2_reports.py --help      # build + judge one run
```

`run_v2_reports.py` judges one run at a time; the pooled six-report calls used for the
paper were driven by a script that maps configuration and seed to report paths and is
not part of the repository. GPT-5.6-sol rejects
`temperature=0` and needs `max_completion_tokens`; the judge call retries without the
temperature and records the deviation in the result JSON.
