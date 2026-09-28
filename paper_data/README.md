# paper_data

Figures and the numbers behind them for every experiment reported in the manuscript.
Figures are PDF and PNG; data are CSV or JSON. `manifest.json` lists size and SHA-256 for
every file.

| Manuscript item | Figure files | Data files |
|---|---|---|
| Table 2 (subsystem and total scores, mean ± s.d. over seeds) | | `ablation/scores_table.csv`, `ablation/aggregates.json` |
| Figure 5: overall and per-subsystem scores, six configurations | `ablation/fig5_overall.*`, `ablation/fig6a_systems.*` | `ablation/scores_table.csv` |
| Figure 6: criterion-level scores of the two MARS backbones | `ablation/fig7_backbones.*` | `ablation/per_criterion.csv` |
| Criterion-level scores, all six configurations (bar and radar) | `ablation/fig6c_criteria.*`, `ablation/fig6b_radar.*` | `ablation/per_criterion.csv` |
| Per-call judge output (scores, strengths, weaknesses, rationale, shuffle map) | | `ablation/judge_scores_gpt6_astra/eval_s<system>_seed<seed>_*.json` (primary judge, 15 calls); `ablation/judge_scores_gpt56_sol/` (second judge, same 15 calls) |
| Agent interactions and wall-clock per query; Query 1 backbone run-time table | `runtime/fig_all_runtime.*` | `runtime/counts.csv`, `runtime/table_q1_runtime.csv` |
| SI: System 3 to System 2 feedback isolation, fixed trajectory | `feedback_ablation/fig_fixed_trajectory.*` | `feedback_ablation/fixed_trajectory_summary.json` |
| SI: feedback isolation by seed replay | `feedback_ablation/fig_seed_replay.*` | `feedback_ablation/seed_replay_summary.json` |
| Text Boxes 2 to 4 and Figure 4: reported THV run (pipeline 2026041810) | | `case_study/mars.json`, `case_study/rejected_candidates.json` (full artifacts in `results_from_paper/Query1/`) |

## How the numbers are defined

- Judge scores are 0 to 10 in steps of 0.5, four criteria per subsystem, one pooled
  blind call per seed and subsystem with all six configurations shuffled (see
  `evaluation_refined/README.md`). Means and standard deviations are over the five seeds
  (sample s.d.).
- For the reduced configurations a criterion whose capability is structurally absent is
  null in the judge output and counts as 0 in the subsystem mean. For the two MARS
  configurations a null (adaptive reasoning when the first candidate is accepted) is
  dropped from that run's mean; `per_criterion.csv` gives the number of seeds scored.
- Total = per-seed mean of the three subsystem means, over seeds that have all three.
  MARS (GPT-5.6-sol) seed 303 exhausted System 2 and has no System 3, so its System 3
  and Total are over four seeds (`*_n` columns).
- `runtime/counts.csv`: Queries 5 and 6 predate the System 3 chat-logging fix, so their
  RAG counts exclude System 3 retrieval and are lower bounds; Queries 1 to 3 are complete.
- `aggregates.json` keys: `SM` (per subsystem, per configuration: mean, s.d., n),
  `TOT` (total), `per_criterion`, and `TOT_all_seeds_unused` (the variant that keeps seed
  303 with two subsystems; not used anywhere).
