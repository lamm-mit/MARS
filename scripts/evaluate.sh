#!/usr/bin/env bash
# Run LLM-as-judge evaluation on a results tree (e.g. results/).
#
# Requires OPENAI_API_KEY. Writes per-query eval_*.json and aggregate_results.json under
# <RESULTS_ROOT>/evaluation/.
#
# Usage:
#   export OPENAI_API_KEY="sk-..."
#   ./scripts/evaluate.sh results
#   ./scripts/evaluate.sh results --queries Query1 --model gpt-4o
#
# Any arguments after RESULTS_ROOT are passed to scripts/run_evaluation.py.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

usage() {
  cat <<EOF
Usage: $(basename "$0") <RESULTS_ROOT> [run_evaluation.py options]

  RESULTS_ROOT   Directory containing QueryN/ folders with all 5 condition files:
                 mars.json, ablation_3agent.json, ablation_1agent_rag.json,
                 ablation_1agent_no_rag.json, ablation_1agent_no_rag_openai.json
                 (e.g. results). Evaluation output goes to RESULTS_ROOT/evaluation/.

  Remaining args are forwarded to scripts/run_evaluation.py (e.g. --queries Query1).

Environment: OPENAI_API_KEY must be set.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "ERROR: OPENAI_API_KEY is not set." >&2
  exit 1
fi

RESULTS_ROOT="${1:-results}"
if [[ $# -gt 0 ]]; then shift; fi

EVAL_DIR="${RESULTS_ROOT%/}/evaluation"

python scripts/run_evaluation.py \
  --results-root "${RESULTS_ROOT}" \
  --output-dir "${EVAL_DIR}" \
  "$@"

echo "[done] Evaluation: ${EVAL_DIR}"
