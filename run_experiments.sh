#!/usr/bin/env bash
# run_experiments.sh — Reproduce MARS experiments
#
# Usage:
#   ./run_experiments.sh [OPTIONS]
#
# Options:
#   -q, --queries     Comma-separated query names (default: Query1)
#   -a, --ablations   Run MARS + all ablations (3agent, 1agent_rag,
#                     1agent_no_rag, 1agent_no_rag_openai)
#   -e, --eval        Run MARS + all ablations + LLM-judge evaluation
#   --local           Shorthand for --override config/overrides/local_LLM.yaml.
#                     No OPENAI_API_KEY needed for the main pipeline; still
#                     needed for the evaluation judge.
#   --override FILE   Apply a custom YAML override file (deep-merged on top of
#                     config/config.yaml). Skips the OPENAI_API_KEY check for
#                     the main pipeline — set --no-openai too if needed.
#   --no-openai       Skip 1agent_no_rag_openai ablation condition
#   --skip-mars       Skip Step 1 (full MARS pipeline)
#   --skip-ablations  Skip Step 2 (ablations); useful with -e to re-run evaluation only
#   -c, --condition   Run a single ablation condition (3agent|1agent_rag|
#                     1agent_no_rag|1agent_no_rag_openai). Only with -a or -e.
#   -h, --help        Show this help message
#
# Examples:
#   ./run_experiments.sh                                  # MARS only, OpenAI API (default)
#   ./run_experiments.sh --local                          # MARS only, local LLM
#   ./run_experiments.sh -e                               # full run, OpenAI API
#   ./run_experiments.sh -e --local                       # full run, local LLM
#   ./run_experiments.sh -e --override config/overrides/paper_reproduction.yaml
#   ./run_experiments.sh -e --no-openai                   # skip the gpt-5.4 ablation condition
#   ./run_experiments.sh -e --skip-mars --skip-ablations  # evaluation only (results already exist)

set -euo pipefail

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
QUERIES="Query1"
RUN_ABLATIONS=false
RUN_EVAL=false
LOCAL_BACKEND=false
OVERRIDE_FILE=""
SKIP_OPENAI=false
SKIP_MARS=false
SKIP_ABLATIONS=false
CONDITION=""

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    -q|--queries)      QUERIES="$2"; shift 2 ;;
    -a|--ablations)    RUN_ABLATIONS=true; shift ;;
    -e|--eval)         RUN_EVAL=true; RUN_ABLATIONS=true; shift ;;
    --local)           LOCAL_BACKEND=true; shift ;;
    --override)        OVERRIDE_FILE="$2"; shift 2 ;;
    --no-openai)       SKIP_OPENAI=true; shift ;;
    --skip-mars)       SKIP_MARS=true; shift ;;
    --skip-ablations)  SKIP_ABLATIONS=true; shift ;;
    -c|--condition)    CONDITION="$2"; shift 2 ;;
    -h|--help)
      grep '^#' "$0" | sed 's/^# \{0,1\}//' | sed -n '2,33p'
      exit 0 ;;
    *)
      echo "Unknown option: $1"
      echo "Run './run_experiments.sh --help' for usage."
      exit 1 ;;
  esac
done

# ---------------------------------------------------------------------------
# Locate project root (directory containing this script)
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ---------------------------------------------------------------------------
# Build override argument
# --local is a shorthand; --override takes an explicit file
# ---------------------------------------------------------------------------
OVERRIDE_ARG=""
if [[ -n "$OVERRIDE_FILE" ]]; then
  OVERRIDE_ARG="--override $OVERRIDE_FILE"
elif $LOCAL_BACKEND; then
  OVERRIDE_ARG="--override config/overrides/local_LLM.yaml"
fi

# ---------------------------------------------------------------------------
# Validate: OPENAI_API_KEY required for default OpenAI backend and openai ablation
# ---------------------------------------------------------------------------
_needs_openai=false
if ! $LOCAL_BACKEND && [[ -z "$OVERRIDE_FILE" ]]; then
  _needs_openai=true
fi
if $RUN_ABLATIONS && ! $SKIP_OPENAI; then
  if [[ -z "$CONDITION" || "$CONDITION" == "1agent_no_rag_openai" ]]; then
    _needs_openai=true
  fi
fi
if $_needs_openai && [[ -z "${OPENAI_API_KEY:-}" ]]; then
  if ! $LOCAL_BACKEND && [[ -z "$OVERRIDE_FILE" ]]; then
    echo "ERROR: OPENAI_API_KEY is not set (required for the OpenAI API backend)."
    echo "  Export OPENAI_API_KEY before running, or use --local / --override to select a different backend."
  else
    echo "ERROR: OPENAI_API_KEY is not set (required for 1agent_no_rag_openai)."
    echo "  Pass --no-openai to skip that condition."
  fi
  exit 1
fi

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo "============================================================"
echo "MARS Experiment Runner"
echo "============================================================"
echo "  Queries:    $QUERIES"
if [[ -n "$OVERRIDE_FILE" ]]; then
  echo "  Override:   $OVERRIDE_FILE"
elif $LOCAL_BACKEND; then
  echo "  Backend:    local (config/overrides/local_LLM.yaml)"
else
  echo "  Backend:    OpenAI API (gpt-5-nano)"
fi
if $SKIP_MARS; then
  echo "  MARS:       skipped (--skip-mars)"
else
  echo "  MARS:       yes"
fi
if $RUN_ABLATIONS; then
  if $SKIP_ABLATIONS; then
    echo "  Ablations:  skipped (--skip-ablations)"
  elif [[ -n "$CONDITION" ]]; then
    echo "  Ablations:  $CONDITION only"
  elif $SKIP_OPENAI; then
    echo "  Ablations:  3agent, 1agent_rag, 1agent_no_rag (OpenAI condition skipped)"
  else
    echo "  Ablations:  3agent, 1agent_rag, 1agent_no_rag, 1agent_no_rag_openai"
  fi
else
  echo "  Ablations:  no"
fi
echo "  Evaluation: $RUN_EVAL"
echo "============================================================"
echo ""

# ---------------------------------------------------------------------------
# Step 1: Full MARS pipeline
# ---------------------------------------------------------------------------
if $SKIP_MARS; then
  echo "Skipping Step 1 (--skip-mars)"
else
  echo "------------------------------------------------------------"
  echo "Step 1: Full MARS Pipeline"
  echo "------------------------------------------------------------"
  # shellcheck disable=SC2086
  python scripts/run_mars.py --queries "$QUERIES" $OVERRIDE_ARG
fi

# ---------------------------------------------------------------------------
# Step 2: Ablation conditions
# ---------------------------------------------------------------------------
if $RUN_ABLATIONS; then
  echo ""
  echo "------------------------------------------------------------"
  echo "Step 2: Ablation Conditions"
  echo "------------------------------------------------------------"
  if $SKIP_ABLATIONS; then
    echo "Skipping Step 2 (--skip-ablations)"
  elif [[ -n "$CONDITION" ]]; then
    # shellcheck disable=SC2086
    python scripts/run_ablations.py --queries "$QUERIES" --condition "$CONDITION" $OVERRIDE_ARG
  elif $SKIP_OPENAI; then
    for cond in 3agent 1agent_rag 1agent_no_rag; do
      # shellcheck disable=SC2086
      python scripts/run_ablations.py --queries "$QUERIES" --condition "$cond" $OVERRIDE_ARG
    done
  else
    # shellcheck disable=SC2086
    python scripts/run_ablations.py --queries "$QUERIES" $OVERRIDE_ARG
  fi
fi

# ---------------------------------------------------------------------------
# Step 3: LLM-judge evaluation
# ---------------------------------------------------------------------------
if $RUN_EVAL; then
  echo ""
  echo "------------------------------------------------------------"
  echo "Step 3: LLM-Judge Evaluation"
  echo "------------------------------------------------------------"
  python scripts/run_evaluation.py --queries "$QUERIES"
fi

echo ""
echo "============================================================"
echo "Done."
echo "============================================================"
