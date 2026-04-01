#!/bin/bash
# =============================================================================
# MARL Vector Baseline — Full Training Run Script
# =============================================================================
# Algorithms : QMIX, MAPPO, MADDPG
# Environments: LBF, RWARE, MPE, Overcooked
# Seeds       : 1, 2, 3, 4, 5
# Total runs  : 3 x 4 x 5 = 60
#
# Usage:
#   chmod +x run_baselines.sh
#   ./run_baselines.sh                    # run all 60 experiments
#   ./run_baselines.sh qmix lbf           # run one algorithm/env (all 5 seeds)
#   ./run_baselines.sh qmix lbf 1         # run one specific seed
#
# Results saved to: results/sacred/<alg>/<env>/
# Log file:         results/run_log.txt
# =============================================================================

set -e

# --- Configuration ---
SEEDS=(1 2 3 4 5)
ALGORITHMS=(qmix mappo maddpg)
ENVIRONMENTS=(lbf rware mpe overcooked)
LOG_FILE="results/run_log.txt"
SRC_DIR="$(cd "$(dirname "$0")/src" && pwd)"

# --- Colours for terminal output ---
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# --- Helper functions ---
log() {
    echo -e "$1" | tee -a "$LOG_FILE"
}

timestamp() {
    date '+%Y-%m-%d %H:%M:%S'
}

# --- Create results directory ---
mkdir -p results
echo "=== MARL Baseline Run Started: $(timestamp) ===" >> "$LOG_FILE"

# --- Parse optional arguments ---
# Usage: ./run_baselines.sh [alg] [env] [seed]
if [ "$#" -eq 3 ]; then
    ALGORITHMS=("$1")
    ENVIRONMENTS=("$2")
    SEEDS=("$3")
elif [ "$#" -eq 2 ]; then
    ALGORITHMS=("$1")
    ENVIRONMENTS=("$2")
elif [ "$#" -eq 1 ]; then
    ALGORITHMS=("$1")
fi

# --- Count total runs ---
TOTAL=$(( ${#ALGORITHMS[@]} * ${#ENVIRONMENTS[@]} * ${#SEEDS[@]} ))
CURRENT=0
FAILED=0

log "\n${BLUE}============================================${NC}"
log "${BLUE}  MARL Vector Baseline — Full Training Run  ${NC}"
log "${BLUE}============================================${NC}"
log "Algorithms   : ${ALGORITHMS[*]}"
log "Environments : ${ENVIRONMENTS[*]}"
log "Seeds        : ${SEEDS[*]}"
log "Total runs   : $TOTAL"
log "Started      : $(timestamp)"
log "Source dir   : $SRC_DIR"
log "${BLUE}============================================${NC}\n"

# --- Main loop ---
for ALG in "${ALGORITHMS[@]}"; do
    for ENV in "${ENVIRONMENTS[@]}"; do
        for SEED in "${SEEDS[@]}"; do
            CURRENT=$(( CURRENT + 1 ))
            RUN_ID="${ALG}_${ENV}_seed${SEED}"

            log "\n${YELLOW}[$CURRENT/$TOTAL] Starting: $RUN_ID${NC}"
            log "  Time: $(timestamp)"

            # Run the experiment
            START_T=$(date +%s)

            if python "$SRC_DIR/main.py" \
                --config="$ALG" \
                --env-config="$ENV" \
                with seed="$SEED" \
                >> "$LOG_FILE" 2>&1; then

                END_T=$(date +%s)
                ELAPSED=$(( END_T - START_T ))
                MINS=$(( ELAPSED / 60 ))
                SECS=$(( ELAPSED % 60 ))
                log "  ${GREEN}✓ COMPLETED${NC} — ${MINS}m ${SECS}s"
            else
                END_T=$(date +%s)
                ELAPSED=$(( END_T - START_T ))
                MINS=$(( ELAPSED / 60 ))
                SECS=$(( ELAPSED % 60 ))
                log "  ${RED}✗ FAILED${NC} — ${MINS}m ${SECS}s — check $LOG_FILE for details"
                FAILED=$(( FAILED + 1 ))
            fi

        done
    done
done

# --- Summary ---
SUCCEEDED=$(( TOTAL - FAILED ))
log "\n${BLUE}============================================${NC}"
log "  Run Complete: $(timestamp)"
log "  Total    : $TOTAL"
log "  ${GREEN}Succeeded: $SUCCEEDED${NC}"
if [ "$FAILED" -gt 0 ]; then
    log "  ${RED}Failed   : $FAILED${NC}"
else
    log "  Failed   : 0"
fi
log "${BLUE}============================================${NC}"

echo "=== MARL Baseline Run Finished: $(timestamp) ===" >> "$LOG_FILE"