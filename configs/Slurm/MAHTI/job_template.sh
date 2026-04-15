#!/bin/bash
#=============================================================================
# MAHTI Job Template - Parameterized via environment variables
#=============================================================================
# Required environment variables (passed via sbatch --export):
#   AREA          - Network area: "toy" or "area1"
#   DEMAND        - Demand float: 0.1, 0.2, etc.
#   BASIC_EPISODES - Number of basic training episodes: 32, 64, 128, 256, 512
#   SEED          - Random seed for reproducibility: 42, 123, 456, 789, 1024
#   DELTA         - RL step duration: 1, 3, 9, 30
#   NUM_ENVS      - Number of parallel SUMO environments: 1, 2, 4, 8, 16, 32
#   CONFIG        - Config file name (e.g., faster.yaml, slower.yaml)
#   GRADIENT_STEPS - Gradient steps per update: -1 (num_envs), 1
#   SCALING_COEF  - Episode scaling coefficient: 0, 0.01, ..., 1
#   JOB_NAME      - Constructed job name for logging/identification
#=============================================================================
# Note: All SBATCH directives are set via sbatch command line in run_experiments.sh
#=============================================================================

#=============================================================================
# Construct SUMOCFG path based on area
#=============================================================================

if [ "$AREA" == "toy" ]; then
    SUMOCFG="nets/ridepooling/older_net.sumocfg"
else
    # area1 path pattern with parametrized DEMAND
    SUMOCFG="nets/ridepooling/Helsinki_updated_areas/${AREA}/${AREA}_sampled_${DEMAND}_3000.sumocfg.xml"
fi

#=============================================================================
# Environment setup
#=============================================================================

# Load modules
module load pytorch/2.7

# List all modules
module list

# SUMO environment variables
export LIBSUMO_AS_TRACI=1
export SUMO_HOME="/projappl/project_2016787/AIforLEssAuto/WP4/rl-ridepooling/ridepool-venv/bin"

# OMP variables
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

export OMP_DISPLAY_AFFINITY=true

# Navigate to project directory
cd /projappl/project_2016787/AIforLEssAuto/WP4/rl-ridepooling
pwd

# Debug info
echo "=== Job Parameters ==="
echo "JOB_NAME: $JOB_NAME"
echo "AREA: $AREA"
echo "DEMAND: $DEMAND"
echo "BASIC_EPISODES: $BASIC_EPISODES"
echo "SEED: $SEED"
echo "DELTA: $DELTA"
echo "NUM_ENVS: $NUM_ENVS"
echo "CONFIG: $CONFIG"
if [ -n "$TRAIN_FREQ" ]; then
    echo "TRAIN_FREQ: $TRAIN_FREQ (override)"
fi
echo "GRADIENT_STEPS: $GRADIENT_STEPS"
echo "SCALING_COEF: $SCALING_COEF"
echo "SUMOCFG: $SUMOCFG"
echo "======================"

taskset -cp $$
srun echo "OMP_NUM_THREADS: $OMP_NUM_THREADS"

#=============================================================================
# Run training
#=============================================================================

# Build train-freq argument if TRAIN_FREQ is provided
TRAIN_FREQ_ARG=""
if [ -n "$TRAIN_FREQ" ]; then
    TRAIN_FREQ_ARG="--train-freq ${TRAIN_FREQ}"
fi

srun singularity exec \
    -B "/usr/lib64/libnsl.so.1" \
    -B /users/volodymy \
    "$SING_IMAGE" \
    bash -c "source ridepool-venv/bin/activate && \
    python src/tests/gym_test-rs.py \
    --config configs/policy_training/${CONFIG} \
    --num-envs ${NUM_ENVS} \
    --basic-episodes ${BASIC_EPISODES} \
    --seed ${SEED} \
    --delta ${DELTA} \
    ${TRAIN_FREQ_ARG} \
    --gradient-steps ${GRADIENT_STEPS} \
    --scaling-coef ${SCALING_COEF} \
    --sumocfg '${SUMOCFG}' \
    --postfix ${SLURM_JOB_ID}_${JOB_NAME}"
