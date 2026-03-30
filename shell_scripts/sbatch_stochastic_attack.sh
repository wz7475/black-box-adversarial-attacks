#!/bin/bash

# Configuration
logs_dir=${1:-"slurm_logs/stochastic_attack"}
CPUS_PER_TASK=${2:-4}
MEM=${3:-"16G"}
ACCOUNT=${4:-"plggolemml25-gpu-a100"}
PARTITION=${5:-"plgrid-gpu-a100"}
JOB_TIME=${6:-"05:00:00"}

# Create logs directory
mkdir -p ${logs_dir}

# Stochastic attack parameters
base_idx=100
increment=5
num_jobs=40
test_size=5
repeat=100

array_end=$((num_jobs - 1))

echo "Starting stochastic attack sbatch job array submission..."
echo "Parameters:"
echo "  base_idx:  ${base_idx}"
echo "  increment: ${increment}"
echo "  num_jobs:  ${num_jobs}"
echo "  test_size: ${test_size}"
echo "  repeat:    ${repeat}"
echo "  Total tasks: ${num_jobs}"
echo ""

sbatch <<EOF
#!/bin/bash -l
#SBATCH --job-name=stochastic_attack
#SBATCH --account=${ACCOUNT}
#SBATCH --partition=${PARTITION}
#SBATCH --gres=gpu:1
#SBATCH --time=${JOB_TIME}
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=${CPUS_PER_TASK}
#SBATCH --mem=${MEM}
#SBATCH --output=${logs_dir}/job_%A_%a.out
#SBATCH --error=${logs_dir}/job_%A_%a.err
#SBATCH --array=0-${array_end}

# Derive start index from task ID
start_idx=\$((${base_idx} + \$SLURM_ARRAY_TASK_ID * ${increment}))

echo "------------------------------------------------"
echo "Date:         \$(date)"
echo "SLURM Job ID: \$SLURM_JOB_ID (Array: \$SLURM_ARRAY_JOB_ID, Task: \$SLURM_ARRAY_TASK_ID)"
echo "Start Index:  \${start_idx}"
echo "Test Size:    ${test_size}"
echo "Repeat:       ${repeat}"
echo "Host:         \$(hostname)"
echo "------------------------------------------------"

# Activate virtual environment
source .venv/bin/activate

echo "Running stochastic growth attack..."

python scripts/stochastic_growth_attack.py \
    --test_size ${test_size} \
    --start_from_test_idx \${start_idx} \
    --repeat ${repeat} \
    --alpha 0.1 \
    --output_dir outputs_grouped/stochastic_growth_alpha_0.1

echo "Job finished with exit code \$?"
echo "------------------------------------------------"
EOF

echo "Job array submitted. Total: ${num_jobs} tasks (0-${array_end})"
echo "Logs directory: ${logs_dir}"
