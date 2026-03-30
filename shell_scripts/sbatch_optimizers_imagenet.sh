#!/bin/bash

# Configuration
logs_dir=${1:-"slurm_logs/grid_search_imagenet"}
CPUS_PER_TASK=${2:-16}
MEM=${3:-"32G"}
ACCOUNT=${4:-"plggolemml25-gpu-a100"}
PARTITION=${5:-"plgrid-gpu-a100"}
JOB_TIME=${6:-"05:00:00"}

# Create logs directory
mkdir -p ${logs_dir}

# Grid search parameters
eps_values=(0.01 0.1 0.2)
alpha_values=(0.1 0)
optimizers=(gen jade de sade gwo shade lshade info)
models=(imagenet)

# Fixed parameters
test_size=100
pop_size=500
num_iters=300
output_dir="outputs_grouped/benchmark_imagenet"

# Calculate total number of array tasks
n_models=${#models[@]}
n_eps=${#eps_values[@]}
n_alpha=${#alpha_values[@]}
n_optimizers=${#optimizers[@]}
total_jobs=$((n_models * n_eps * n_alpha * n_optimizers))
array_end=$((total_jobs - 1))

echo "Starting imagenet grid search sbatch job array submission..."
echo "Grid parameters:"
echo "  eps: ${eps_values[@]}"
echo "  alpha: ${alpha_values[@]}"
echo "  optimizers: ${optimizers[@]}"
echo "  models: ${models[@]}"
echo "  test_size: ${test_size}"
echo "  pop_size: ${pop_size}"
echo "  num_iters: ${num_iters}"
echo "  Total tasks: ${total_jobs}"
echo ""

sbatch <<EOF
#!/bin/bash -l
#SBATCH --job-name=grid_search_imagenet
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

# Parameter arrays (expanded at submission time)
models=(${models[@]})
eps_values=(${eps_values[@]})
alpha_values=(${alpha_values[@]})
optimizers=(${optimizers[@]})

n_models=${n_models}
n_eps=${n_eps}
n_alpha=${n_alpha}
n_optimizers=${n_optimizers}

# Derive parameter indices from SLURM_ARRAY_TASK_ID
task_id=\$SLURM_ARRAY_TASK_ID

optimizer_idx=\$((task_id % n_optimizers))
remaining=\$((task_id / n_optimizers))

alpha_idx=\$((remaining % n_alpha))
remaining=\$((remaining / n_alpha))

eps_idx=\$((remaining % n_eps))
remaining=\$((remaining / n_eps))

model_idx=\$((remaining % n_models))

# Resolve parameter values
model=\${models[\$model_idx]}
eps=\${eps_values[\$eps_idx]}
alpha=\${alpha_values[\$alpha_idx]}
optimizer=\${optimizers[\$optimizer_idx]}

job_name="\${model}_\${optimizer}_eps\${eps}_alpha\${alpha}"

echo "------------------------------------------------"
echo "Date:         \$(date)"
echo "SLURM Job ID: \$SLURM_JOB_ID (Array: \$SLURM_ARRAY_JOB_ID, Task: \$SLURM_ARRAY_TASK_ID)"
echo "Job Name:     \${job_name}"
echo "Model:        \${model}"
echo "Optimizer:    \${optimizer}"
echo "Epsilon:      \${eps}"
echo "Alpha:        \${alpha}"
echo "Test Size:    ${test_size}"
echo "Pop Size:     ${pop_size}"
echo "Num Iters:    ${num_iters}"
echo "Host:         \$(hostname)"
echo "------------------------------------------------"

# Activate virtual environment
source .venv/bin/activate

echo "Running adversarial attack on ImageNet (microsoft/resnet-18)..."

python scripts/adversarial_attack.py \\
    --model \${model} \\
    --optimizer \${optimizer} \\
    --eps \${eps} \\
    --alpha \${alpha} \\
    --test_size ${test_size} \\
    --pop_size ${pop_size} \\
    --num_iters ${num_iters} \\
    --output_dir ${output_dir}

echo "Job finished with exit code \$?"
echo "------------------------------------------------"
EOF

echo "Job array submitted. Total: ${total_jobs} tasks (0-${array_end})"
echo "Logs directory: ${logs_dir}"
