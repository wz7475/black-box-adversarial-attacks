#!/bin/bash

# Configuration
logs_dir=${1:-"slurm_logs/grid_search"}
CPUS_PER_TASK=${2:-16}
MEM=${3:-"16G"}
ACCOUNT=${4:-"plggolemml25-gpu-a100-gpu-a100"}
PARTITION=${5:-"plgrid-gpu-a100"}
JOB_TIME=${6:-"06:00:00"}

# Create logs directory
mkdir -p ${logs_dir}

# Grid search parameters
eps_values=(0.01 0.1 0.2)
alpha_values=(1 10 100)
optimizers=(gen jade de sade gwo shade lshade info)
# models=(mnist cifar10)
models=(cifar10)

# Fixed parameters
test_size=150
pop_size=500
num_iters=500
output_dir="output"


echo "Starting grid search sbatch job submission..."
echo "Grid parameters:"
echo "  eps: ${eps_values[@]}"
echo "  alpha: ${alpha_values[@]}"
echo "  optimizers: ${optimizers[@]}"
echo "  models: ${models[@]}"
echo "  test_size: ${test_size}"
echo "  pop_size: ${pop_size}"
echo "  num_iters: ${num_iters}"
echo ""

job_count=0

for model in "${models[@]}"; do
    for eps in "${eps_values[@]}"; do
        for alpha in "${alpha_values[@]}"; do
            for optimizer in "${optimizers[@]}"; do
                job_name="${model}_${optimizer}_eps${eps}_alpha${alpha}"
                
                echo "Submitting job: ${job_name}"
                
                sbatch <<EOF
#!/bin/bash -l
#SBATCH --job-name=${job_name}
#SBATCH --account=${ACCOUNT}
#SBATCH --partition=${PARTITION}
#SBATCH --gres=gpu:1
#SBATCH --time=${JOB_TIME}
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=${CPUS_PER_TASK}
#SBATCH --mem=${MEM}
#SBATCH --output=${logs_dir}/${job_name}_%j.out
#SBATCH --error=${logs_dir}/${job_name}_%j.err

echo "------------------------------------------------"
echo "Date:         \$(date)"
echo "SLURM Job ID: \$SLURM_JOB_ID"
echo "Job Name:     ${job_name}"
echo "Model:        ${model}"
echo "Optimizer:    ${optimizer}"
echo "Epsilon:      ${eps}"
echo "Alpha:        ${alpha}"
echo "Test Size:    ${test_size}"
echo "Pop Size:     ${pop_size}"
echo "Num Iters:    ${num_iters}"
echo "Host:         \$(hostname)"
echo "------------------------------------------------"

# Activate virtual environment
source .venv/bin/activate

echo "Running adversarial attack..."

python scripts/adversarial_attack.py \\
    --model ${model} \\
    --optimizer ${optimizer} \\
    --eps ${eps} \\
    --alpha ${alpha} \\
    --test_size ${test_size} \\
    --pop_size ${pop_size} \\
    --num_iters ${num_iters} \\
    --output_dir ${output_dir}

echo "Job finished with exit code \$?"
echo "------------------------------------------------"
EOF

                job_count=$((job_count + 1))
                
            done
        done
    done
done

echo ""
echo "All jobs submitted. Total: ${job_count} jobs"
echo "Logs directory: ${logs_dir}"
