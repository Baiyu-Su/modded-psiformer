#!/bin/bash
#SBATCH -J ferminet_training
#SBATCH -t 24:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:8        # Request 8 GPUs on the node
#SBATCH --ntasks-per-node=1 # 1 task per node; JAX will spawn per GPU
#SBATCH --cpus-per-task=96  # Allocate all CPUs to the single task
#SBATCH -o logs/ferminet_%j.o
#SBATCH -e logs/ferminet_%j.e

# Create logs directory if it doesn't exist
mkdir -p logs

# Source environment
source ~/.bashrc
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ferminet

# Set up wandb (if using wandb monitoring)
# export WANDB_API_KEY=your_api_key_here

# JAX/GPU specific environment variables
export JAX_PLATFORM_NAME=gpu
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# Fix for JAX cuSPARSE library path issue (adjust path as needed)
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

# Enable JAX debugging and error reporting
export JAX_ENABLE_X64=true
export JAX_DISABLE_JIT=false

# Set up cache directories for JAX compilation
export JAX_CACHE_DIR="/tmp/jax_cache_${SLURM_JOB_ID}"
mkdir -p $JAX_CACHE_DIR

# Python error handling
export PYTHONFAULTHANDLER=1

# Generate timestamp for unique run identification
TIMESTEP=$(date +"%m.%d.%H%M%S")

echo "Starting FermiNet training at $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "Timestamp: $TIMESTEP"

# Run FermiNet training with your configuration
# Adjust config file and parameters as needed
srun python3 -u ferminet/main.py \
    --config ferminet/configs/c2h4.py \
    --config.log.save_path="./runs/c2h4_${TIMESTEP}" \
    --config.log.use_wandb=False \
    --config.optim.kfac.nesterov=True \
    --config.optim.kfac.momentum=0.98 \
    --config.optim.lr.rate=0.005 \
    --config.optim.kfac.norm_constraint=5.0 \
    --config.optim.kfac.constrain_l2=True \
    --config.mcmc.algorithm=mh \
    --config.pretrain.iterations=0

echo "FermiNet training completed at $(date)"
