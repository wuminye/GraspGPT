#!/bin/bash


#SBATCH --job-name=exp24             # Descriptive job name
#SBATCH --time=11:30:00                       # Maximum wall time (hh:mm:ss)
#SBATCH --nodes=1                             # Number of nodes to use
#SBATCH --ntasks-per-node=4                   # Number of MPI tasks per node (e.g., 1 per GPU)
#SBATCH --cpus-per-task=5                    # Number of CPU cores per task (adjust as needed)
#SBATCH --gres=gpu:4                          # Number of GPUs per node (adjust to match hardware)
#SBATCH --partition=boost_usr_prod           # GPU-enabled partition
#SBATCH --output=logs/exp24.out              # File for standard output
#SBATCH --error=logs/exp24.err               # File for standard error
#SBATCH --account=EUHPC_D22_064          # Project account number

# Load necessary modules (adjust to your environment)
module load cuda/12.1                        # Load CUDA toolkit
module load gcc                           # Load MPI implementation



# Launch the distributed GPU application
# Replace with your actual command (e.g., mpirun or srun)
cd graspGPT

echo "Starting DeepSpeed training with 4 GPUs..."


# Run DeepSpeed training
# batch_size will be automatically calculated based on micro_batch_size and world_size
deepspeed --num_gpus=4 train_deepspeed.py \
    --deepspeed_config ../deepspeed_config.json \
    --batch_size 16 \
    --micro_batch_size 2 \
    --learning_rate 2e-4 \
    --max_iters 300000 \
    --wandb_project "graspgpt-deepspeed" \
    --sort_unseg \
    --translation_argument \
    --output_dir ../output/exp24

echo "Training completed!"