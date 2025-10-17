#!/bin/bash


#SBATCH --job-name=exp29             # Descriptive job name
#SBATCH --time=11:30:00                       # Maximum wall time (hh:mm:ss)
#SBATCH --nodes=1                             # Number of nodes to use
#SBATCH --ntasks-per-node=4                   # Number of MPI tasks per node (e.g., 1 per GPU)
#SBATCH --cpus-per-task=5                    # Number of CPU cores per task (adjust as needed)
#SBATCH --gres=gpu:4                          # Number of GPUs per node (adjust to match hardware)
#SBATCH --partition=boost_usr_prod           # GPU-enabled partition
#SBATCH --output=logs/exp29.out              # File for standard output
#SBATCH --error=logs/exp29.err               # File for standard error
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
    --batch_size 8 \
    --micro_batch_size 1 \
    --learning_rate 2e-4 \
    --max_iters 300000 \
    --wandb_project "graspgpt-deepspeed" \
    --sort_unseg \
    --output_dir ../output/exp31 \
    --model_type gpt2-shallow-wide-4096 \
    --translation_argument \
    --add_unlabel_noise
    #--model_type gpt2-shallow-wide
    #--resume ../output/exp22/iter_15000 \
    #--translation_argument \
    #--model_type gpt2-shallow-wide \
    #--add_unlabel_noise \

echo "Training completed!"