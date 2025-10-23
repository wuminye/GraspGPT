#!/bin/bash


#SBATCH --job-name=exp46            # Descriptive job name
#SBATCH --time=18:30:00                       # Maximum wall time (hh:mm:ss)
#SBATCH --nodes=1                             # Number of nodes to use
#SBATCH --ntasks-per-node=4                   # Number of MPI tasks per node (e.g., 1 per GPU)
#SBATCH --cpus-per-task=5                    # Number of CPU cores per task (adjust as needed)
#SBATCH --gres=gpu:4                          # Number of GPUs per node (adjust to match hardware)
#SBATCH --partition=boost_usr_prod           # GPU-enabled partition
#SBATCH --output=logs/exp46.out              # File for standard output
#SBATCH --error=logs/exp46.err               # File for standard error
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
    --max_iters 450000 \
    --wandb_project "graspgpt-deepspeed" \
    --sort_unseg \
    --output_dir ../output/exp49 \
    --model_type gpt2-shallow-wide-1600-25 \
    --translation_argument \
    --add_unlabel_cropping \
    --enable_flood_fill \
    --del_z 0 \
    --token_mode unseg_and_scene_grasp \
    --data_path ../output/precomputed_data_large \
    --resume ../output/exp49/iter_372000 \
    #--resume ../output/exp45/iter_213000 \
    #--resume ../output/exp38/iter_276000\
    #--model_type gpt2-shallow-wide
    #--resume ../output/exp22/iter_15000 \
    #--translation_argument \
    #--model_type gpt2-shallow-wide \
    #--add_unlabel_noise \
    #['unseg_and_scene_grasp', 'unseg_only', 'unseg_grasp']

echo "Training completed!"