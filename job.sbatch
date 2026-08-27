#!/bin/bash
#SBATCH --job-name=hybridRe
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=2
#SBATCH --output=my-slurm-job.out
#SBATCH --error=my-slurm-job.err
#SBATCH --mail-user=deyilmaz
#SBATCH --mail-type=END,FAIL
#SBATCH --time=6:00:00

CUDA_DEVICE=$(echo "$CUDA_VISIBLE_DEVICES," | cut -d',' -f $((SLURM_LOCALID + 1)) );
T_REGEX='^[0-9]$';
if ! [[ "$CUDA_DEVICE" =~ $T_REGEX ]]; then
        echo "error no reserved gpu provided"
fi
echo "Process $SLURM_PROCID of Job $SLURM_JOBID with the local id $SLURM_LOCALID using gpu id $CUDA_DEVICE (we may use gpu: $CUDA_VISIBLE_DEVICES on $(hostname))"
echo "computing on $(nvidia-smi --query-gpu=gpu_name --format=csv -i $CUDA_DEVICE | tail -n 1)"

python main.py
echo "done"