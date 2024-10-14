#!/usr/bin/bash
#SBATCH --job-name=euler_parallel
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --partition gpu
#SBATCH --constraint=a100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00-00:10

module purge
module load cuda
module load python

export PYTHONUNBUFFERED=TRUE

source $VENVDIR/my-jax-venv/bin/activate

srun nsys profile --cuda-graph-trace=node --stats=true python euler_distributed.py --N=512
