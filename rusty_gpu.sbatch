#!/usr/bin/bash
#SBATCH --job-name=euler_parallel
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --partition gpu
#SBATCH --constraint=a100
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00-01:00

module purge
module load cuda
module load python

export PYTHONUNBUFFERED=TRUE

cd $HOME/Projects/Beast/beast/sandbox

source $VENVDIR/my-jax-venv/bin/activate

srun python euler_distributed.py --N=8192 --double
