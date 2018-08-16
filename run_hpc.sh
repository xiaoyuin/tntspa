#!/bin/bash
#SBATCH --time=4:00:00   # walltime
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8  # number of processor cores (i.e. threads)
#SBATCH -p gpu1    # K80 GPUs on Haswell node
#SBATCH -J "run_hpc"   # job name
#SBATCH --mem=20000   # minimum amount of real memory
#SBATCH -A p_adm # name of the project
#SBATCH --mail-user xiaoyu.yin@mailbox.tu-dresden.de
#SBATCH --mail-type ALL


srun python3 run_hpc.py

exit 0
