#!/bin/bash
#SBATCH --time=2:00:00   # walltime
#SBATCH --cpus-per-task=8  # number of processor cores (i.e. threads)
#SBATCH -p haswell,sandy,west    # K80 GPUs on Haswell node
#SBATCH -J "kerastest"   # job name
#SBATCH --mem=10000   # minimum amount of real memory
#SBATCH -A p_adm # name of the project

module purge
module load modenv/eb
module load Keras
module load TensorFlow

srun python mnist_cnn.py
