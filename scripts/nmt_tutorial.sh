#!/bin/bash
#SBATCH --time=4:00:00   # walltime
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8  # number of processor cores (i.e. threads)
#SBATCH -p gpu1    # K80 GPUs on Haswell node
#SBATCH -J "nmt_tutorial"   # job name
#SBATCH --mem=10000   # minimum amount of real memory
#SBATCH -A p_adm # name of the project
#SBATCH --mail-user xiaoyu.yin@mailbox.tu-dresden.de
#SBATCH --mail-type END

module purge
module load modenv/eb
module load TensorFlow

nmt/scripts/download_iwslt15.sh /tmp/nmt_data
mkdir /scratch/p_adm/nmt_model
srun python3 -m nmt.nmt \
    --src=vi --tgt=en \
    --vocab_prefix=/tmp/nmt_data/vocab  \
    --train_prefix=/tmp/nmt_data/train \
    --dev_prefix=/tmp/nmt_data/tst2012  \
    --test_prefix=/tmp/nmt_data/tst2013 \
    --out_dir=/scratch/p_adm/nmt_model \
    --num_train_steps=12000 \
    --steps_per_stats=100 \
    --num_layers=2 \
    --num_units=128 \
    --dropout=0.2 \
    --metrics=bleu

exit 0
