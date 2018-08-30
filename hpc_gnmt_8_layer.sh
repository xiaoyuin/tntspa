#!/bin/bash
#SBATCH --time=8:00:00   # walltime
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8  # number of processor cores (i.e. threads)
#SBATCH -p gpu1,gpu2    # K80 GPUs on Haswell node
#SBATCH -J "hpc_gnmt_8_layer"   # job name
#SBATCH --mem=20000   # minimum amount of real memory
#SBATCH -A p_adm # name of the project
#SBATCH --mail-user xiaoyu.yin@mailbox.tu-dresden.de
#SBATCH --mail-type ALL

srun python3 -m nmt.nmt.nmt \
    --src=en --tgt=sparql \
    --hparams_path=nmt_hparams/wmt16_gnmt_8_layer.json \
    --out_dir=output/models/wmt16_gnmt_8_layer \
    --vocab_prefix=data/monument_600/vocab \
    --train_prefix=data/monument_600/train \
    --dev_prefix=data/monument_600/dev \
    --test_prefix=data/monument_600/test

exit 0