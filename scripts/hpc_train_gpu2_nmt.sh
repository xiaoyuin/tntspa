#!/bin/bash
#SBATCH --time=24:00:00   # walltime
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6  # number of processor cores (i.e. threads)
#SBATCH -p gpu2    # K80 GPUs
#SBATCH -J "hpc_gpu2_nmt"   # job name
#SBATCH -o "train_gpu2_nmt-%j.out"   # output name
#SBATCH --mem=40000   # minimum amount of real memory
#SBATCH -A p_adm # name of the project
#SBATCH --mail-user xiaoyu.yin@mailbox.tu-dresden.de
#SBATCH --mail-type ALL
module load TensorFlow/1.8.0-foss-2018a-Python-3.6.4-CUDA-9.2.88

DDIR=data/monument_600
MDIR=output/models

if [ -n "$1" ]
    then DDIR=$1
fi

if [ -n "$2" ]
    then MDIR=$2
fi

if [ -n "$3" ]
then
    if [ "$3" == "gnmt_8" ]
    then 
        srun python3 -m nmt.nmt.nmt \
        --src=en --tgt=sparql \
        --hparams_path=nmt_hparams/wmt16_gnmt_8_layer.json \
        --out_dir=$MDIR/wmt16_gnmt_8_layer \
        --vocab_prefix=$DDIR/vocab \
        --train_prefix=$DDIR/train \
        --dev_prefix=$DDIR/dev \
        --test_prefix=$DDIR/test
    elif [ "$3" == "gnmt_4" ]
    then
        srun python3 -m nmt.nmt.nmt \
        --src=en --tgt=sparql \
        --hparams_path=nmt_hparams/wmt16_gnmt_4_layer.json \
        --out_dir=$MDIR/wmt16_gnmt_4_layer \
        --vocab_prefix=$DDIR/vocab \
        --train_prefix=$DDIR/train \
        --dev_prefix=$DDIR/dev \
        --test_prefix=$DDIR/test
    fi
fi

