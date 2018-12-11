#!/bin/bash
#SBATCH -J "hpc_ml_nmt"   # job name
#SBATCH -A p_adm # name of the project
#SBATCH --mail-type=ALL
#SBATCH --mail-user xiaoyu.yin@mailbox.tu-dresden.de
#SBATCH --time=8:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:1
#SBATCH --partition=ml
#SBATCH -o "train_ml_nmt-%j.out"   # output name
#SBATCH --mem-per-cpu=6000   # minimum amount of real memory

ANACONDA3_INSTALL_PATH='/opt/anaconda3'
export PATH=$ANACONDA3_INSTALL_PATH/bin:$PATH
source /opt/DL/tensorflow/bin/tensorflow-activate

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

