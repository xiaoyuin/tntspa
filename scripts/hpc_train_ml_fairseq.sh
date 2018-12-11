#!/bin/bash
#SBATCH -J "hpc_ml_fairseq"   # job name
#SBATCH -A p_adm # name of the project
#SBATCH --mail-type=ALL
#SBATCH --mail-user xiaoyu.yin@mailbox.tu-dresden.de
#SBATCH --time=8:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:1
#SBATCH --partition=ml
#SBATCH -o "train_ml_fairseq-%j.out"   # output name
#SBATCH --mem-per-cpu=6000  # minimum amount of real memory

ANACONDA3_INSTALL_PATH='/opt/anaconda3'
export PATH=$ANACONDA3_INSTALL_PATH/bin:$PATH
source /opt/DL/pytorch/bin/pytorch-activate

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
    if [ "$3" == "conv" ]
    then
        srun python3 fairseq/train.py $DDIR/fairseq-data-bin -s en -t sparql \
        --lr 0.5 --clip-norm 0.1 --dropout 0.2 --max-tokens 8000 \
        --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
        --arch fconv_wmt_en_de --lr-scheduler fixed \
        --max-epoch 500 --save-interval 50 --valid-subset valid,test \
        --save-dir $MDIR/fairseq_fconv_wmt_en_de
    elif [ "$3" == "lstm" ]
    then
        srun python3 fairseq/train.py $DDIR/fairseq-data-bin \
        -a lstm_luong_wmt_en_de --optimizer adam --lr 0.001 -s en -t sparql \
        --label-smoothing 0.1 --dropout 0.3 --max-tokens 4000 \
        --min-lr '1e-09' --lr-scheduler inverse_sqrt --weight-decay 0.0001 \
        --criterion label_smoothed_cross_entropy \
        --warmup-updates 4000 --warmup-init-lr '1e-07' \
        --max-epoch 100 --save-interval 10 --valid-subset valid,test \
        --adam-betas '(0.9, 0.98)' --save-dir $MDIR/lstm_luong_wmt_en_de
    elif [ "$3" == "transformer" ]
    then
        srun python3 fairseq/train.py $DDIR/fairseq-data-bin-joined -s en -t sparql \
        --arch transformer --share-all-embeddings \
        --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
        --lr-scheduler inverse_sqrt --warmup-init-lr 1e-07 --warmup-updates 4000 \
        --lr 0.0007 --min-lr 1e-09 \
        --dropout 0.3 --weight-decay 0.0 --criterion label_smoothed_cross_entropy \
        --label-smoothing 0.1 --max-tokens 2048 --update 2 \
        --max-epoch 100 --save-interval 10 --valid-subset valid,test \
        --save-dir $MDIR/transformer
    fi
fi

