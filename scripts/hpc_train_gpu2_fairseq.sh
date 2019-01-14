#!/bin/bash
#SBATCH --time=24:00:00   # walltime
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6  # number of processor cores (i.e. threads)
#SBATCH -p gpu2    # K80 GPUs
#SBATCH -J "hpc_gpu2_fairseq"   # job name
#SBATCH -o "train_gpu2_fairseq-%j.out"   # output name
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
    if [ "$3" == "conv" ]
    then
        srun python3 fairseq/train.py $DDIR/fairseq-data-bin -s en -t sparql \
        --lr 0.5 --clip-norm 0.1 --dropout 0.2 --max-tokens 4000 \
        --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
        --arch fconv_wmt_en_de --lr-scheduler fixed --force-anneal 50 \
        --max-epoch 55 --save-interval 10 --valid-subset valid,test \
        --save-dir $MDIR/fairseq_fconv_wmt_en_de
    elif [ "$3" == "lstm" ]
    then
        srun python3 fairseq/train.py $DDIR/fairseq-data-bin \
        -a lstm_luong_wmt_en_de --optimizer adam --lr 0.001 -s en -t sparql \
        --label-smoothing 0.1 --dropout 0.3 --max-tokens 4000 \
        --min-lr '1e-09' --lr-scheduler inverse_sqrt --weight-decay 0.0001 \
        --criterion label_smoothed_cross_entropy \
        --warmup-updates 4000 --warmup-init-lr '1e-07' \
        --max-epoch 55 --save-interval 10 --valid-subset valid,test \
        --adam-betas '(0.9, 0.98)' --save-dir $MDIR/lstm_luong_wmt_en_de
    elif [ "$3" == "transformer" ]
    then
        srun python3 fairseq/train.py $DDIR/fairseq-data-bin \
        -a transformer_iwslt_de_en --optimizer adam --lr 0.0005 -s en -t sparql \
        --label-smoothing 0.1 --dropout 0.3 --max-tokens 4000 \
        --min-lr '1e-09' --lr-scheduler inverse_sqrt --weight-decay 0.0001 \
        --criterion label_smoothed_cross_entropy \
        --warmup-updates 4000 --warmup-init-lr '1e-07' \
        --max-epoch 55 --save-interval 10 --valid-subset valid,test \
        --adam-betas '(0.9, 0.98)' --save-dir $MDIR/transformer_iwslt_de_en
    fi
fi

