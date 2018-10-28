#!/bin/bash
#SBATCH --time=12:00:00   # walltime
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8  # number of processor cores (i.e. threads)
#SBATCH -p gpu1,gpu2    # K80 GPUs on Haswell node
#SBATCH -J "fairseq_transformer"   # job name
#SBATCH -o "train_fairseq_transformer-%j.out"   # output name
#SBATCH --mem=20000   # minimum amount of real memory
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

# python3 preprocess.py -s en -t sparql --trainpref $DDIR/train --validpref $DDIR/dev --testpref $DDIR/test --destdir $DDIR/fairseq-data-bin

srun python3 fairseq/train.py $DDIR/fairseq-data-bin \
-a transformer_iwslt_de_en --optimizer adam --lr 0.0005 -s en -t sparql \
--label-smoothing 0.1 --dropout 0.3 --max-tokens 4000 \
--min-lr '1e-09' --lr-scheduler inverse_sqrt --weight-decay 0.0001 \
--criterion label_smoothed_cross_entropy \
--warmup-updates 4000 --warmup-init-lr '1e-07' \
--max-epoch 500 --save-interval 50 \
--adam-betas '(0.9, 0.98)' --save-dir $MDIR/transformer_iwslt_de_en