#!/bin/bash
#SBATCH --time=12:00:00   # walltime
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8  # number of processor cores (i.e. threads)
#SBATCH -p gpu1,gpu2    # K80 GPUs on Haswell node
#SBATCH -J "t2t_lstm"   # job name
#SBATCH -o "train_t2t_lstm_bahdanau-%j.out"   # output name
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

USR_DIR=.
PROBLEM=translate_ensparql
MODEL=lstm_seq2seq_attention_bidirectional_encoder
HPARAMS=lstm_bahdanau_attention

# Train
# *  If you run out of memory, add --hparams='batch_size=1024'.
srun ~/.local/bin/t2t-trainer \
  --t2t_usr_dir=$USR_DIR \
  --data_dir=$DDIR \
  --problem=$PROBLEM \
  --model=$MODEL \
  --hparams_set=$HPARAMS \
  --hparams='batch_size=1024' \
  --train_steps=30000 \
  --keep_checkpoint_max 10 \
  --output_dir=$MDIR/lstm_bahdanau_attention