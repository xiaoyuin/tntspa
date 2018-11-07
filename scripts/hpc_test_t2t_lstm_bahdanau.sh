#!/bin/bash
#SBATCH --time=12:00:00   # walltime
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8  # number of processor cores (i.e. threads)
#SBATCH -p gpu1,gpu2    # K80 GPUs on Haswell node
#SBATCH -J "t2t_test"   # job name
#SBATCH -o "test_t2t_lstm-%j.out"   # output name
#SBATCH --mem=20000   # minimum amount of real memory
#SBATCH -A p_adm # name of the project
#SBATCH --mail-user xiaoyu.yin@mailbox.tu-dresden.de
#SBATCH --mail-type ALL

DDIR=data/monument_600
MDIR=output/models
RDIR=results/result

if [ -n "$1" ]
    then DDIR=$1
fi

if [ -n "$2" ]
    then MDIR=$2
fi

if [ -n "$3" ]
then 
    RDIR=$3
    if ! [ -e $RDIR ]
    then mkdir -p $RDIR
    fi
fi



# Decode

USR_DIR=.
PROBLEM=translate_ensparql
MODEL=lstm_seq2seq_attention_bidirectional_encoder
HPARAMS=lstm_bahdanau_attention

BEAM_SIZE=4
ALPHA=0.6

srun ~/.local/bin/t2t-decoder \
  --t2t_usr_dir=$USR_DIR \
  --data_dir=$DDIR \
  --problem=$PROBLEM \
  --output_dir=$MDIR \
  --hparams_set=$HPARAMS \
  --model=$MODEL \
  --decode_hparams="beam_size=$BEAM_SIZE,alpha=$ALPHA" \
  --decode_from_file=$DDIR/dev.en \
  --decode_to_file=$RDIR/dev_translation.sparql

srun ~/.local/bin/t2t-decoder \
  --t2t_usr_dir=$USR_DIR \
  --data_dir=$DDIR \
  --problem=$PROBLEM \
  --output_dir=$MDIR \
  --hparams_set=$HPARAMS \
  --model=$MODEL \
  --decode_hparams="beam_size=$BEAM_SIZE,alpha=$ALPHA" \
  --decode_from_file=$DDIR/test.en \
  --decode_to_file=$RDIR/test_translation.sparql

# Evaluate the BLEU score
# Note: Report this BLEU score in papers, not the internal approx_bleu metric.
# t2t-bleu --translation=translation.en --reference=ref-translation.de

# Query and Analyze
srun python3 generate.py $DDIR $RDIR