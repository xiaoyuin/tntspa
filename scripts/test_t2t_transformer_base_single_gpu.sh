
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
MODEL=transformer
HPARAMS=transformer_base_single_gpu

BEAM_SIZE=4
ALPHA=0.6

t2t-decoder \
  --t2t_usr_dir=$USR_DIR \
  --data_dir=$DDIR \
  --problem=$PROBLEM \
  --output_dir=$MDIR \
  --hparams_set=$HPARAMS \
  --model=$MODEL \
  --decode_hparams="beam_size=$BEAM_SIZE,alpha=$ALPHA" \
  --decode_from_file=$DDIR/dev.en \
  --decode_to_file=$RDIR/dev_translation.sparql

t2t-decoder \
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
python3 generate.py $DDIR $RDIR