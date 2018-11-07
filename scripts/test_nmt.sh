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

python3 -m nmt.nmt.nmt \
    --src=en --tgt=sparql \
    --out_dir=$MDIR \
    --vocab_prefix=$DDIR/vocab \
    --inference_input_file=$DDIR/dev.en \
    --inference_output_file=$RDIR/dev_translation.sparql \
    --inference_ref_file=$DDIR/dev.sparql

python3 -m nmt.nmt.nmt \
    --src=en --tgt=sparql \
    --out_dir=$MDIR \
    --vocab_prefix=$DDIR/vocab \
    --inference_input_file=$DDIR/test.en \
    --inference_output_file=$RDIR/test_translation.sparql \
    --inference_ref_file=$DDIR/test.sparql

# Query and Analyze
python3 generate.py $DDIR $RDIR