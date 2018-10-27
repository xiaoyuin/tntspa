USR_DIR=.
PROBLEM=translate_ensparql
DATA_DIR=data/monument_600
TMP_DIR=/tmp/t2t_datagen
mkdir -p $DATA_DIR $TMP_DIR


t2t-datagen \
  --t2t_usr_dir=$USR_DIR \
  --data_dir=$DATA_DIR \
  --tmp_dir=$TMP_DIR \
  --problem=$PROBLEM

MODEL=transformer
HPARAMS=transformer_base_single_gpu
TRAIN_DIR=output/models/transformer_base_single_gpu

# Train
# *  If you run out of memory, add --hparams='batch_size=1024'.
t2t-trainer \
  --t2t_usr_dir=$USR_DIR \
  --data_dir=$DATA_DIR \
  --problem=$PROBLEM \
  --model=$MODEL \
  --hparams_set=$HPARAMS \
  --output_dir=$TRAIN_DIR