# Usage: sh build_dataset.sh $DATA_DIRECTORY $DATA_PREFIX
# Requirement: there exists "$DATA_PREFIX.en" and "$DATA_PREFIX.sparql" under $DATA_DIRECTORY

SPLIT=80/10/10

if [ -n "$3" ]
then SPLIT=$3
fi

# Split the data into 80/10/10 or given percentages
echo "Spliting the data into train/dev/test"
python3 split.py $1 --data_prefix $2 --split_rates $SPLIT

# Generate vocabulary files
echo "Generating vocabulary files"
python3 build_vocab.py $1/$2.en > $1/vocab.en
python3 build_vocab.py $1/$2.sparql > $1/vocab.sparql
python3 build_vocab.py $1/$2.en $1/$2.sparql > $1/vocab.shared

# Generate dataset binaries for fairseq framework
echo "Generating dataset binaries for fairseq framework"
sh _build_dataset_fairseq.sh $1

# Generate tensor2tensor data folder
echo "Generating tensor2tensor data folder"
USR_DIR=.
DATA_DIR=$1
TMP_DIR=/tmp/t2t_datagen
mkdir -p $1 $TMP_DIR

t2t-datagen \
  --t2t_usr_dir=$USR_DIR \
  --data_dir=$1 \
  --tmp_dir=$TMP_DIR \
  --problem=translate_ensparql

