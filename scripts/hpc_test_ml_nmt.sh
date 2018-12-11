#!/bin/bash
#SBATCH -J "hpc_ml_nmt"   # job name
#SBATCH -A p_adm # name of the project
#SBATCH --mail-type=ALL
#SBATCH --mail-user xiaoyu.yin@mailbox.tu-dresden.de
#SBATCH --time=8:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --partition=ml
#SBATCH -o "test_ml_nmt-%j.out"   # output name
#SBATCH --mem-per-cpu=6000   # minimum amount of real memory

ANACONDA3_INSTALL_PATH='/opt/anaconda3'
export PATH=$ANACONDA3_INSTALL_PATH/bin:$PATH
source /opt/DL/tensorflow/bin/tensorflow-activate

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

srun python3 -m nmt.nmt.nmt \
    --src=en --tgt=sparql \
    --out_dir=$MDIR \
    --vocab_prefix=$DDIR/vocab \
    --inference_input_file=$DDIR/dev.en \
    --inference_output_file=$RDIR/dev_translation.sparql \
    --inference_ref_file=$DDIR/dev.sparql

srun python3 -m nmt.nmt.nmt \
    --src=en --tgt=sparql \
    --out_dir=$MDIR \
    --vocab_prefix=$DDIR/vocab \
    --inference_input_file=$DDIR/test.en \
    --inference_output_file=$RDIR/test_translation.sparql \
    --inference_ref_file=$DDIR/test.sparql

# Query and Analyze
# srun python3 generate.py $DDIR $RDIR