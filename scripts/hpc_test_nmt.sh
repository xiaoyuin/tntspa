#!/bin/bash
#SBATCH --time=12:00:00   # walltime
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8  # number of processor cores (i.e. threads)
#SBATCH -p gpu1,gpu2    # K80 GPUs on Haswell node
#SBATCH -J "fairseq_test"   # job name
#SBATCH -o "test_fairseq-%j.out"   # output name
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
srun python3 generate.py $DDIR $RDIR