#!/bin/bash
#SBATCH -J "hpc_ml_fairseq"   # job name
#SBATCH -A p_adm # name of the project
#SBATCH --mail-type=ALL
#SBATCH --mail-user xiaoyu.yin@mailbox.tu-dresden.de
#SBATCH --time=8:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --partition=ml
#SBATCH -o "test_ml_fairseq-%j.out"   # output name
#SBATCH --mem-per-cpu=6000  # minimum amount of real memory

ANACONDA3_INSTALL_PATH='/opt/anaconda3'
export PATH=$ANACONDA3_INSTALL_PATH/bin:$PATH
source /opt/DL/pytorch/bin/pytorch-activate

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

srun python3 fairseq/generate.py $DDIR/fairseq-data-bin \
--gen-subset valid \
--path $MDIR/checkpoint_best.pt \
--beam 5 > $RDIR/dev_output.txt

srun python3 fairseq_output_reader.py $RDIR/dev_output.txt > $RDIR/dev_translation.sparql

srun python3 fairseq/generate.py $DDIR/fairseq-data-bin \
--gen-subset test \
--path $MDIR/checkpoint_best.pt \
--beam 5 > $RDIR/test_output.txt

srun python3 fairseq_output_reader.py $RDIR/test_output.txt > $RDIR/test_translation.sparql

# Query and Analyze
# srun python3 generate.py $DDIR $RDIR