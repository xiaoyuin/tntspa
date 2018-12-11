#!/bin/bash
#SBATCH --time=8:00:00   # walltime
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6  # number of processor cores (i.e. threads)
#SBATCH -p gpu2    # K80 GPUs
#SBATCH -J "hpc_gpu2_fairseq"   # job name
#SBATCH -o "test_gpu2_fairseq-%j.out"   # output name
#SBATCH --mem=40000   # minimum amount of real memory
#SBATCH -A p_adm # name of the project
#SBATCH --mail-user xiaoyu.yin@mailbox.tu-dresden.de
#SBATCH --mail-type ALL
module load TensorFlow/1.8.0-foss-2018a-Python-3.6.4-CUDA-9.2.88

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