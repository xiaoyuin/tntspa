import subprocess
import os
import sys
import argparse

# This python script is made only for running experiments on High-Performance Computing Server of TU Dresden

parser = argparse.ArgumentParser()
parser.add_argument('data_directory')
parser.add_argument('models_diretory')
args = parser.parse_args()

data_directory = args.data_directory

models_diretory = args.models_diretory

print("The available training scripts are:")
training_scripts = sorted([file for file in os.listdir('./scripts') if file.startswith('hpc_train_')])
for i, s in enumerate(training_scripts):
    print(i, '->', s)
print('Type job ids to submit into hpc queue: ')
job_ids = list(map(str, input().split()))
for i in job_ids:
    subprocess.run(["sbatch", "./scripts/"+training_scripts[i], data_directory, models_diretory])
