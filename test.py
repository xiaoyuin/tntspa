import os
import argparse
import subprocess

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_directory')
    parser.add_argument('model_diretory')
    parser.add_argument('result_directory')
    args = parser.parse_args()

    data_directory = args.data_directory
    model_diretory = args.model_diretory
    result_directory = args.result_directory

    print("The available testing scripts are:")
    testing_scripts = sorted([file for file in os.listdir('./scripts') if file.startswith('test_') or file.startswith('hpc_test_')])
    for i, s in enumerate(testing_scripts):
        print(i, '->', s)
    print('Type a job id to run: ')
    job_id = int(input())
    runned_script = testing_scripts[job_id]
    subprocess.run(["sbatch" if runned_script.startswith('hpc_') else "sh", "./scripts/"+runned_script, data_directory, model_diretory, result_directory])
