import subprocess
from multiprocessing import Pool

dataset = 'fuglsang2018'
nb_train_trials_options = [6, 12, 18, 24, 30, 36, 42]
methods = ['single', 'single_warminit', 'two', 'bpsk_local', 'single_unbiased']

script_name = 'speech_iter.py'

experiments = []
for nbtraintrials in nb_train_trials_options:
    for method in methods:
        experiments.append({
            'dataset': dataset,
            'nbtraintrials': nbtraintrials,
            'method': method,
            'shuffle': True
        })

# Function to run a single experiment
def run_experiment(arg_set):
    cmd = ['python', script_name]
    for key, value in arg_set.items():
        if isinstance(value, bool):
            if value:
                cmd.append(f'--{key}')
        elif isinstance(value, list):
            cmd.append(f'--{key}')
            cmd.extend(map(str, value))
        else:
            cmd.append(f'--{key}={value}')
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd)

if __name__ == "__main__":
    # You can change this to the number of parallel processes you want
    with Pool(processes=1) as pool:
        pool.map(run_experiment, experiments)
