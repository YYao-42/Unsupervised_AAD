import subprocess
from multiprocessing import Pool

script_name = 'speech_iter.py'
experiments = [
    {'dataset': 'Neetha', 'nbtraintrials': 40, 'method': 'single', 'shuffle': True},
    {'dataset': 'Neetha', 'nbtraintrials': 40, 'method': 'single_warminit', 'shuffle': True},
    {'dataset': 'Neetha', 'nbtraintrials': 40, 'method': 'two', 'shuffle': True},
    {'dataset': 'Neetha', 'nbtraintrials': 40, 'method': 'bpsk_local', 'shuffle': True},
    {'dataset': 'Neetha', 'nbtraintrials': 40, 'method': 'single_unbiased', 'shuffle': True},

    {'dataset': 'Neetha', 'nbtraintrials': 45, 'method': 'single', 'shuffle': True},
    {'dataset': 'Neetha', 'nbtraintrials': 45, 'method': 'single_warminit', 'shuffle': True},
    {'dataset': 'Neetha', 'nbtraintrials': 45, 'method': 'two', 'shuffle': True},
    {'dataset': 'Neetha', 'nbtraintrials': 45, 'method': 'bpsk_local', 'shuffle': True},
    {'dataset': 'Neetha', 'nbtraintrials': 45, 'method': 'single_unbiased', 'shuffle': True},
]


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
