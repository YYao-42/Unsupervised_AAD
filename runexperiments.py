import subprocess
from multiprocessing import Pool

script_name = 'unsupervised.py'
experiments = [
    # {'mod': 'GAZE_V', 'truelabelpct': 0.0, 'label_resolu': 60, 'track_resolu': 60, 'mm_resolu': 60},
    # {'mod': 'GAZE_V', 'truelabelpct': 0.25, 'label_resolu': 60, 'track_resolu': 60, 'mm_resolu': 60},
    {'mod': 'GAZE_V', 'truelabelpct': 0.5, 'label_resolu': 60, 'track_resolu': 60, 'mm_resolu': 60},
    # {'mod': 'GAZE_V', 'truelabelpct': 0.75, 'label_resolu': 60, 'track_resolu': 60, 'mm_resolu': 60},
    # {'mod': 'GAZE_V', 'truelabelpct': 1.0, 'label_resolu': 60, 'track_resolu': 60, 'mm_resolu': 60},
]

# Function to run a single experiment
def run_experiment(arg_set):
    cmd = ['python', script_name]
    for key, value in arg_set.items():
        if isinstance(value, bool):
            if value:
                cmd.append(f'--{key}')
        else:
            cmd.append(f'--{key}={value}')
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd)

if __name__ == "__main__":
    # You can change this to the number of parallel processes you want
    with Pool(processes=1) as pool:
        pool.map(run_experiment, experiments)
