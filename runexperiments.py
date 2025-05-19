import subprocess
from multiprocessing import Pool

# script_name = 'speech_iter.py'
# experiments = [
#     {'dataset': 'Neetha', 'nbtraintrials': 10, 'predtriallen': 20, 'shuffle': True},
#     {'dataset': 'Neetha', 'nbtraintrials': 20, 'predtriallen': 20, 'shuffle': True},
#     {'dataset': 'Neetha', 'nbtraintrials': 30, 'predtriallen': 20, 'shuffle': True},
#     {'dataset': 'Neetha', 'nbtraintrials': 40, 'predtriallen': 20, 'shuffle': True},
#     # {'dataset': 'fuglsang2018', 'nbtraintrials': 10},
#     # {'dataset': 'fuglsang2018', 'nbtraintrials': 20},
#     # {'dataset': 'fuglsang2018', 'nbtraintrials': 30},
#     # {'dataset': 'fuglsang2018', 'nbtraintrials': 40},
#     # {'dataset': 'earEEG', 'nbtraintrials': 1, 'predtriallen': 30, 'updatewinlen': 60},
#     # {'dataset': 'earEEG', 'nbtraintrials': 2, 'predtriallen': 30, 'updatewinlen': 60},
#     # {'dataset': 'earEEG', 'nbtraintrials': 3, 'predtriallen': 30, 'updatewinlen': 60},
#     # {'dataset': 'earEEG', 'nbtraintrials': 4, 'predtriallen': 30, 'updatewinlen': 60},
# ]

# script_name = 'unsupervised_single_enc.py'
# experiments = [
#     {'flippct': 1.0, 'bootstrap': True, 'svad': True, 'lwcov': True, 'mixpair': False, 'randinit': False},
#     {'flippct': 1.0, 'bootstrap': True, 'svad': True, 'lwcov': True, 'mixpair': True, 'randinit': False},
#     {'flippct': 1.0, 'bootstrap': True, 'svad': True, 'lwcov': True, 'mixpair': False, 'randinit': True},
#     {'flippct': 1.0, 'bootstrap': True, 'svad': True, 'lwcov': True, 'mixpair': True, 'randinit': True},
# ]

# script_name = 'unsupervised_speech.py'
# experiments = [
#     # {'dataset': 'fuglsang2018', 'nbtraintrials': 20, 'lwcov': True, 'flippct': 1.0, 'randinit': False},
#     # {'dataset': 'fuglsang2018', 'nbtraintrials': 20, 'lwcov': True, 'flippct': 1.0, 'randinit': True},
#     # {'dataset': 'fuglsang2018', 'nbtraintrials': 20, 'lwcov': True, 'flippct': 1.0, 'randinit': True, 'unbiased': True},
#     # {'dataset': 'earEEG', 'nbtraintrials': 2, 'lwcov': True, 'flippct': 1.0, 'randinit': False},
#     # {'dataset': 'earEEG', 'nbtraintrials': 1, 'lwcov': True, 'flippct': 1.0, 'randinit': True},
#     # {'dataset': 'earEEG', 'nbtraintrials': 2, 'lwcov': True, 'flippct': 1.0, 'randinit': True},
#     # {'dataset': 'earEEG', 'nbtraintrials': 1, 'lwcov': True, 'flippct': 1.0, 'randinit': True, 'unbiased': True},
#     # {'dataset': 'earEEG', 'nbtraintrials': 2, 'lwcov': True, 'flippct': 1.0, 'randinit': True, 'unbiased': True},
#     # {'dataset': 'Neetha', 'nbtraintrials': 20, 'lwcov': True, 'flippct': 1.0, 'randinit': False},
#     # {'dataset': 'Neetha', 'nbtraintrials': 20, 'lwcov': True, 'flippct': 1.0, 'randinit': True},
#     # {'dataset': 'Neetha', 'nbtraintrials': 20, 'lwcov': True, 'flippct': 1.0, 'randinit': True, 'unbiased': True},

#     {'dataset': 'Neetha', 'folds': 3, 'slidingwin': True, 'compete_resolu': 30, 'hparadata': [6,5], 'hparafeats': [1,0], 'evalpara': [1,1], 'poolsize': 19, 'initwtrainedmodel': True},
#     {'dataset': 'Neetha', 'folds': 3, 'slidingwin': True, 'compete_resolu': 30, 'hparadata': [6,5], 'hparafeats': [1,0], 'evalpara': [1,1], 'poolsize': 19},
#     {'dataset': 'Neetha', 'folds': 3, 'recursive': True, 'compete_resolu': 30, 'hparadata': [6,5], 'hparafeats': [1,0], 'evalpara': [1,1], 'weightpara': [0.9, 0.9], 'initwtrainedmodel': True},
#     {'dataset': 'Neetha', 'folds': 3, 'recursive': True, 'compete_resolu': 30, 'hparadata': [6,5], 'hparafeats': [1,0], 'evalpara': [1,1], 'weightpara': [0.9, 0.9]},
#     # {'dataset': 'fuglsang2018', 'folds': 5, 'slidingwin': True, 'compete_resolu': 25, 'hparadata': [6,5], 'hparafeats': [9,0], 'evalpara': [3,2], 'poolsize': 23, 'seeds': [2]},
#     # {'dataset': 'fuglsang2018', 'folds': 5, 'recursive': True, 'compete_resolu': 25, 'hparadata': [6,5], 'hparafeats': [9,0], 'evalpara': [3,2], 'weightpara': [0.916, 0.916], 'seeds': [2]},
#     # {'dataset': 'earEEG', 'folds': 5, 'slidingwin': True, 'compete_resolu': 30, 'hparadata': [4,3], 'hparafeats': [6,0], 'evalpara': [3,2], 'poolsize': 19, 'seeds': [2]},
#     # {'dataset': 'earEEG', 'folds': 5, 'recursive': True, 'compete_resolu': 30, 'hparadata': [4,3], 'hparafeats': [6,0], 'evalpara': [3,2], 'weightpara': [0.9, 0.9], 'seeds': [2]},
# ]

script_name = 'speech_stream.py'
experiments = [
    {'dataset': 'Neetha', 'paratrans': True, 'method': 'all', 'predtriallen': 30, 'updatestep': 1},
    {'dataset': 'Neetha', 'paratrans': True, 'method': 'all', 'predtriallen': 15, 'updatestep': 2},
    {'dataset': 'fuglsang2018', 'paratrans': True, 'method': 'all', 'predtriallen': 25, 'updatestep': 1},
    {'dataset': 'earEEG', 'paratrans': True, 'method': 'all', 'predtriallen': 30, 'updatestep': 1},
    {'dataset': 'earEEG', 'paratrans': True, 'method': 'all', 'predtriallen': 15, 'updatestep': 2},
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
