import os
import utils
import utils_iter
import utils_prob
import argparse
import copy
import numpy as np
import pickle
from scipy.io import loadmat
from sklearn.model_selection import KFold

def data_per_subj(Subj_ID, data_folder):
    file_path = f'{data_folder}/dataSubject{Subj_ID}.mat'
    data_dict = loadmat(file_path, squeeze_me=True)
    if 'earEEG' in data_folder:
        eeg_trials = [trial[:,:29] for trial in data_dict['eegTrials']]
    else:
        eeg_trials = data_dict['eegTrials']
    feats_trials = data_dict['audioTrials']
    label_trials = data_dict['attSpeaker']
    nb_trials = len(eeg_trials)
    eeg_trials = [eeg_trials[i] for i in range(nb_trials)]
    feats_trials = [feats_trials[i] for i in range(nb_trials)]
    label_trials = [label_trials[i] for i in range(nb_trials)]
    return eeg_trials, feats_trials, label_trials

def prepare_fold_per_view(trials, hpara, n_folds, fold_idx, SEED=None, SHUFFLE=False, PROCESS=True):
    if not SHUFFLE:
        SEED = None
    kf = KFold(n_splits=n_folds, shuffle=SHUFFLE, random_state=SEED)
    train_index, test_index = list(kf.split(trials))[fold_idx]
    if PROCESS:
        train_trials = [utils_iter.process_data_per_view(trials[i], hpara[0], hpara[1], NORMALIZE=True) for i in train_index]
        test_trials = [utils_iter.process_data_per_view(trials[i], hpara[0], hpara[1], NORMALIZE=True) for i in test_index]
    else:
        train_trials = [trials[i] for i in train_index]
        test_trials = [trials[i] for i in test_index]
    return train_trials, test_trials

argparser = argparse.ArgumentParser()
argparser.add_argument('--dataset', type=str, help='The name of the dataset')
argparser.add_argument('--method', type=str, help='The method to be used')
argparser.add_argument('--nbdisconnected', type=int, default=0, help='Number of disconnected channels')
argparser.add_argument('--predtriallen', type=int, help='The length of the prediction trials')
argparser.add_argument('--updatewinlen', type=int, help='The length of the update windows')
argparser.add_argument('--nbtraintrials', type=int, help='The number of trials to be used in training')
argparser.add_argument('--hparadata', type=int, nargs='+', help='Parameters of the hankel matrix for the data')
argparser.add_argument('--hparafeats', type=int, nargs='+', help='Parameters of the hankel matrix for the features')
argparser.add_argument('--evalpara', type=int, nargs='+', help='Parameters (range_into_account, nb_comp_into_account) for the evaluation')
argparser.add_argument('--shuffle', action='store_true', default=False, help='Shuffle the trials')
# argparser.add_argument('--labelnoise', type=float, help='Percentage of wrong labels')
argparser.add_argument('--seeds', type=int, nargs='+', default=[16, 32, 64, 128], help='Random seeds')
args = argparser.parse_args()

if args.dataset == 'Neetha':
    fs = 20
    trial_len = 60
    hparadata = [4, 3] if args.hparadata is None else args.hparadata
    hparafeats = [6, 0] if args.hparafeats is None else args.hparafeats
    evalpara = [3, 2] if args.evalpara is None else args.evalpara
    pred_len = 30 if args.predtriallen is None else args.predtriallen
    update_len = trial_len if args.updatewinlen is None else args.updatewinlen
    leave_out_persubj = 12
elif args.dataset == 'fuglsang2018':
    fs = 32
    trial_len = 50
    hparadata = [4, 3] if args.hparadata is None else args.hparadata
    hparafeats = [9, 0] if args.hparafeats is None else args.hparafeats
    evalpara = [3, 2] if args.evalpara is None else args.evalpara
    pred_len = 25 if args.predtriallen is None else args.predtriallen
    update_len = trial_len if args.updatewinlen is None else args.updatewinlen
    leave_out_persubj = 12
elif args.dataset == 'earEEG':
    fs = 20
    trial_len = 600
    hparadata = [4, 3] if args.hparadata is None else args.hparadata
    hparafeats = [6, 0] if args.hparafeats is None else args.hparafeats
    evalpara = [3, 2] if args.evalpara is None else args.evalpara
    pred_len = 30 if args.predtriallen is None else args.predtriallen
    update_len = 60  if args.updatewinlen is None else args.updatewinlen
    leave_out_persubj = 2

method = args.method
SHUFFLE = args.shuffle
nb_disconnected = args.nbdisconnected
latent_dimensions = 5
nb_folds = 3
ITERS = 6
data_folder = rf'C:\Users\Gebruiker\Documents\Experiments\data\{args.dataset}'
files = [f for f in os.listdir(data_folder) if f.endswith('.mat')]
subjects = [int(''.join(filter(str.isdigit, f))) for f in files]
eeg_dict = {}
feats_dict = {}
labels_dict = {}
for subject in subjects:
    eeg_trials, feats_trials, label_trials = data_per_subj(subject, data_folder)
    eeg_dict[subject] = eeg_trials
    feats_dict[subject] = feats_trials
    labels_dict[subject] = label_trials

for SEED in args.seeds:
    print(f"Running with seed: {SEED}")
    selected_subjects = subjects

    acc_sup_all = []
    # acc_sup_disc_all = []
    acc_unsup_all = []
    acc_train_all = []
    cpu_time_all = []

    for subject in selected_subjects:
        eeg_trials = eeg_dict[subject]
        feats_trials = feats_dict[subject]
        labels_trials = labels_dict[subject]

        true_labels_train = []
        true_labels = []
        cpu_times = []
        pred_sup = []
        # pred_sup_disc = []
        pred_unsup = []
        repred = []

        for fold_idx in range(nb_folds):
            data_train_trials, data_test_trials = prepare_fold_per_view(eeg_trials, hparadata, nb_folds, fold_idx, SHUFFLE=SHUFFLE, SEED=SEED)
            feats_train_trials, feats_test_trials = prepare_fold_per_view(feats_trials, hparafeats, nb_folds, fold_idx, SHUFFLE=SHUFFLE, SEED=SEED)
            labels_train_trials, labels_test_trials = prepare_fold_per_view(labels_trials, hparafeats, nb_folds, fold_idx, PROCESS=False, SHUFFLE=SHUFFLE, SEED=SEED)
            if args.nbtraintrials is not None:
                data_train_trials = data_train_trials[:args.nbtraintrials]
                feats_train_trials = feats_train_trials[:args.nbtraintrials]
                labels_train_trials = labels_train_trials[:args.nbtraintrials]
            if update_len != trial_len:
                data_train_trials, feats_train_trials, labels_train_trials = utils_iter.further_split_and_shuffle(data_train_trials, feats_train_trials, labels_train_trials, update_len, fs, SHUFFLE=SHUFFLE, SEED=SEED)
            data_test_trials, feats_test_trials, labels_test_trials = utils_iter.further_split_and_shuffle(data_test_trials, feats_test_trials, labels_test_trials, pred_len, fs, SHUFFLE=SHUFFLE, SEED=SEED)
            iteration = utils_iter.ITERATIVE(data_train_trials, data_test_trials, feats_train_trials, feats_test_trials, labels_test_trials, hparadata[0], hparafeats[0], latent_dimensions, SEED, evalpara, ITERS=ITERS)
            true_labels_train.append(np.array(labels_train_trials))
            true_labels.append(np.tile(np.array(labels_test_trials), (ITERS+1, 1)))
            pred_labels_sup = iteration.supervised(labels_train_trials)
            pred_sup.append(pred_labels_sup)
            # pred_labels_supdisc = iteration.supervised(labels_train_trials, DISCRIMINATIVE=True)
            # pred_sup_disc.append(pred_labels_supdisc)
            if method == 'single':
                pred_labels_unsup, repred_labels, cpu_time = iteration.unsupervised(SINGLEENC=True)
            elif method == 'single_warminit':
                pred_labels_unsup, repred_labels, cpu_time = iteration.unsupervised(SINGLEENC=True, WARMINIT=True)
            elif method == 'discriminative':
                pred_labels_unsup, repred_labels, cpu_time = iteration.discriminative()
            elif method == 'two':
                pred_labels_unsup, repred_labels, cpu_time = iteration.unsupervised(SINGLEENC=False)
            elif method == 'single_unbiased':
                pred_labels_unsup, repred_labels, cpu_time = iteration.unbiased(SINGLEENC=True)
            elif method == 'two_unbiased':
                pred_labels_unsup, repred_labels, cpu_time = iteration.unbiased(SINGLEENC=False)
            elif method == 'bpsk':
                pred_labels_unsup, repred_labels, cpu_time = iteration.soft_bpsk(GLOBAL=True)
            elif method == 'bpsk_local':
                pred_labels_unsup, repred_labels, cpu_time = iteration.soft_bpsk(GLOBAL=False)
            else:
                raise ValueError(f"Unknown method: {method}")
            pred_unsup.append(pred_labels_unsup)
            repred.append(repred_labels)
            cpu_times.append(cpu_time)
        true_labels_train = np.concatenate(true_labels_train)
        true_labels = np.concatenate(true_labels, axis=1)
        pred_sup = np.concatenate(pred_sup)
        # pred_sup_disc = np.concatenate(pred_sup_disc)
        pred_unsup = np.concatenate(pred_unsup, axis=1)
        repred = np.concatenate(repred)
        cpu_times = np.array(cpu_times)
        acc_sup = np.sum(pred_sup == true_labels[0,:]) / true_labels.shape[1]
        # acc_sup_disc = np.sum(pred_sup_disc == true_labels[0,:]) / true_labels.shape[1]
        acc_unsup = np.sum(pred_unsup == true_labels, axis=1) / true_labels.shape[1]
        acc_train = np.sum(repred == true_labels_train) / len(true_labels_train)
        print(f"Subject: {subject}, Acc supervised: {acc_sup:.2f}, Acc unsupervised: {acc_unsup}, Acc unsupervised (Train): {acc_train:.2f}, CPU time: {np.mean(cpu_times):.2f} seconds")
        acc_sup_all.append(acc_sup)
        # acc_sup_disc_all.append(acc_sup_disc)
        acc_unsup_all.append(acc_unsup)
        acc_train_all.append(acc_train)
        cpu_time_all.append(np.mean(cpu_times))
    acc_sup_all = np.array(acc_sup_all)
    # acc_sup_disc_all = np.array(acc_sup_disc_all)
    acc_unsup_all = np.array(acc_unsup_all)
    acc_train_all = np.array(acc_train_all)
    cpu_time_all = np.array(cpu_time_all)
    
    folder_path = f'tables/{args.dataset}/iter/'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    file_name = f"{folder_path}nbtraintrials{args.nbtraintrials}_nbdisconnected{nb_disconnected}_updatelen{update_len}_predlen{pred_len}_hparadata{hparadata[0]}_{hparadata[1]}_hparafeats{hparafeats[0]}_{hparafeats[1]}_evalpara{evalpara[0]}_{evalpara[1]}_SHUFFLE{SHUFFLE}_SEED{SEED}.pkl"
    if os.path.exists(file_name):
        with open(file_name, 'rb') as f:
            res_dict = pickle.load(f)
    else:
        res_dict = {}
    res_dict['acc_sup'] = acc_sup_all
    # res_dict['acc_sup_disc'] = acc_sup_disc_all
    res_dict[f'acc_{method}'] = acc_unsup_all
    res_dict[f'acctrain_{method}'] = acc_train_all
    res_dict[f'cpu_time_{method}'] = cpu_time_all

    with open(file_name, 'wb') as f:
        pickle.dump(res_dict, f)
