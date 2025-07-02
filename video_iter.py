import os
import utils
import utils_iter
import utils_prob
import argparse
import numpy as np
import pickle

def get_video_sequences(eeg_folder):
    eeg_files_all = [file for file in os.listdir(eeg_folder) if file.endswith('.set')]
    files = [file for file in eeg_files_all if len(file.split('_')) == 3]
    files.sort()
    return files

def data_per_subj(eeg_folder, feats_path_folder, fsStim, bads, trial_len=60):
    files = get_video_sequences(eeg_folder)
    eeg_list = []
    feats_list = []
    label_list = []
    for file in files:
        eeg_downsampled, eog_downsampled, fs = utils.get_eeg_eog(eeg_folder + file, fsStim, bads, expdim=False)
        eeg_reg = utils.regress_out(eeg_downsampled.T, eog_downsampled.T)
        len_video = eeg_reg.shape[0]
        name = file[:-4]
        id_att = name.split('_')[-1]
        ids = list(set(name.split('_')))
        f1 = utils.get_features(feats_path_folder, ids[0], len_video, offset=122*fs, smooth=True)
        f2 = utils.get_features(feats_path_folder, ids[1], len_video, offset=122*fs, smooth=True)
        feats = np.stack((f1[:,8], f2[:,8]), axis=1)
        label = 1 if id_att == ids[0] else 2
        eeg_list.append(eeg_reg)
        feats_list.append(feats)
        label_list.append(label)
    return eeg_list, feats_list, label_list

subjects = ['Pilot_1', 'Pilot_2', 'Pilot_4', 'Pilot_5', 'Pilot_6', 'Pilot_7', 'Pilot_8', 'Pilot_9', 'Pilot_10', 'Pilot_11', 'Pilot_12', 'Pilot_13', 'Pilot_14', 'Pilot_15', 'Pilot_17', 'Pilot_18', 'Pilot_19', 'Pilot_20', 'Pilot_21']
# bads = [['A30', 'B25'], ['B25'], ['B25'], [], ['A31', 'B31'], ['B25'], ['A30', 'B25'], ['A30', 'B25'], ['B25'], ['B25', 'B26'], ['A30', 'B25'], ['B31'], ['B25', 'A23'], ['A30', 'B25'], ['B25'], ['B25'], ['A30', 'B25'], ['A30', 'B25'], ['B25']] 
# eeg_dict = {}
# feats_dict = {}
# labels_dict = {}
# for subject, bad in zip(subjects, bads):
#     eeg_folder = f"../../Experiments/data/Two_Obj/Overlay/{subject}/"
#     feats_path_folder = '../Feat_Multi/features/'
#     fsStim = 30
#     eeg_trials, feats_trials, label_trials = data_per_subj(eeg_folder, feats_path_folder, fsStim, bad)
#     eeg_dict[subject] = eeg_trials
#     feats_dict[subject] = feats_trials
#     labels_dict[subject] = label_trials

# data_path = 'data/'
# if not os.path.exists(data_path):
#     os.makedirs(data_path)

# with open(data_path + 'eeg_video.pkl', 'wb') as f:
#     pickle.dump(eeg_dict, f)
# with open(data_path + 'feats_video.pkl', 'wb') as f:
#     pickle.dump(feats_dict, f)
# with open(data_path + 'labels_video.pkl', 'wb') as f:
#     pickle.dump(labels_dict, f)

# load data
with open('data/eeg_video.pkl', 'rb') as f:
    eeg_dict = pickle.load(f)
with open('data/feats_video.pkl', 'rb') as f:
    feats_dict = pickle.load(f)
with open('data/labels_video.pkl', 'rb') as f:
    labels_dict = pickle.load(f)

argparser = argparse.ArgumentParser()
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
argparser.add_argument('--seeds', type=int, nargs='+', default=[1, 2, 4, 8, 16], help='Random seeds')
args = argparser.parse_args()

latent_dimensions = 5
fs = 30
method = args.method
hparadata = [3, 1] if args.hparadata is None else args.hparadata
hparafeats = [15, 0] if args.hparafeats is None else args.hparafeats
evalpara = [3, 2] if args.evalpara is None else args.evalpara
nb_disconnected = args.nbdisconnected
trial_len = 60
pred_len = 60 if args.predtriallen is None else args.predtriallen
update_len = trial_len if args.updatewinlen is None else args.updatewinlen
SHUFFLE = args.shuffle
ITERS = 8

# files = get_video_sequences(rf"C:\Users\yyao\Documents\Experiments\data\SVAD\SUB_01")
files = get_video_sequences(rf"C:\Users\Gebruiker\Documents\Experiments\data\SUB_01")
# video pairs for testing
pairs_test = ['03_05', '06_16', '08_15', '01_13', '02_12', '04_09', '07_14']

for SEED in args.seeds:
    print(f"Running with seed: {SEED}")
    # selected_subjects = ['Pilot_1', 'Pilot_5', 'Pilot_6', 'Pilot_10', 'Pilot_11', 'Pilot_12', 'Pilot_13', 'Pilot_14', 'Pilot_15', 'Pilot_17', 'Pilot_19', 'Pilot_20', 'Pilot_21']
    selected_subjects = subjects

    acc_sup_all = []
    acc_unsup_all = []

    for subj in selected_subjects:
        data_videos = eeg_dict[subj]
        feats_videos = feats_dict[subj]
        labels_videos = labels_dict[subj]

        true_labels = []
        pred_sup = []
        pred_unsup = []

        for pair in pairs_test:
            pairs_mask = [pair in file for file in files]
            test_idx = [i for i, mask in enumerate(pairs_mask) if mask]
            train_idx = [i for i, mask in enumerate(pairs_mask) if not mask]
            data_train_trials = []
            feats_train_trials = []
            labels_train_trials = []
            for i in train_idx:
                data_video_trials = utils.into_trials(data_videos[i], fs, update_len)
                data_video_trials = [utils_iter.process_data_per_view(view, hparadata[0], hparadata[1]) for view in data_video_trials]
                data_train_trials += data_video_trials
                feats_video_trials = utils.into_trials(feats_videos[i], fs, update_len)
                feats_video_trials = [utils_iter.process_data_per_view(view, hparafeats[0], hparafeats[1]) for view in feats_video_trials]
                feats_train_trials += feats_video_trials
                labels_train_trials += [labels_videos[i]] * len(data_video_trials)
            data_test_trials = []
            feats_test_trials = []
            labels_test_trials = []
            for i in test_idx:
                data_video_trials = utils.into_trials_with_overlap(data_videos[i], fs, update_len, overlap=0.9)
                data_video_trials = [utils_iter.process_data_per_view(view, hparadata[0], hparadata[1]) for view in data_video_trials]
                data_test_trials += data_video_trials
                feats_video_trials = utils.into_trials_with_overlap(feats_videos[i], fs, update_len, overlap=0.9)
                feats_video_trials = [utils_iter.process_data_per_view(view, hparafeats[0], hparafeats[1]) for view in feats_video_trials]
                feats_test_trials += feats_video_trials
                labels_test_trials += [labels_videos[i]] * len(data_video_trials)
            if args.nbtraintrials is not None:
                data_train_trials = data_train_trials[:args.nbtraintrials]
                feats_train_trials = feats_train_trials[:args.nbtraintrials]
                labels_train_trials = labels_train_trials[:args.nbtraintrials]
            if update_len != trial_len:
                data_train_trials, feats_train_trials, labels_train_trials = utils_iter.further_split_and_shuffle(data_train_trials, feats_train_trials, labels_train_trials, update_len, fs, SHUFFLE=SHUFFLE, SEED=SEED)
            if pred_len != trial_len:
                data_test_trials, feats_test_trials, labels_test_trials = utils_iter.further_split_and_shuffle(data_test_trials, feats_test_trials, labels_test_trials, pred_len, fs, SHUFFLE=SHUFFLE, SEED=SEED)
            iteration = utils_iter.ITERATIVE(data_train_trials, data_test_trials, feats_train_trials, feats_test_trials, labels_test_trials, hparadata[0], hparafeats[0], latent_dimensions, SEED, evalpara, ITERS=ITERS)
            true_labels.append(np.tile(np.array(labels_test_trials), (ITERS+1, 1)))
            pred_labels_sup = iteration.supervised(labels_train_trials)
            pred_sup.append(pred_labels_sup)
            if method == 'single':
                pred_labels_single = iteration.unsupervised(SINGLEENC=True)
                pred_unsup.append(pred_labels_single)
            elif method == 'single_warminit':
                pred_labels_single_warminit = iteration.unsupervised(SINGLEENC=True, WARMINIT=True)
                pred_unsup.append(pred_labels_single_warminit)
            elif method == 'discriminative':
                pred_labels_discriminative = iteration.discriminative()
                pred_unsup.append(pred_labels_discriminative)
            elif method == 'two':
                pred_labels_two = iteration.unsupervised(SINGLEENC=False)
                pred_unsup.append(pred_labels_two)
            elif method == 'single_unbiased':
                pred_labels_single_unbiased = iteration.unbiased(SINGLEENC=True)
                pred_unsup.append(pred_labels_single_unbiased)
            elif method == 'two_unbiased':
                pred_labels_two_unbiased = iteration.unbiased(SINGLEENC=False)
                pred_unsup.append(pred_labels_two_unbiased)
            elif method == 'bpsk':
                pred_labels_bpsk = iteration.soft_bpsk(GLOBAL=True)
                pred_unsup.append(pred_labels_bpsk)
            elif method == 'bpsk_local':
                pred_labels_bpsk_local = iteration.soft_bpsk(GLOBAL=False)
                pred_unsup.append(pred_labels_bpsk_local)
            else:
                raise ValueError(f"Unknown method: {method}")
        true_labels = np.concatenate(true_labels, axis=1)
        pred_sup = np.concatenate(pred_sup)
        pred_unsup = np.concatenate(pred_unsup, axis=1)
        acc_sup = np.sum(np.array(pred_sup) == np.array(true_labels[0,:])) / true_labels.shape[1]
        acc_unsup = np.sum(np.array(pred_unsup) == np.array(true_labels), axis=1) / true_labels.shape[1]
        print(f"Subject: {subj}, Acc supervised: {acc_sup:.2f}, Acc unsupervised: {acc_unsup}")
        acc_sup_all.append(acc_sup)
        acc_unsup_all.append(acc_unsup)
    acc_sup_all = np.array(acc_sup_all)
    acc_unsup_all = np.array(acc_unsup_all)
    
    folder_path = f'tables/EEG-EOG/iter/'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    file_name = f"{folder_path}nbtraintrials{args.nbtraintrials}_nbdisconnected{nb_disconnected}_updatelen{update_len}_predlen{pred_len}_hparadata{hparadata[0]}_{hparadata[1]}_hparafeats{hparafeats[0]}_{hparafeats[1]}_evalpara{evalpara[0]}_{evalpara[1]}_SHUFFLE{SHUFFLE}_SEED{SEED}.pkl"
    if os.path.exists(file_name):
        with open(file_name, 'rb') as f:
            res_dict = pickle.load(f)
    else:
        res_dict = {}
    res_dict['acc_sup'] = acc_sup_all
    res_dict[f'acc_{method}'] = acc_unsup_all

    with open(file_name, 'wb') as f:
        pickle.dump(res_dict, f)