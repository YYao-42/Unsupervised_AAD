import os
import utils
import utils_stream
import utils_prob
import argparse
import numpy as np
import pickle

def data_per_subj(eeg_folder, feats_path_folder, fsStim, bads, trial_len=60):
    eeg_files_all = [file for file in os.listdir(eeg_folder) if file.endswith('.set')]
    files = [file for file in eeg_files_all if len(file.split('_')) == 3]
    files.sort()
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
argparser.add_argument('--nbdisconnected', type=int, default=0, help='Number of disconnected channels')
argparser.add_argument('--predtriallen', type=int, help='The length of the prediction trials')
argparser.add_argument('--nbtrialsused', type=int, help='The number of trials to be used in each dataset')
argparser.add_argument('--updatestep', type=int, help='The step size for the update')
argparser.add_argument('--nbestsubj', type=int, default=3, help='Number of the subjects to be used for estimating the distribution of att_uatt')
argparser.add_argument('--hparadata', type=int, nargs='+', help='Parameters of the hankel matrix for the data')
argparser.add_argument('--hparafeats', type=int, nargs='+', help='Parameters of the hankel matrix for the features')
argparser.add_argument('--evalpara', type=int, nargs='+', help='Parameters (range_into_account, nb_comp_into_account) for the evaluation')
argparser.add_argument('--weightpara', type=float, nargs='+', help='alpha and beta for the time-adaptive version')
argparser.add_argument('--paratrans', action='store_true', default=False, help='Whether to enable parameter transfer')
argparser.add_argument('--shuffle', action='store_true', default=False, help='Shuffle the trials')
argparser.add_argument('--seeds', type=int, nargs='+', default=[1, 2, 4, 8, 16, 32, 64, 128, 256, 512], help='Random seeds')
args = argparser.parse_args()

latent_dimensions = 5
fs = 30
hparadata = [3, 1] if args.hparadata is None else args.hparadata
hparafeats = [15, 0] if args.hparafeats is None else args.hparafeats
evalpara = [3, 2] if args.evalpara is None else args.evalpara
weightpara = [0.9, 0.9] if args.weightpara is None else args.weightpara
nb_calibsessions = 1
nb_disconnected = args.nbdisconnected
trial_len = 60
sub_trial_length = 60 if args.predtriallen is None else args.predtriallen
SHUFFLE = args.shuffle
PARATRANS = args.paratrans
UPDATE_STEP = 1 if args.updatestep is None else args.updatestep
update_len = sub_trial_length * UPDATE_STEP
data_len_persubj = 32 * trial_len if args.nbtrialsused is None else args.nbtrialsused * trial_len
nb_update_trials = data_len_persubj // update_len

for SEED in args.seeds:
    eeg_trials_dict = {}
    feats_trials_dict = {}
    labels_trials_dict = {}
    for subject in eeg_dict.keys():
        eeg_videos = eeg_dict[subject]
        feats_videos = feats_dict[subject]
        labels_videos = labels_dict[subject]
        eeg_trials_all = []
        feats_trials_all = []
        labels_trials_all = []
        for eeg, feats, label in zip(eeg_videos, feats_videos, labels_videos):
            eeg_trials = utils.into_trials(eeg, fs, t=trial_len)
            feats_trials = utils.into_trials(feats, fs, t=trial_len)
            labels_trials = [label] * len(eeg_trials)
            eeg_trials_all += eeg_trials
            feats_trials_all += feats_trials
            labels_trials_all += labels_trials
        eeg_trials_dict[subject] = eeg_trials_all
        feats_trials_dict[subject] = feats_trials_all
        labels_trials_dict[subject] = labels_trials_all

    selected_subjects = ['Pilot_1', 'Pilot_5', 'Pilot_6', 'Pilot_10', 'Pilot_11', 'Pilot_12', 'Pilot_13', 'Pilot_14', 'Pilot_15', 'Pilot_17', 'Pilot_19', 'Pilot_20', 'Pilot_21']
    # selected_subjects = subjects
    print(f"Running with seed: {SEED}")
    rng = np.random.default_rng(SEED)
    rng.shuffle(selected_subjects)

    est_subs = selected_subjects[:args.nbestsubj]
    est_corr_att_sum, est_corr_unatt_sum = utils_prob.estimate_distribution_corr(eeg_trials_dict, feats_trials_dict, labels_trials_dict, est_subs, fs, hparadata, hparafeats, leave_out_persubj=4, trial_len=60, range_into_account=evalpara[0], nb_comp_into_account=evalpara[1])
    gmm_0, gmm_1 = utils_prob.fit_gmm(est_corr_att_sum, est_corr_unatt_sum, n_components_per_class=1)
    selected_subjects = selected_subjects[args.nbestsubj:]

    data_subjects_dict = {}
    feats_subjects_dict = {}
    labels_subjects_dict = {}
    rng = np.random.RandomState(SEED)
    for subj in selected_subjects:
        data_trials = eeg_trials_dict[subj]
        if nb_disconnected > 0:
            disconnected_channels = rng.choice(data_trials[0].shape[1], nb_disconnected, replace=False)
            for i in range(len(data_trials)):
                data_trials[i][:, disconnected_channels] = 0
        data_trials = [utils_stream.process_data_per_view(d, hparadata[0], hparadata[1], NORMALIZE=True) for d in data_trials]
        feats_trials = feats_trials_dict[subj]
        feats_trials = [utils_stream.process_data_per_view(f, hparafeats[0], hparafeats[1], NORMALIZE=True) for f in feats_trials]
        labels_trials = labels_trials_dict[subj]
        if sub_trial_length is not None:
            data_trials, feats_trials, labels_trials = utils_stream.further_split_and_shuffle(data_trials, feats_trials, labels_trials, sub_trial_length, fs, SHUFFLE=SHUFFLE, SEED=SEED)
        data_subjects_dict[subj] = data_trials
        feats_subjects_dict[subj] = feats_trials
        labels_subjects_dict[subj] = labels_trials
    
    stream = utils_stream.STREAM(data_subjects_dict, feats_subjects_dict, hparadata[0], hparafeats[0], latent_dimensions, SEED, evalpara, nb_update_trials, UPDATE_STEP)
    true_labels = np.stack([v[:nb_update_trials*UPDATE_STEP] for v in labels_subjects_dict.values()], axis=0)
    print("#####Fixed Supervised#####")
    pred_labels_dict = stream.fixed_supervised(labels_subjects_dict, selected_subjects[:nb_calibsessions], selected_subjects[nb_calibsessions:])
    pred_labels_fixed = np.stack([v for v in pred_labels_dict.values()], axis=0)
    print("#####Adaptive Supervised#####")
    pred_labels_dict = stream.adaptive_supervised(labels_subjects_dict, weightpara, PARATRANS=PARATRANS)
    pred_labels_adapsup = np.stack([v for v in pred_labels_dict.values()], axis=0)
    print("#####Single-Enc#####")
    pred_labels_dict = stream.recursive(weightpara, PARATRANS=PARATRANS, SINGLEENC=True)
    pred_labels_single = np.stack([v for v in pred_labels_dict.values()], axis=0)
    print("#####Two-Enc#####")
    pred_labels_dict = stream.recursive(weightpara, PARATRANS=PARATRANS, SINGLEENC=False)
    pred_labels_two = np.stack([v for v in pred_labels_dict.values()], axis=0)
    print("#####Soft Single-Enc#####")
    pred_labels_dict = stream.recursive_soft(weightpara, gmm_0, gmm_1, PARATRANS=PARATRANS)
    pred_labels_soft = np.stack([v for v in pred_labels_dict.values()], axis=0)

    res_dict = {'true': true_labels, 'fixed': pred_labels_fixed, 'adapsup': pred_labels_adapsup, 'single': pred_labels_single, 'two': pred_labels_two, 'soft': pred_labels_soft}
    folder_path = f'tables/EEG-EOG/stream/'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    file_name = f"{folder_path}{'nbtrialsused'+str(args.nbtrialsused) if args.nbtrialsused is not None else ''}nbdisconnected{nb_disconnected}_predtriallen{sub_trial_length}_updatestep{UPDATE_STEP}_nbestsubj{args.nbestsubj}_hparadata{hparadata[0]}_{hparadata[1]}_hparafeats{hparafeats[0]}_{hparafeats[1]}_evalpara{evalpara[0]}_{evalpara[1]}_weightpara{weightpara[0]}_{weightpara[1]}_PARATRANS{PARATRANS}_SHUFFLE{SHUFFLE}_SEED{SEED}.pkl"
    with open(file_name, 'wb') as f:
        pickle.dump(res_dict, f)