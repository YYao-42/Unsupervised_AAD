import os
import utils_stream
import utils_prob
import argparse
import numpy as np
import pickle
from scipy.io import loadmat


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


argparser = argparse.ArgumentParser()
argparser.add_argument('--dataset', type=str, help='The name of the dataset')
argparser.add_argument('--method', type=str, help='The method to be used')
argparser.add_argument('--nbdisconnected', type=int, default=0, help='Number of disconnected channels')
argparser.add_argument('--predtriallen', type=int, help='The length of the prediction trials')
argparser.add_argument('--nbtrialsused', type=int, help='The number of trials to be used in each dataset')
argparser.add_argument('--updatestep', type=int, default=2, help='The step size for the update')
argparser.add_argument('--nbestsubj', type=int, default=2, help='Number of the subjects to be used for estimating the distribution of att_uatt')
argparser.add_argument('--hparadata', type=int, nargs='+', help='Parameters of the hankel matrix for the data')
argparser.add_argument('--hparafeats', type=int, nargs='+', help='Parameters of the hankel matrix for the features')
argparser.add_argument('--evalpara', type=int, nargs='+', help='Parameters (range_into_account, nb_comp_into_account) for the evaluation')
argparser.add_argument('--weightpara', type=float, nargs='+', help='alpha and beta for the time-adaptive version')
argparser.add_argument('--paratrans', action='store_true', default=False, help='Whether to enable parameter transfer')
argparser.add_argument('--shuffle', action='store_true', default=False, help='Shuffle the trials')
argparser.add_argument('--labelnoise', type=float, help='Percentage of wrong labels')
argparser.add_argument('--seeds', type=int, nargs='+', default=[1, 2, 4, 8, 16, 32, 64, 128, 256, 512], help='Random seeds')
args = argparser.parse_args()


if args.dataset == 'Neetha':
    fs = 20
    trial_len = 60
    hparadata = [4, 3] if args.hparadata is None else args.hparadata
    hparafeats = [6, 0] if args.hparafeats is None else args.hparafeats
    evalpara = [3, 2] if args.evalpara is None else args.evalpara
    weightpara = [0.9, 0.9] if args.weightpara is None else args.weightpara
    sub_trial_length = 30 if args.predtriallen is None else args.predtriallen
    leave_out_persubj = 12
    data_len_persubj = 72 * trial_len if args.nbtrialsused is None else args.nbtrialsused * trial_len
elif args.dataset == 'fuglsang2018':
    fs = 32
    trial_len = 50
    hparadata = [4, 3] if args.hparadata is None else args.hparadata
    hparafeats = [9, 0] if args.hparafeats is None else args.hparafeats
    evalpara = [3, 2] if args.evalpara is None else args.evalpara
    weightpara = [0.916, 0.916] if args.weightpara is None else args.weightpara
    sub_trial_length = 25 if args.predtriallen is None else args.predtriallen
    leave_out_persubj = 12
    data_len_persubj = 60 * trial_len if args.nbtrialsused is None else args.nbtrialsused * trial_len
elif args.dataset == 'earEEG':
    fs = 20
    trial_len = 600
    hparadata = [4, 3] if args.hparadata is None else args.hparadata
    hparafeats = [6, 0] if args.hparafeats is None else args.hparafeats
    evalpara = [3, 2] if args.evalpara is None else args.evalpara
    weightpara = [0.9, 0.9] if args.weightpara is None else args.weightpara
    sub_trial_length = 30 if args.predtriallen is None else args.predtriallen
    leave_out_persubj = 2
    data_len_persubj = 6 * trial_len if args.nbtrialsused is None else args.nbtrialsused * trial_len


method = args.method
SHUFFLE = args.shuffle
PARATRANS = args.paratrans
nb_calibsessions = 1
UPDATE_STEP = args.updatestep
nb_disconnected = args.nbdisconnected
latent_dimensions = 5
update_len = sub_trial_length * UPDATE_STEP
nb_update_trials = data_len_persubj // update_len
data_folder = f'../../Experiments/Data/{args.dataset}/'
files = [f for f in os.listdir(data_folder) if f.endswith('.mat')]


for SEED in args.seeds:
    subjects = [int(''.join(filter(str.isdigit, f))) for f in files]
    print(f"Running with seed: {SEED}")
    rng = np.random.default_rng(SEED)
    rng.shuffle(subjects)
    eeg_dict = {}
    feats_dict = {}
    labels_dict = {}
    for subject in subjects:
        eeg_trials, feats_trials, label_trials = data_per_subj(subject, data_folder)
        eeg_dict[subject] = eeg_trials
        feats_dict[subject] = feats_trials
        labels_dict[subject] = label_trials

    est_subs = subjects[:args.nbestsubj]
    est_corr_att_sum, est_corr_unatt_sum = utils_prob.estimate_distribution_corr(eeg_dict, feats_dict, labels_dict, est_subs, fs, hparadata, hparafeats, leave_out_persubj=leave_out_persubj, trial_len=update_len, range_into_account=evalpara[0], nb_comp_into_account=evalpara[1])
    gmm_0, gmm_1 = utils_prob.fit_gmm(est_corr_att_sum, est_corr_unatt_sum, n_components_per_class=1)
    subjects = subjects[args.nbestsubj:]

    data_subjects_dict = {}
    feats_subjects_dict = {}
    labels_subjects_dict = {}
    labels_noisy_dict = {}
    for subj in subjects:
        data_trials = eeg_dict[subj]
        if nb_disconnected > 0:
            disconnected_channels = rng.choice(data_trials[0].shape[1], nb_disconnected, replace=False)
            for i in range(len(data_trials)):
                data_trials[i][:, disconnected_channels] = 0
        data_trials = [utils_stream.process_data_per_view(d, hparadata[0], hparadata[1], NORMALIZE=True) for d in data_trials]
        feats_trials = feats_dict[subj]
        feats_trials = [utils_stream.process_data_per_view(f, hparafeats[0], hparafeats[1], NORMALIZE=True) for f in feats_trials]
        labels_trials = labels_dict[subj]
        if sub_trial_length is not None:
            data_trials, feats_trials, labels_trials = utils_stream.further_split_and_shuffle(data_trials, feats_trials, labels_trials, sub_trial_length, fs, SHUFFLE=SHUFFLE, SEED=SEED)
        data_subjects_dict[subj] = data_trials
        feats_subjects_dict[subj] = feats_trials
        labels_subjects_dict[subj] = labels_trials
        labels_noisy_dict[subj] = utils_stream.add_label_noise(labels_trials, args.labelnoise, rng)

    stream = utils_stream.STREAM(data_subjects_dict, feats_subjects_dict, hparadata[0], hparafeats[0], latent_dimensions, SEED, evalpara, nb_update_trials, UPDATE_STEP)
    true_labels = np.stack([v[:nb_update_trials*UPDATE_STEP] for v in labels_subjects_dict.values()], axis=0)
    folder_path = f'tables/{args.dataset}/stream/'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    file_name = f"{folder_path}{'nbtrialsused'+str(args.nbtrialsused)+'_' if args.nbtrialsused is not None else ''}nbdisconnected{nb_disconnected}_predtriallen{sub_trial_length}_updatestep{UPDATE_STEP}_nbestsubj{args.nbestsubj}_hparadata{hparadata[0]}_{hparadata[1]}_hparafeats{hparafeats[0]}_{hparafeats[1]}_evalpara{evalpara[0]}_{evalpara[1]}_weightpara{weightpara[0]}_{weightpara[1]}_PARATRANS{PARATRANS}_SHUFFLE{SHUFFLE}{'_labelnoise'+str(args.labelnoise) if args.labelnoise is not None else ''}_SEED{SEED}.pkl"
    if os.path.exists(file_name):
        # read the existing file
        with open(file_name, 'rb') as f:
            res_dict = pickle.load(f)
            true_labels_saved = res_dict['true']
            assert np.array_equal(true_labels, true_labels_saved), "True labels do not match with the saved file."
    else:
        res_dict = {'true': true_labels}

    if method == 'fixsup' or 'all':
        print("#####Fixed Supervised#####")
        pred_labels_dict = stream.fixed_supervised(labels_noisy_dict, subjects[:nb_calibsessions], subjects[nb_calibsessions:])
        pred_labels_fixed = np.stack([v for v in pred_labels_dict.values()], axis=0)
        res_dict['fixed'] = pred_labels_fixed
    if method == 'adapsupsingle' or 'all':
        print("#####Adaptive Supervised (Single-Enc)#####")
        pred_labels_dict = stream.adaptive_supervised(labels_noisy_dict, weightpara, PARATRANS=PARATRANS, SINGLEENC=True)
        pred_labels_sup_single = np.stack([v for v in pred_labels_dict.values()], axis=0)
        res_dict['adapsup_single'] = pred_labels_sup_single
    if method == 'adapsuptwo' or 'all':
        print("#####Adaptive Supervised (Two-Enc)#####")
        pred_labels_dict = stream.adaptive_supervised(labels_noisy_dict, weightpara, PARATRANS=PARATRANS, SINGLEENC=False)
        pred_labels_sup_two = np.stack([v for v in pred_labels_dict.values()], axis=0)
        res_dict['adapsup_two'] = pred_labels_sup_two
    if method == 'adapsupsoft' or 'all':
        print("#####Adaptive Supervised (Soft)#####")
        pred_labels_dict = stream.recursive_soft(weightpara, gmm_0, gmm_1, PARATRANS=PARATRANS, labels_conditions_dict=labels_noisy_dict, confi=0.85)
        pred_labels_sup_soft = np.stack([v for v in pred_labels_dict.values()], axis=0)
        res_dict['adapsup_soft'] = pred_labels_sup_soft
    if method == 'single' or 'all':
        print("#####Single-Enc#####")
        pred_labels_dict = stream.recursive(weightpara, PARATRANS=PARATRANS, SINGLEENC=True)
        pred_labels_single = np.stack([v for v in pred_labels_dict.values()], axis=0)
        res_dict['single'] = pred_labels_single
    if method == 'two' or 'all':
        print("#####Two-Enc#####")
        pred_labels_dict = stream.recursive(weightpara, PARATRANS=PARATRANS, SINGLEENC=False)
        pred_labels_two = np.stack([v for v in pred_labels_dict.values()], axis=0)
        res_dict['two'] = pred_labels_two
    if method == 'soft' or 'all':
        print("#####Soft Single-Enc#####")
        pred_labels_dict = stream.recursive_soft(weightpara, gmm_0, gmm_1, PARATRANS=PARATRANS)
        pred_labels_soft = np.stack([v for v in pred_labels_dict.values()], axis=0)
        res_dict['soft'] = pred_labels_soft

    with open(file_name, 'wb') as f:
        pickle.dump(res_dict, f)
