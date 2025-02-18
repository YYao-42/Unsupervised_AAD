import utils
import utils_unsup
import numpy as np
import argparse
import pickle
import os
from sklearn.model_selection import KFold


def prepare_folds_per_view(trials, n_folds, nbtraintrials, SEED):
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    train_folds = []
    test_folds = []
    for train_index, test_index in kf.split(trials):
        if nbtraintrials is not None:
            train_index = train_index[:nbtraintrials]
        train_folds.append([trials[i] for i in train_index])
        test_folds.append([trials[i] for i in test_index])
    train_folds = [np.concatenate(fold, axis=0) for fold in train_folds]
    test_folds = [np.concatenate(fold, axis=0) for fold in test_folds]
    return train_folds, test_folds

def prepare_folds_all_views(views, hparas, n_folds, nbtraintrials, SEED):
    train_folds_views = []
    test_folds_views = []
    for view, hpara in zip(views, hparas):
        train_folds, test_folds = prepare_folds_per_view(view, n_folds, nbtraintrials, SEED)
        train_folds_views.append([utils_unsup.process_data_per_view(fold, hpara[0], hpara[1], NORMALIZE=True) for fold in train_folds])
        test_folds_views.append([utils_unsup.process_data_per_view(fold, hpara[0], hpara[1], NORMALIZE=True) for fold in test_folds])
    views_train_folds = [folds for folds in zip(*train_folds_views)]
    views_test_folds = [folds for folds in zip(*test_folds_views)]
    return views_train_folds, views_test_folds


argparser = argparse.ArgumentParser()
argparser.add_argument('--dataset', type=str, help='The name of the dataset')
argparser.add_argument('--folds', type=int, help='The number of folds')
argparser.add_argument('--nbtraintrials', type=int, help='Restrict the number of training trials')
argparser.add_argument('--track_resolu', type=int, help='Tracking resolution in the training set')
argparser.add_argument('--compete_resolu', type=int, help='Resolution of the compete task')
argparser.add_argument('--maxiter', type=int, default=8, help='Maximum number of iterations')
argparser.add_argument('--flippct', type=float, help='The percentage of flipped labels. If None, the percentage changes with the iteration.')
argparser.add_argument('--hparadata', type=int, nargs='+', help='Parameters of the hankel matrix for the data')
argparser.add_argument('--hparafeats', type=int, nargs='+', help='Parameters of the hankel matrix for the features')
argparser.add_argument('--evalpara', type=int, nargs='+', default=[3, 2], help='Parameters (range_into_account, nb_comp_into_account) for the evaluation')
argparser.add_argument('--seeds', type=int, nargs='+', default=[2, 4, 8], help='Random seeds')
argparser.add_argument('--lwcov', action='store_true', default=False, help='Whether to use ledoit-wolf covariance estimator')
argparser.add_argument('--bootstrap', action='store_true', default=False, help='Whether to use bootstrap for mm task')
argparser.add_argument('--randinit', action='store_true', default=False, help='Start with random initialization')
argparser.add_argument('--twoenc', action='store_true', default=False, help='Whether to use two encoders for the svad task')
argparser.add_argument('--unbiased', action='store_true', default=False, help='Use the unbiased version')
args = argparser.parse_args()

dataset = args.dataset
nbtraintrials = args.nbtraintrials
if dataset == 'Neetha':
    folds = 4 if args.folds is None else args.folds
    trainmin = nbtraintrials
    fs = 20
    track_resolu = 60 if args.track_resolu is None else args.track_resolu
    compete_resolu = 20 if args.compete_resolu is None else args.compete_resolu
    hparadata = [4, 3] if args.hparadata is None else args.hparadata
    hparafeats = [6, 0] if args.hparafeats is None else args.hparafeats
if dataset == 'earEEG':
    folds = 3 if args.folds is None else args.folds
    trainmin = int(nbtraintrials*10)
    fs = 20
    track_resolu = 60 if args.track_resolu is None else args.track_resolu
    compete_resolu = 20 if args.compete_resolu is None else args.compete_resolu
    hparadata = [4, 3] if args.hparadata is None else args.hparadata
    hparafeats = [6, 0] if args.hparafeats is None else args.hparafeats
if dataset == 'fuglsang2018':
    folds = 4 if args.folds is None else args.folds
    trainmin = round(nbtraintrials*5/6)
    fs = 32
    track_resolu = 50 if args.track_resolu is None else args.track_resolu
    compete_resolu = 25 if args.compete_resolu is None else args.compete_resolu
    hparadata = [4, 3] if args.hparadata is None else args.hparadata
    hparafeats = [9, 0] if args.hparafeats is None else args.hparafeats

MAX_ITER = args.maxiter
coe = args.flippct
LWCOV = args.lwcov
BOOTSTRAP = args.bootstrap
RANDINIT = args.randinit
TWOENC = args.twoenc
UNBIASED = args.unbiased
L_data, offset_data = hparadata
L_feats, offset_feats = hparafeats
evalpara = args.evalpara
params_hankel = [(L_data, offset_data), (L_feats, offset_feats)]
latent_dimensions = 5
table_path = f'tables/{dataset}/'
utils.create_dir(table_path, CLEAR=False)

# Load the data
data_folder = f'../../Experiments/Data/{dataset}/'
files = [f for f in os.listdir(data_folder) if f.endswith('.mat')]
Subj_IDs = [int(''.join(filter(str.isdigit, f))) for f in files]
Subj_IDs.sort()

for SEED in args.seeds:
    print(f'#########Seed: {SEED}#########')
    for Subj_ID in Subj_IDs:
        eeg_trials, att_trials, unatt_trials = utils.prepare_speech_data(Subj_ID, data_folder)
        att_unatt_trials = [np.stack([att, unatt], axis=1) for att, unatt in zip(att_trials, unatt_trials)]
        file_name = f'{table_path}{Subj_ID}_twoenc_folds{folds}{'_trainmin'+str(trainmin) if trainmin is not None else ''}_hankel{str(params_hankel)}_eval{str(evalpara)}{'_track_resolu'+str(track_resolu) if args.track_resolu is not None else ''}_compete_resolu{compete_resolu}_coe{coe}_nbiter{MAX_ITER}_seed{SEED}{'_lwcov' if LWCOV else ''}{'_bootstrap' if BOOTSTRAP else ''}{'_randinit' if RANDINIT else ''}{'_twoenc' if TWOENC else ''}{'_unbiased' if UNBIASED else ''}.pkl'
        print(f'#########Subject: {Subj_ID}#########')
        corr_sum_att_folds = []
        corr_sum_unatt_folds = []
        nb_correct_train_folds = []
        nb_trials_train_folds = []
        nb_correct_folds = []
        nb_trials_folds = []
        views_train_folds, views_test_folds = prepare_folds_all_views([eeg_trials, att_unatt_trials], [(L_data, offset_data), (L_feats, offset_feats)], folds, nb_trials, SEED)
        for i, (views_train, views_test) in enumerate(zip(views_train_folds, views_test_folds)):
            print(f'############Fold: {i}############')
            views_val = None
            _, _, _, _, _, corr_sum_att_list, corr_sum_unatt_list, nb_correct_train_list, nb_trials_train_list, nb_correct_list, nb_trials_list  = utils_unsup.iterate(views_train, views_val, views_test, fs, track_resolu, compete_resolu, L_data, L_feats, SEED, SVAD=True, MAX_ITER=MAX_ITER, LWCOV=LWCOV, coe=coe, latent_dimensions=latent_dimensions, evalpara=evalpara, BOOTSTRAP=BOOTSTRAP, MIXPAIR=False, TWOENC=TWOENC, RANDINIT=RANDINIT, UNBIASED=UNBIASED)
            corr_sum_att_folds.append(np.array(corr_sum_att_list))
            corr_sum_unatt_folds.append(np.array(corr_sum_unatt_list))
            nb_correct_train_folds.append(np.array(nb_correct_train_list))
            nb_trials_train_folds.append(np.array(nb_trials_train_list))
            nb_correct_folds.append(np.array(nb_correct_list))
            nb_trials_folds.append(np.array(nb_trials_list))
        corr_sum_att = np.stack(corr_sum_att_folds, axis=0)
        corr_sum_unatt = np.stack(corr_sum_unatt_folds, axis=0)
        nb_correct_train = np.stack(nb_correct_train_folds, axis=0)
        nb_trials_train = np.stack(nb_trials_train_folds, axis=0)
        nb_correct = np.stack(nb_correct_folds, axis=0)
        nb_trials = np.stack(nb_trials_folds, axis=0)
        acc_train = nb_correct_train/nb_trials_train
        acc = nb_correct/nb_trials
        print(f'################Correlation (Att): {np.mean(corr_sum_att, axis=0)}################')
        print(f'################Correlation (Unatt): {np.mean(corr_sum_unatt, axis=0)}################')
        print(f'################Accuracy (Train): {np.mean(acc_train, axis=0)}################)')
        print(f'################Accuracy (Test): {np.mean(acc, axis=0)}################')
        with open(file_name, 'wb') as f:
            res = {'corr_sum_att': corr_sum_att, 'corr_sum_unatt': corr_sum_unatt, 'acc_train': acc_train, 'acc': acc}
            pickle.dump(res, f)
    
    if not RANDINIT:
        break