import utils
import utils_unsup
import numpy as np
import argparse
import pickle
import os
from sklearn.model_selection import KFold

def trial_further_split(trials, trial_len, fs):
    new_trials = []
    for trial in trials:
        new_trials.extend([trial[i:i+fs*trial_len] for i in range(0, len(trial), fs*trial_len)])
    return new_trials

def prepare_folds_per_view(trials, n_folds, nbtraintrials, SEED):
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    train_folds = []
    test_folds = []
    for train_index, test_index in kf.split(trials):
        if nbtraintrials is not None:
            rng = np.random.RandomState(SEED)
            train_index = rng.choice(train_index, nbtraintrials, replace=False)
        train_folds.append([trials[i] for i in train_index])
        test_folds.append([trials[i] for i in test_index])
    train_folds = [np.concatenate(fold, axis=0) for fold in train_folds]
    test_folds = [np.concatenate(fold, axis=0) for fold in test_folds]
    return train_folds, test_folds

def prepare_folds_per_view_Neetha_adaptive(trials, n_folds, SEED):
    # take out the last 24 min to be the test set
    test_trials = trials[-24:]
    test_set = np.concatenate(test_trials, axis=0)
    train_trials = trials[:-24]
    # divide train set into 6-min trials
    train_longer_trials = [np.concatenate(train_trials[i:i+6]) for i in range(0, len(train_trials), 6)]
    rng = np.random.RandomState(SEED)
    train_folds = []
    test_folds = []
    for i in range(n_folds):
        # randomly permute the 6-min trials
        train_longer_trials_perm = rng.permutation(train_longer_trials)
        train_folds.append(np.concatenate(train_longer_trials_perm))
        test_folds.append(test_set)
    return train_folds, test_folds

def prepare_folds_all_views(views, hparas, n_folds, nbtraintrials, SEED, Neetha=False):
    train_folds_views = []
    test_folds_views = []
    for view, hpara in zip(views, hparas):
        train_folds, test_folds = prepare_folds_per_view(view, n_folds, nbtraintrials, SEED) if not Neetha else prepare_folds_per_view_Neetha_adaptive(view, n_folds, SEED)
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
argparser.add_argument('--switch', action='store_true', default=False, help='Switch to single-enc mode after convergence')
argparser.add_argument('--recursive', action='store_true', default=False, help='Use time-adaptive version (recursive implementation)')
argparser.add_argument('--weightpara', type=float, nargs='+', help='alpha and beta for the time-adaptive version')
argparser.add_argument('--slidingwin', action='store_true', default=False, help='Use time-adaptive version (sliding window implementation)')
argparser.add_argument('--poolsize', type=int, help='Pool size for the sliding window implementation')
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
    trainmin = int(nbtraintrials*10) if nbtraintrials is not None else None
    fs = 20
    track_resolu = 60 if args.track_resolu is None else args.track_resolu
    compete_resolu = 20 if args.compete_resolu is None else args.compete_resolu
    hparadata = [4, 3] if args.hparadata is None else args.hparadata
    hparafeats = [6, 0] if args.hparafeats is None else args.hparafeats
if dataset == 'fuglsang2018':
    folds = 4 if args.folds is None else args.folds
    trainmin = round(nbtraintrials*5/6) if nbtraintrials is not None else None
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
SWITCH = args.switch
TIMEADAPTIVE = args.recursive or args.slidingwin
L_data, offset_data = hparadata
L_feats, offset_feats = hparafeats
evalpara = args.evalpara
weightpara = args.weightpara
pool_size = args.poolsize
params_hankel = [(L_data, offset_data), (L_feats, offset_feats)]
latent_dimensions = 5
table_path = f'tables/{dataset}/recursive/' if TIMEADAPTIVE else f'tables/{dataset}/'
utils.create_dir(table_path, CLEAR=False)

# Load the data
data_folder = f'../../Experiments/Data/{dataset}/'
files = [f for f in os.listdir(data_folder) if f.endswith('.mat')]
Subj_IDs = [int(''.join(filter(str.isdigit, f))) for f in files]
Subj_IDs.sort()

if not TIMEADAPTIVE:
    for SEED in args.seeds:
        print(f'#########Seed: {SEED}#########')
        for Subj_ID in Subj_IDs:
            eeg_trials, att_trials, unatt_trials = utils.prepare_speech_data(Subj_ID, data_folder)
            att_unatt_trials = [np.stack([att, unatt], axis=1) for att, unatt in zip(att_trials, unatt_trials)]
            file_name = f'{table_path}{Subj_ID}_twoenc_folds{folds}{'_trainmin'+str(trainmin) if trainmin is not None else ''}_hankel{str(params_hankel)}_eval{str(evalpara)}{'_track_resolu'+str(track_resolu) if args.track_resolu is not None else ''}_compete_resolu{compete_resolu}_coe{coe}_nbiter{MAX_ITER}_seed{SEED}{'_lwcov' if LWCOV else ''}{'_bootstrap' if BOOTSTRAP else ''}{'_randinit' if RANDINIT else ''}{'_twoenc' if TWOENC else ''}{'_unbiased' if UNBIASED else ''}{'_switch' if SWITCH else ''}.pkl'
            print(f'#########Subject: {Subj_ID}#########')
            corr_sum_att_folds = []
            corr_sum_unatt_folds = []
            nb_correct_train_folds = []
            nb_trials_train_folds = []
            nb_correct_folds = []
            nb_trials_folds = []
            views_train_folds, views_test_folds = prepare_folds_all_views([eeg_trials, att_unatt_trials], [(L_data, offset_data), (L_feats, offset_feats)], folds, nbtraintrials, SEED)
            for i, (views_train, views_test) in enumerate(zip(views_train_folds, views_test_folds)):
                print(f'############Fold: {i}############')
                views_val = None
                if SWITCH:
                    corr_sum_att_list, corr_sum_unatt_list, nb_correct_train_list, nb_trials_train_list, nb_correct_list, nb_trials_list = utils_unsup.iterate_switch(views_train, views_val, views_test, fs, track_resolu, compete_resolu, L_data, L_feats, SEED, SVAD=True, MAX_ITER=MAX_ITER, LWCOV=LWCOV, coe=coe, latent_dimensions=latent_dimensions, evalpara=evalpara, BOOTSTRAP=BOOTSTRAP, MIXPAIR=False, TWOENC=TWOENC, RANDINIT=RANDINIT, UNBIASED_SE=False)
                else:
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
else:
    Neetha = False   # dataset == 'Neetha'
    for SEED in args.seeds:
        print(f'#########Seed: {SEED}#########')
        for Subj_ID in Subj_IDs:
            eeg_trials, att_trials, unatt_trials = utils.prepare_speech_data(Subj_ID, data_folder)
            # if dataset == 'earEEG':
            #     eeg_trials = trial_further_split(eeg_trials, 60, fs)
            #     att_trials = trial_further_split(att_trials, 60, fs)
            #     unatt_trials = trial_further_split(unatt_trials, 60, fs)
            att_unatt_trials = [np.stack([att, unatt], axis=1) for att, unatt in zip(att_trials, unatt_trials)]
            file_name = f'{table_path}{Subj_ID}_{'adap' if args.recursive else 'slidingwin'}_twoenc_folds{folds}_hankel{str(params_hankel)}_eval{str(evalpara)}{('_weightpara'+str(weightpara)) if weightpara is not None else ''}{('_poolsize'+str(pool_size)) if pool_size is not None else ''}{'_track_resolu'+str(track_resolu) if args.track_resolu is not None else ''}_compete_resolu{compete_resolu}_seed{SEED}{'_bootstrap' if BOOTSTRAP else ''}{'_twoenc' if TWOENC else ''}{'_newsplit' if Neetha else ''}.pkl'
            print(f'#########Subject: {Subj_ID}#########')
            nb_correct_folds = []
            nb_trials_folds = []
            views_train_folds, views_test_folds = prepare_folds_all_views([eeg_trials, att_unatt_trials], [(L_data, offset_data), (L_feats, offset_feats)], folds, None, SEED, Neetha=Neetha)
            for i, (views_train, views_test) in enumerate(zip(views_train_folds, views_test_folds)):
                print(f'############Fold: {i}############')
                if args.recursive:
                    _, nb_correct_list, nb_trials_list = utils_unsup.recursive(views_train, views_test, fs, track_resolu, compete_resolu, L_data, L_feats, SEED, latent_dimensions=latent_dimensions, weightpara=weightpara, evalpara=evalpara, BOOTSTRAP=BOOTSTRAP)
                else:
                    _, nb_correct_list, nb_trials_list = utils_unsup.sliding_window(views_train, views_test, fs, pool_size, track_resolu, compete_resolu, L_feats, SEED, latent_dimensions=latent_dimensions, evalpara=evalpara, BOOTSTRAP=BOOTSTRAP)
                nb_correct_folds.append(np.array(nb_correct_list))
                nb_trials_folds.append(np.array(nb_trials_list))
            nb_correct = np.stack(nb_correct_folds, axis=0)
            nb_trials = np.stack(nb_trials_folds, axis=0)
            acc = nb_correct/nb_trials
            print(f'################Accuracy (Test): {np.mean(acc, axis=0)}################')
            with open(file_name, 'wb') as f:
                res = {'acc': acc}
                pickle.dump(res, f)