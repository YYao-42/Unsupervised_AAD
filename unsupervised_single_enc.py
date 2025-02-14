import utils
import utils_unsup_single_enc
import numpy as np
import argparse
import pickle

argparser = argparse.ArgumentParser()
argparser.add_argument('--mod', type=str, default='EEG-EOG', help='The modality to use')
argparser.add_argument('--trainmin', type=int, help='Restrict the number of training samples')
argparser.add_argument('--track_resolu', type=int, default=60, help='Resolution of the tracking influence')
argparser.add_argument('--compete_resolu', type=int, default=60, help='Resolution of the compete task')
argparser.add_argument('--maxiter', type=int, default=8, help='Maximum number of iterations')
argparser.add_argument('--flippct', type=float, help='The percentage of flipped labels. If None, the percentage changes with the iteration.')
argparser.add_argument('--hparadata', type=int, nargs='+', default=[3, 1], help='Parameters of the hankel matrix for the data')
argparser.add_argument('--hparafeats', type=int, nargs='+', default=[15, 0], help='Parameters of the hankel matrix for the features')
argparser.add_argument('--evalpara', type=int, nargs='+', default=[2, 2], help='Parameters (range_into_account, nb_comp_into_account) for the evaluation')
argparser.add_argument('--seeds', type=int, nargs='+', default=[2, 4, 8], help='Random seeds')
argparser.add_argument('--svad', action='store_true', default=False, help='Do svad task instead of mm task if True')
argparser.add_argument('--lwcov', action='store_true', default=False, help='Whether to use ledoit-wolf covariance estimator')
argparser.add_argument('--bootstrap', action='store_true', default=False, help='Whether to use bootstrap for mm task')
argparser.add_argument('--randinit', action='store_true', default=False, help='Start with random initialization')
argparser.add_argument('--unbiased', action='store_true', default=False, help='Use the unbiased version')
argparser.add_argument('--mixpair', action='store_true', default=False, help='Whether to mix the pairs in the test set for the svad task')
args = argparser.parse_args()

MOD = args.mod
trainmin = args.trainmin
track_resolu = args.track_resolu if args.track_resolu is not None else 60
compete_resolu = args.compete_resolu
MAX_ITER = args.maxiter
SVAD = args.svad
LWCOV = args.lwcov
BOOTSTRAP = args.bootstrap
RANDINIT = args.randinit
UNBIASED = args.unbiased
MIXPAIR = args.mixpair
coe = args.flippct
L_data, offset_data = args.hparadata
L_feats, offset_feats = args.hparafeats
evalpara = args.evalpara

fs = 30
params_hankel = [(L_data, offset_data), (L_feats, offset_feats)]
latent_dimensions = 5 if MOD != 'GAZE_V' else 3
table_path = f'tables/{MOD}/'
utils.create_dir(table_path, CLEAR=False)

# Load the data
modal_dict, feat_all_att_list, feat_all_unatt_list, Subj_Set = utils.prepare_data_multimod()
nb_subj = len(Subj_Set)
modal_dict_SO, feat_all_SO_list, _, _ = utils.prepare_data_multimod(SINGLEOBJ=True)
for key in modal_dict:
    modal_dict[key] = [data[:,:,Subj_Set] for data in modal_dict[key]]
    modal_dict_SO[key] = [data[:,:,Subj_Set] for data in modal_dict_SO[key]]
modal_dict['EEG-EOG'] = utils.further_regress_out_list(modal_dict['EEG'], modal_dict['EOG'], 1, 1, 0, 0)
modal_dict_SO['EEG-EOG'] = utils.further_regress_out_list(modal_dict_SO['EEG'], modal_dict_SO['EOG'], 1, 1, 0, 0)
feat_att_list = [feats[:,8] for feats in feat_all_att_list]
feat_unatt_list = [feats[:,8] for feats in feat_all_unatt_list]
feat_SO_list = [feats[:,8] for feats in feat_all_SO_list]

for SEED in args.seeds:
    print(f'#########Seed: {SEED}#########')
    for Subj_ID in range(nb_subj):
        file_name = f'{table_path}{Subj_ID}_singleenc{'_trainmin'+str(trainmin) if trainmin is not None else ''}_hankel{str(params_hankel)}_eval{str(evalpara)}_trackresolu{track_resolu}_{'svadresolu' if SVAD else 'mmresolu'}_{compete_resolu}_coe{coe}_nbiter{MAX_ITER}_seed{SEED}{'_lwcov' if LWCOV else ''}{'_bootstrap' if BOOTSTRAP else ''}{'_randinit' if RANDINIT else ''}{'_unbiased' if UNBIASED else ''}{'_mixpair' if MIXPAIR else ''}.pkl'
        print(f'#########Subject: {Subj_ID}#########')
        if not SVAD:
            views_train, views_val, views_test = utils_unsup_single_enc.prepare_train_val_test_data(Subj_ID, MOD, modal_dict, modal_dict_SO, feat_att_list, feat_unatt_list, feat_SO_list, params_hankel, fs, trainmin=trainmin)
            model_list, corr_pair_list, mask_list, updated_label_list, rt_list, corr_sum_list, _, nb_correct_train_list, nb_trials_train_list, nb_correct_list, nb_trials_list = utils_unsup_single_enc.iterate(views_train, views_val, views_test, fs, track_resolu, compete_resolu, SEED, SVAD=SVAD, MAX_ITER=MAX_ITER, LWCOV=LWCOV, coe=coe, latent_dimensions=latent_dimensions, evalpara=evalpara, BOOTSTRAP=BOOTSTRAP, MIXPAIR=MIXPAIR, RANDINIT=RANDINIT, UNBIASED=UNBIASED)
            acc_train_list = [nb_correct/nb_trials for nb_correct, nb_trials in zip(nb_correct_train_list, nb_trials_train_list)]
            acc_list = [nb_correct/nb_trials for nb_correct, nb_trials in zip(nb_correct_list, nb_trials_list)]
            with open(file_name, 'wb') as f:
                # create a dictionary to save the data
                res = {'model': model_list, 'corr_pair_list': corr_pair_list, 'mask_list':mask_list, 'updated_label_list':updated_label_list, 'rt_list': rt_list, 'corr_sum_list': corr_sum_list, 'acc_train_list': acc_train_list, 'acc_list': acc_list}
                pickle.dump(res, f)
        else:
            corr_sum_att_folds = []
            corr_sum_unatt_folds = []
            nb_correct_train_folds = []
            nb_trials_train_folds = []
            nb_correct_folds = []
            nb_trials_folds = []
            views_train_folds, views_val_folds, views_test_folds = utils_unsup_single_enc.prepare_train_val_test_data_svad(Subj_ID, MOD, modal_dict, feat_att_list, feat_unatt_list, params_hankel, fs, leave_out=2, trainmin=trainmin, LWCOV=LWCOV)
            for views_train, views_val, views_test in zip(views_train_folds, views_val_folds, views_test_folds):
                _, _, _, _, _, corr_sum_att_list, corr_sum_unatt_list, nb_correct_train_list, nb_trials_train_list, nb_correct_list, nb_trials_list = utils_unsup_single_enc.iterate(views_train, views_val, views_test, fs, track_resolu, compete_resolu, SEED, SVAD=SVAD, MAX_ITER=MAX_ITER, LWCOV=LWCOV, coe=coe, latent_dimensions=latent_dimensions, evalpara=evalpara, BOOTSTRAP=BOOTSTRAP, MIXPAIR=MIXPAIR, RANDINIT=RANDINIT, UNBIASED=UNBIASED)
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