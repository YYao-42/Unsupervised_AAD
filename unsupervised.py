import utils
import utils_unsup
import numpy as np
import argparse
import pickle

argparser = argparse.ArgumentParser()
argparser.add_argument('--mod', type=str, default='EEG-EOG', help='The modality to use')
argparser.add_argument('--truelabelpct', type=float, default=0.5, help='The percentage of true labels in the training set')
argparser.add_argument('--keeptrainpct', type=float, help='The percentage of training set to keep')
argparser.add_argument('--sameweight', action='store_true', default=False, help='Whether to use same weights for attended and unattended features')
argparser.add_argument('--label_resolu', type=int, default=60, help='Resolution of the label (in the training set)')
argparser.add_argument('--track_resolu', type=int, default=60, help='Resolution of the tracking influence')
argparser.add_argument('--compete_resolu', type=int, default=60, help='Resolution of the compete task')
argparser.add_argument('--maxiter', type=int, default=6, help='Maximum number of iterations')
argparser.add_argument('--flippct', type=float, help='The percentage of flipped labels. If None, the percentage changes with the iteration.')
argparser.add_argument('--seeds', type=list, default=[2, 4, 8, 16, 32], help='Random seeds')
argparser.add_argument('--svad', action='store_true', default=False, help='Do svad task instead of mm task if True')
argparser.add_argument('--lwcov', action='store_true', default=False, help='Whether to use ledoit-wolf covariance estimator')
argparser.add_argument('--bootstrap', action='store_true', default=False, help='Whether to use bootstrap for mm task')
argparser.add_argument('--mixpair', action='store_true', default=False, help='Whether to mix the pairs in the test set for the svad task')
args = argparser.parse_args()

MOD = args.mod
TRUELABEL_PERCENT = args.truelabelpct
KEEP_TRAIN_PERCENT = args.keeptrainpct
SAMEWEIGHT = args.sameweight
label_resolu = args.label_resolu
track_resolu = args.track_resolu
compete_resolu = args.compete_resolu
MAX_ITER = args.maxiter
SVAD = args.svad
LWCOV = args.lwcov
BOOTSTRAP = args.bootstrap
MIXPAIR = args.mixpair
coe = args.flippct

fs = 30
CROSSVIEW = True
L_data = 3 
L_feats = int(fs/2) 
offset_data = 1
offset_feats = 0 
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
feat_att_unatt_list =[np.stack([att[:,8], unatt[:,8]], axis=1) for att, unatt in zip(feat_all_att_list, feat_all_unatt_list)]
feat_unatt_att_list = [np.stack([unatt[:,8], att[:,8]], axis=1) for att, unatt in zip(feat_all_att_list, feat_all_unatt_list)]
feat_SO_list = [feats[:,8] for feats in feat_all_SO_list]

for SEED in args.seeds:
    print(f'#########Seed: {SEED}#########')
    for Subj_ID in range(nb_subj):
        file_name = f'{table_path}{Subj_ID}_{'crossview' if CROSSVIEW else 'featview'}_trackresolu{track_resolu}_truelabelpct_{TRUELABEL_PERCENT}_labelresolu_{label_resolu}_{'svadresolu' if SVAD else 'mmresolu'}_{compete_resolu}coe{coe}_nbiter{MAX_ITER}_seed{SEED}{'_lwcov' if LWCOV else ''}{'_samew' if SAMEWEIGHT else ''}{KEEP_TRAIN_PERCENT if KEEP_TRAIN_PERCENT is not None else ''}{'_bootstrap' if BOOTSTRAP else ''}{'_mixpair' if MIXPAIR else ''}.pkl'
        print(f'#########Subject: {Subj_ID}#########')
        if not SVAD:
            views_train, views_val, views_test = utils_unsup.prepare_train_val_test_data(Subj_ID, MOD, modal_dict, modal_dict_SO, feat_att_unatt_list, feat_SO_list, L_data, offset_data, L_feats, offset_feats, fs, TRUELABEL_PERCENT, label_resolu, RANDSEED=SEED, KEEP_TRAIN_PERCENT=KEEP_TRAIN_PERCENT)
            model_list, influence_list, mask_list, rt_list, corr_sum_list, _, nb_correct_list, nb_trials_list = utils_unsup.iterate(views_train, views_val, views_test, fs, track_resolu, compete_resolu, L_data, L_feats, SVAD=SVAD, MAX_ITER=MAX_ITER, LWCOV=LWCOV, CROSSVIEW=CROSSVIEW, coe=coe, SAMEWEIGHT=SAMEWEIGHT, latent_dimensions=latent_dimensions, BOOTSTRAP=BOOTSTRAP, MIXPAIR=MIXPAIR)
            acc_list = [nb_correct/nb_trials for nb_correct, nb_trials in zip(nb_correct_list, nb_trials_list)]
            with open(file_name, 'wb') as f:
                # create a dictionary to save the data
                res = {'model': model_list, 'influence_list': influence_list, 'mask_list':mask_list, 'rt_list': rt_list, 'corr_sum_list': corr_sum_list, 'acc_list': acc_list}
                pickle.dump(res, f)
        else:
            views_train_folds, views_val_folds, views_test_folds = utils_unsup.prepare_train_val_test_data_svad(Subj_ID, MOD, modal_dict, feat_att_unatt_list, L_data, offset_data, L_feats, offset_feats, fs, TRUELABEL_PERCENT, label_resolu, RANDSEED=SEED, leave_out=2, KEEP_TRAIN_PERCENT=KEEP_TRAIN_PERCENT, LWCOV=LWCOV)
            corr_sum_att_folds = []
            corr_sum_unatt_folds = []
            nb_correct_folds = []
            nb_trials_folds = []
            for views_train, views_val, views_test in zip(views_train_folds, views_val_folds, views_test_folds):
                _, _, _, _, corr_sum_att_list, corr_sum_unatt_list, nb_correct_list, nb_trials_list = utils_unsup.iterate(views_train, views_val, views_test, fs, track_resolu, compete_resolu, L_data, L_feats, SVAD=SVAD, MAX_ITER=MAX_ITER, LWCOV=LWCOV, CROSSVIEW=CROSSVIEW, coe=coe, SAMEWEIGHT=SAMEWEIGHT, latent_dimensions=latent_dimensions, BOOTSTRAP=BOOTSTRAP)
                corr_sum_att_folds.append(np.array(corr_sum_att_list))
                corr_sum_unatt_folds.append(np.array(corr_sum_unatt_list))
                nb_correct_folds.append(np.array(nb_correct_list))
                nb_trials_folds.append(np.array(nb_trials_list))
            corr_sum_att = np.stack(corr_sum_att_folds, axis=0)
            corr_sum_unatt = np.stack(corr_sum_unatt_folds, axis=0)
            nb_correct = np.stack(nb_correct_folds, axis=0)
            nb_trials = np.stack(nb_trials_folds, axis=0)
            acc = nb_correct/nb_trials
            print(f'################Correlation (Att): {np.mean(corr_sum_att, axis=0)}################')
            print(f'################Correlation (Unatt): {np.mean(corr_sum_unatt, axis=0)}################')
            print(f'################Accuracy: {np.mean(acc, axis=0)}################')
            with open(file_name, 'wb') as f:
                res = {'corr_sum_att': corr_sum_att, 'corr_sum_unatt': corr_sum_unatt, 'acc': acc}
                pickle.dump(res, f)

    if TRUELABEL_PERCENT==0.0 or TRUELABEL_PERCENT==1.0:
        break