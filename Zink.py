import utils
import utils_unsup
import numpy as np
import argparse
import pickle
import scipy
import copy
import matplotlib.pyplot as plt

def select_att_unatt_feats(feats_trials, label_trials):
    nb_trials = len(feats_trials)
    att_trials = []
    unatt_trials = []
    for i in range(nb_trials):
        feats = feats_trials[i]
        att = feats[:, label_trials[i]-1]
        unatt = feats[:, 2-label_trials[i]]
        att_trials.append(np.expand_dims(att, axis=1))
        unatt_trials.append(np.expand_dims(unatt, axis=1))
    return att_trials, unatt_trials

def split_trial(views, nb_trials=2):
    T = views[0].shape[0]
    views_split = [[view[:T//nb_trials,...], view[T//nb_trials:,...]] for view in views]
    return views_split

def predict_labels(views, model, L_data, L_feats, evalpara, SPLIT=False):
    data, att_unatt = views
    if SPLIT:
        dim_feats = att_unatt.shape[1]//2
        att = att_unatt[:, :dim_feats]
        unatt = att_unatt[:, dim_feats:]
        model_single_enc = copy.deepcopy(model)
        model_single_enc.weights_[1] = model.weights_[1][:L_feats, :]
        data_pred_trials, att_pred_trials, unatt_pred_trials = split_trial([data, att, unatt])
        corr_a = [model_single_enc.average_pairwise_correlations([d, f]) for d, f in zip(data_pred_trials, att_pred_trials)]
        corr_sum_a = [utils_unsup.cal_corr_sum(corr, evalpara[0], evalpara[1]) for corr in corr_a]
        corr_u = [model_single_enc.average_pairwise_correlations([d, f]) for d, f in zip(data_pred_trials, unatt_pred_trials)]
        corr_sum_u = [utils_unsup.cal_corr_sum(corr, evalpara[0], evalpara[1]) for corr in corr_u]
        label = [corr_sum_a[i] > corr_sum_u[i] for i in range(len(corr_sum_a))]
        # data_pred_trials, att_unatt_trials = split_trial([data, att_unatt])
        # influ_trials = [utils.get_influence_all_views([d, f], model.weights_, [L_data, L_feats], 'SDP', CROSSVIEW=True, NORMALIZATION=False)[1] for d, f in zip(data_pred_trials, att_unatt_trials)]
        # label = [influ[0,0] > influ[1,0] for influ in influ_trials]
    else:
        # dim_feats = att_unatt.shape[1]//2
        # att = att_unatt[:, :dim_feats]
        # unatt = att_unatt[:, dim_feats:]
        # model_single_enc = copy.deepcopy(model)
        # model_single_enc.weights_[1] = model.weights_[1][:L_feats, :]
        # corr_a = model_single_enc.average_pairwise_correlations([data, att])
        # corr_sum_a = utils_unsup.cal_corr_sum(corr_a, evalpara[0], evalpara[1])
        # corr_u = model_single_enc.average_pairwise_correlations([data, unatt])
        # corr_sum_u = utils_unsup.cal_corr_sum(corr_u, evalpara[0], evalpara[1])
        # label = corr_sum_a > corr_sum_u
        # label = [corr_sum_a[i] > corr_sum_u[i] for i in range(len(corr_sum_a))]
        influ = utils.get_influence_all_views([data, att_unatt], model.weights_, [L_data, L_feats], 'SDP', CROSSVIEW=True, NORMALIZATION=False)[1]
        label = influ[0,0] > influ[1,0]
    return label

def recursive_multi_sessions(data_conditions_dict, att_conditions_dict, unatt_conditions_dict, L_data, L_feats, latent_dimensions, weightpara, SEED, evalpara, PARATRANS=False, nb_trials=None):
    model_init = None
    pred_labels = []
    for cond in ['CS-1', 'CS-2', 'TS-1', 'TS-2', 'TS-3', 'TS-4', 'FUS-1', 'FUS-2']:
        segs_views = [[data, np.concatenate((att, unatt), axis=1)] for data, att, unatt in zip(data_conditions_dict[cond], att_conditions_dict[cond], unatt_conditions_dict[cond])]
        if model_init is not None:
            model = model_init
            Rinit = Rinit
            Dinit = Dinit
        else:
            model = utils_unsup.train_cca_model_adaptive(segs_views[0], None, None, latent_dimensions=latent_dimensions, weightpara=weightpara, RANDMODEL=True, SEED=SEED)
            Rinit = None 
            Dinit = None
        pred_labels.append(predict_labels(segs_views[0], model, L_data, L_feats, evalpara, SPLIT=True))
        nb_trials = len(segs_views) if nb_trials is None else nb_trials
        for i in range(nb_trials):
            # update the model
            seg_to_pred = segs_views[i]
            label = predict_labels(seg_to_pred, model, L_data, L_feats, evalpara, SPLIT=False)
            if label:
                seg_predicted = seg_to_pred
            else:
                seg_predicted = [seg_to_pred[0], np.concatenate([seg_to_pred[1][:, L_feats:], seg_to_pred[1][:, :L_feats]], axis=1)]
            model = utils_unsup.train_cca_model_adaptive(seg_predicted, Rinit, Dinit, latent_dimensions=latent_dimensions, weightpara=weightpara, RANDMODEL=False, SEED=SEED)
            Rinit = model.Rxx 
            Dinit = model.Dxx
            if i < nb_trials - 1:
                # predict labels of the next trial
                pred_labels.append(predict_labels(segs_views[i+1], model, L_data, L_feats, evalpara, SPLIT=True))
        if PARATRANS:
            model_init = model
    return pred_labels

def sliding_multi_sessions(data_conditions_dict, att_conditions_dict, unatt_conditions_dict, L_data, L_feats, latent_dimensions, pool_size, SEED, evalpara, PARATRANS=False, nb_trials=None):
    model_init = None
    pred_labels = []
    for cond in ['CS-1', 'CS-2', 'TS-1', 'TS-2', 'TS-3', 'TS-4', 'FUS-1', 'FUS-2']:
        segs_views = [[data, np.concatenate((att, unatt), axis=1)] for data, att, unatt in zip(data_conditions_dict[cond], att_conditions_dict[cond], unatt_conditions_dict[cond])]
        dim_hankel = [view.shape[1] for view in segs_views[0]]
        cov_tensor_precomputed = utils_unsup.get_cov_tensor(segs_views, regularization='lwcov')
        if model_init is not None:
            model = model_init
            pool_init = pool_init
        else:
            pool_init = np.zeros((cov_tensor_precomputed.shape[0], cov_tensor_precomputed.shape[1], pool_size))
            pool_init[:,:,-1] = cov_tensor_precomputed[:,:,0]
            model = utils_unsup.train_cca_model_pool(pool_init, dim_hankel[0], latent_dimensions=latent_dimensions, RANDMODEL=True, SEED=SEED)
        pred_labels.append(predict_labels(segs_views[0], model, L_data, L_feats, evalpara, SPLIT=True))
        nb_trials = len(segs_views) if nb_trials is None else nb_trials
        for i in range(nb_trials):
            # Move to the next segment, predict the labels, and update the pool
            pool_tensor = np.zeros_like(pool_init)
            pool_tensor[:, :, 1:] = pool_init[:, :, :-1]
            pool_tensor[:, :, 0] = cov_tensor_precomputed[:, :, i]
            pool_init = pool_tensor
            pool_tensor, _ = utils_unsup.update_pool(pool_tensor, dim_hankel, model.weights_, evalpara)
            model = utils_unsup.train_cca_model_pool(pool_tensor, dim_hankel[0], latent_dimensions=latent_dimensions, RANDMODEL=False, SEED=SEED)
            if i < nb_trials - 1:
                # predict labels of the next trial
                pred_labels.append(predict_labels(segs_views[i+1], model, L_data, L_feats, evalpara, SPLIT=True))
        if PARATRANS:
            model_init = model
    return pred_labels

def calc_smooth_acc(pred_labels, nb_trials, nearby=14):
    labels_non_calib = pred_labels[2*nb_trials:]
    labels_non_calib = [item for sublist in labels_non_calib for item in sublist]
    acc_non_calib = np.sum(labels_non_calib)/len(labels_non_calib)
    print("Avg acc non-calib: ", acc_non_calib)
    labels_all = [item for sublist in pred_labels for item in sublist]
    acc = []
    for i in range(len(labels_all)):
        idx_range = (max(0, i-nearby), min(len(labels_all), i+nearby))
        acc.append(np.sum(labels_all[idx_range[0]:idx_range[1]])/len(labels_all[idx_range[0]:idx_range[1]]))
    return acc_non_calib, acc


argparser = argparse.ArgumentParser()
argparser.add_argument('--Subj_ID', type=int, help='Subject ID')
argparser.add_argument('--nbdisconnected', type=int, default=0, help='Number of disconnected channels')
argparser.add_argument('--nbtrials', type=int, default=24, help='Restrict the number of trials in each condition')
argparser.add_argument('--hparadata', type=int, nargs='+', default=[9, 8], help='Parameters of the hankel matrix for the data')
argparser.add_argument('--hparafeats', type=int, nargs='+', default=[1, 0], help='Parameters of the hankel matrix for the features')
argparser.add_argument('--evalpara', type=int, nargs='+', default=[1, 1], help='Parameters (range_into_account, nb_comp_into_account) for the evaluation')
argparser.add_argument('--weightpara', type=float, default=[0.9, 0.9], nargs='+', help='alpha and beta for the time-adaptive version')
argparser.add_argument('--poolsize', type=int, default=19, help='Pool size for the sliding window implementation')
argparser.add_argument('--paratrans', action='store_true', default=False, help='Whether to enable parameter transfer')
argparser.add_argument('--seeds', type=int, nargs='+', default=[1, 2, 4, 8, 16, 32, 64, 128, 256, 512], help='Random seeds')
args = argparser.parse_args()

latent_dimensions = 5
fs = 20
nb_disconnected = args.nbdisconnected
nb_trials = args.nbtrials
hparadata = args.hparadata
hparafeats = args.hparafeats
evalpara = args.evalpara
weightpara = args.weightpara
pool_size = args.poolsize
PARATRANS = args.paratrans
table_path = f'tables/Zink/Two-Enc/'
utils.create_dir(table_path, CLEAR=False)
fig_path = f'figures/Zink/Two-Enc/'
utils.create_dir(fig_path, CLEAR=False)

data_path = '../../Experiments/data/Zink/dataSubjectOfficial{}.mat'.format(args.Subj_ID)
for SEED in args.seeds:
    data = scipy.io.loadmat(data_path, squeeze_me=True)
    conditions = data['condition']
    unique_conditions = np.unique(conditions)
    data_conditions_dict = {}
    att_conditions_dict = {}
    unatt_conditions_dict = {}
    rng = np.random.RandomState(SEED)
    for cond in unique_conditions:
        data_trials = data['eegTrials'][conditions == cond]
        data_conditions_dict[cond] = [d for d in data_trials]
        if nb_disconnected > 0:
            disconnected_channels = rng.choice(data_conditions_dict[cond][0].shape[1], nb_disconnected, replace=False)
            for i in range(len(data_conditions_dict[cond])):
                data_conditions_dict[cond][i][:, disconnected_channels] = 0
        data_conditions_dict[cond] = [utils_unsup.process_data_per_view(d, hparadata[0], hparadata[1], NORMALIZE=True) for d in data_conditions_dict[cond]]
        att_conditions_dict[cond], unatt_conditions_dict[cond] = select_att_unatt_feats(data['audioTrials'][conditions == cond], data['attSpeaker'][conditions == cond])
        att_conditions_dict[cond] = [utils_unsup.process_data_per_view(d, hparafeats[0], hparafeats[1], NORMALIZE=True) for d in att_conditions_dict[cond]]
        unatt_conditions_dict[cond] = [utils_unsup.process_data_per_view(d, hparafeats[0], hparafeats[1], NORMALIZE=True) for d in unatt_conditions_dict[cond]]
    
    print(f"##################Recursive, SEED{SEED}##################")
    pred_labels_recur = recursive_multi_sessions(data_conditions_dict, att_conditions_dict, unatt_conditions_dict, hparadata[0], hparafeats[0], latent_dimensions, weightpara, SEED, evalpara, PARATRANS=PARATRANS, nb_trials=nb_trials)
    _, acc_recur = calc_smooth_acc(pred_labels_recur, nb_trials)

    print(f"##################Sliding, SEED{SEED}##################")
    pred_labels_slid = sliding_multi_sessions(data_conditions_dict, att_conditions_dict, unatt_conditions_dict, hparadata[0], hparafeats[0], latent_dimensions, pool_size, SEED, evalpara, PARATRANS=PARATRANS, nb_trials=nb_trials)
    _, acc_slid = calc_smooth_acc(pred_labels_slid, nb_trials)

    file_name = f"{table_path}{args.Subj_ID}_SEED{SEED}_nbdisconnected{nb_disconnected}_nbtrials{nb_trials}_hparadata{hparadata[0]}_{hparadata[1]}_hparafeats{hparafeats[0]}_{hparafeats[1]}_evalpara{evalpara[0]}_{evalpara[1]}_weightpara{weightpara[0]}_{weightpara[1]}_poolsize{pool_size}_PARATRANS{PARATRANS}.pkl"
    res_labels = {'recur': pred_labels_recur, 'slid': pred_labels_slid}
    with open(file_name, 'wb') as f:
        pickle.dump(res_labels, f)

    x_axis = np.arange(len(acc_recur))/2
    plt.plot(x_axis, acc_recur, label='Recursive', color='blue')
    plt.plot(x_axis, acc_slid, label='Sliding Window', color='orange')
    for i in range(1, 9):
        plt.axvline(x=i*nb_trials, color='grey', linestyle='--')
    plt.legend()
    plt.savefig(f"{fig_path}{args.Subj_ID}_SEED{SEED}_nbdisconnected{nb_disconnected}_nbtrials{nb_trials}_hparadata{hparadata[0]}_{hparadata[1]}_hparafeats{hparafeats[0]}_{hparafeats[1]}_evalpara{evalpara[0]}_{evalpara[1]}_weightpara{weightpara[0]}_{weightpara[1]}_poolsize{pool_size}_PARATRANS{PARATRANS}.png")
    plt.close()