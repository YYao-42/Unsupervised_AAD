import utils
import numpy as np
import copy
from cca_zoo.linear import MCCA, rCCA
from algo_suppl import MCCA_LW
from itertools import product


def process_data_per_view(view, L, offset, NORMALIZE=True):
    view_hankelized = utils.block_Hankel(view, L, offset)
    if NORMALIZE:
        view_hankelized = utils.normalize_per_view(view_hankelized)
    return view_hankelized


def change_feats_label(views, fs, TRUELABEL_PERCENT, resolu, RANDSEED):
    np.random.seed(RANDSEED)
    data, feats_att_unatt = views
    feats_segs = utils.into_trials_with_overlap(feats_att_unatt, fs, resolu, overlap=0)
    data_segs = utils.into_trials_with_overlap(data, fs, resolu, overlap=0)
    nb_trials = len(feats_segs)
    nb_false_label = nb_trials - int(nb_trials * TRUELABEL_PERCENT)
    # ceate random index for false label
    idx_false_label = np.random.choice(range(nb_trials), nb_false_label, replace=False)
    # change the label
    for idx in idx_false_label:
        temp =  feats_segs[idx][:, 0].copy()
        feats_segs[idx][:, 0] = feats_segs[idx][:, 1]
        feats_segs[idx][:, 1] = temp
    feats_att_unatt = np.concatenate(tuple(feats_segs), axis=0)
    data = np.concatenate(tuple(data_segs), axis=0)
    return [data, feats_att_unatt]


def prepare_train_val_test_data(Subj_ID, MOD, modal_dict, modal_dict_SO, feat_att_unatt_list, feat_SO_list, L_data, offset_data, L_feats, offset_feats, fs, TRUELABEL_PERCENT, resolu, RANDSEED, KEEP_TRAIN_PERCENT=None):
    # Load data for the specific subject
    data_onesubj_list = [data[:,:,Subj_ID] for data in modal_dict[MOD]]
    data_onesubj_SO_list = [data[:,:,Subj_ID] for data in modal_dict_SO[MOD]]
    # Split the SI data into training and validation sets
    train_list_folds, val_list_folds = utils.split_multi_mod_LVO([data_onesubj_list, feat_att_unatt_list], leave_out=2)
    data_train, feats_train = change_feats_label(train_list_folds[0], fs, TRUELABEL_PERCENT, resolu, RANDSEED)
    if KEEP_TRAIN_PERCENT is not None:
        T_train = data_train.shape[0]
        data_train = data_train[:int(T_train*KEEP_TRAIN_PERCENT),:]
        feats_train = feats_train[:int(T_train*KEEP_TRAIN_PERCENT),:]
    data_val, feats_val = val_list_folds[0]
    # Use the SO data for testing
    data_test = np.concatenate(tuple(data_onesubj_SO_list), axis=0)
    feats_test = np.concatenate(tuple(feat_SO_list), axis=0)
    views_train = [process_data_per_view(data_train, L_data, offset_data, NORMALIZE=True), process_data_per_view(feats_train, L_feats, offset_feats, NORMALIZE=True)]
    views_val = [process_data_per_view(data_val, L_data, offset_data, NORMALIZE=True), process_data_per_view(feats_val, L_feats, offset_feats, NORMALIZE=True)]
    views_test = [process_data_per_view(data_test, L_data, offset_data, NORMALIZE=True), process_data_per_view(feats_test, L_feats, offset_feats, NORMALIZE=True)]
    return views_train, views_val, views_test


def cal_corr_sum(corr, range_into_account=3, nb_comp_into_account=1):
    corr_ranked = np.sort(corr[:range_into_account])[::-1]
    corr_sum = np.sum(corr_ranked[:nb_comp_into_account])
    return corr_sum


def train_cca_model(views_train, views_val, LWCOV=False, MULTIVIEW=False, latent_dimensions=5):
    if LWCOV:
        best_model = MCCA_LW(latent_dimensions=latent_dimensions)
        best_model.fit(views_train)
        best_corr_val = best_model.average_pairwise_correlations(views_val)
        print(f'Corr: {best_corr_val}')
    else:
        param_grid = {'c': [1e-6]} if MULTIVIEW else {'c': [(a, b) for a, b in product([1e-8, 1e-7, 1e-6, 1e-5], repeat=2) if a > b]}
        best_corr_sum = -np.inf
        for c in param_grid['c']:
            model = MCCA(latent_dimensions=latent_dimensions, pca=False, eps=0, c=c) if MULTIVIEW else rCCA(latent_dimensions=latent_dimensions, pca=False, eps=0, c=c)
            model.fit(views_train)
            corr = model.average_pairwise_correlations(views_val)
            corr_sum = cal_corr_sum(corr)
            print(f'c: {c}, corr: {corr}')
            if corr_sum > best_corr_sum:
                best_corr_sum = corr_sum
                best_c = c
                best_model = model
                best_corr_val = corr
        print(f'Best c: {best_c}')
    best_corr_train = best_model.average_pairwise_correlations(views_train)
    return best_model, best_corr_val, best_corr_train


def get_mask_from_influence(views_train, model, fs, track_resolu, L_data, L_feats, idx, ITER, CROSSVIEW=True, coe=1, SAMEWEIGHT=False):
    # Convert views into trials with overlap
    views_in_segs = [utils.into_trials_with_overlap(view, fs, track_resolu, overlap=0) for view in views_train]
    nb_views = len(views_in_segs)
    nb_segs = len(views_in_segs[0])
    # Get views in each segment
    segs_views = [[views_in_segs[i][j] for i in range(nb_views)] for j in range(nb_segs)]
    # Get influence of views for each segment
    weights = copy.deepcopy(model.weights_)
    if SAMEWEIGHT:
        weights[1][L_feats:, :] = model.weights_[1][:L_feats, :]
    segs_influence_views = [utils.get_influence_all_views(views, weights, [L_data, L_feats], 'SDP', CROSSVIEW=CROSSVIEW, NORMALIZATION=False) for views in segs_views]
    # Stack the influence of views along axis 2; shape of elements in influence_views: (dim_view, nb_components, nb_segs)
    influence_views = [np.stack([segs_influence_views[j][i] for j in range(nb_segs)], axis=2) for i in range(nb_views)]
    influ_diff = influence_views[1][0, idx, :] - influence_views[1][1, idx, :]
    mask = influ_diff > 0
    nb_detected_seg = np.sum(influ_diff < 0)
    # sort the influence difference from the smallest to the largest
    idx_sort = np.argsort(influ_diff)
    coe = 0.5**(ITER+1) if coe is None else coe
    idx_keep = idx_sort[int(coe*nb_detected_seg):]
    mask[idx_keep] = True
    # Create a mask where the influence of the first component is greater than the second
    # mask = influence_views[1][0, idx, :] > influence_views[1][1, idx, :]
    rt = 1 - nb_detected_seg / len(influ_diff)
    print(f'Ratio of True: {rt}')
    return influence_views, mask, rt, views_in_segs


def update_training_views(mask, views_in_segs, L_feats):
    for i, indicator in enumerate(mask):
        if not indicator:
            feats_seg_i = views_in_segs[1][i]
            temp = np.zeros_like(feats_seg_i)
            temp[:, :L_feats] = feats_seg_i[:, L_feats:2 * L_feats]
            temp[:, L_feats:2 * L_feats] = feats_seg_i[:, :L_feats]
            views_in_segs[1][i] = temp
    data_h_train_updated = np.concatenate(tuple(views_in_segs[0]), axis=0)
    feats_h_train_updated = np.concatenate(tuple(views_in_segs[1]), axis=0)
    views_train_updated = [data_h_train_updated, feats_h_train_updated]
    return views_train_updated


def match_mismatch(views, model, fs, trial_len, overlap=0.8):
    data, feats = views
    data_seg = utils.into_trials_with_overlap(data, fs, trial_len, overlap=overlap)
    feats_seg = utils.into_trials_with_overlap(feats, fs, trial_len, overlap=overlap)
    mismatch_seg = utils.into_trials_with_overlap(feats, fs, trial_len, overlap=overlap, PERMUTE=True)
    corr_match = [model.average_pairwise_correlations([d, f]) for d, f in zip(data_seg, feats_seg)]
    corr_mismatch = [model.average_pairwise_correlations([d, f]) for d, f in zip(data_seg, mismatch_seg)]
    acc = utils.eval_compete(np.stack(corr_match, axis=0), np.stack(corr_mismatch, axis=0), TRAIN_WITH_ATT=True)
    return acc


def iterate(views_train_ori, views_val, views_test, fs, track_resolu, mm_resolu, L_data, L_feats, MAX_ITER=10, LWCOV=True, CROSSVIEW=True, coe=1, SAMEWEIGHT=False, latent_dimensions=5):
    views_train = copy.deepcopy(views_train_ori)
    model_list = []
    influence_list = []
    mask_list = []
    avg_mag_list = []
    rt_list = []
    corr_sum_list = []
    acc_list = []
    for i in range(MAX_ITER):
        model, corr_val, corr_train = train_cca_model(views_train, views_val, LWCOV, latent_dimensions=latent_dimensions)
        model_list.append(model)
        print(f'Corr_sum_train: {cal_corr_sum(corr_train)}')
        print(f'Corr_sum_val: {cal_corr_sum(corr_val)}')
        idx = np.argmax(corr_val)
        influence_views, mask, rt, views_in_segs = get_mask_from_influence(views_train, model, fs, track_resolu, L_data, L_feats, idx, ITER=i, CROSSVIEW=CROSSVIEW, coe=coe, SAMEWEIGHT=SAMEWEIGHT)
        influence_list.append(influence_views[1][:,idx,:])
        mask_list.append(mask)
        rt_list.append(rt)
        avg_mag = np.array([np.mean(feat[:,[0, L_feats]], axis=0) for feat in views_in_segs[1]])
        avg_mag_list.append(avg_mag.T)
        model.weights_[1] = model.weights_[1][:L_feats, :]
        corr_test = model.average_pairwise_correlations(views_test)
        corr_sum = cal_corr_sum(corr_test)
        corr_sum_list.append(corr_sum)
        print(f'Corr_sum_test: {corr_sum}')
        acc = match_mismatch(views_test, model, fs, mm_resolu)
        acc_list.append(acc)
        views_train = update_training_views(mask, views_in_segs, L_feats)
    return model_list, influence_list, mask_list, avg_mag_list, rt_list, corr_sum_list, acc_list
