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


def change_feats_label(views_train, fs, TRUELABEL_PERCENT, resolu, RANDSEED):
    np.random.seed(RANDSEED)
    data, feats_att, feats_unatt = views_train
    feats_att_segs = utils.into_trials_with_overlap(feats_att, fs, resolu, overlap=0)
    feats_unatt_segs = utils.into_trials_with_overlap(feats_unatt, fs, resolu, overlap=0)
    data_segs = utils.into_trials_with_overlap(data, fs, resolu, overlap=0)
    nb_trials = len(data_segs)
    nb_false_label = nb_trials - int(nb_trials * TRUELABEL_PERCENT)
    # ceate random index for false label
    idx_false_label = np.random.choice(range(nb_trials), nb_false_label, replace=False)
    # change the label
    for idx in idx_false_label:
        temp =  feats_att_segs[idx].copy()
        feats_att_segs[idx] = feats_unatt_segs[idx]
        feats_unatt_segs[idx] = temp
    data = np.concatenate(tuple(data_segs), axis=0)
    feats_att = np.concatenate(tuple(feats_att_segs), axis=0)
    feats_unatt = np.concatenate(tuple(feats_unatt_segs), axis=0)
    return [data, feats_att, feats_unatt]


def prepare_train_val_test_data(Subj_ID, MOD, modal_dict, modal_dict_SO, feat_att_list, feat_unatt_list, feat_SO_list, L_data, offset_data, L_feats, offset_feats, fs, TRUELABEL_PERCENT, resolu, RANDSEED, KEEP_TRAIN_PERCENT=None):
    # Load data for the specific subject
    data_onesubj_list = [data[:,:,Subj_ID] for data in modal_dict[MOD]]
    data_onesubj_SO_list = [data[:,:,Subj_ID] for data in modal_dict_SO[MOD]]
    # Use SI data for training
    data_SI = np.concatenate(tuple(data_onesubj_list), axis=0)
    feats_att = np.concatenate(tuple(feat_att_list), axis=0)
    feats_unatt = np.concatenate(tuple(feat_unatt_list), axis=0)
    data_train, feats_att_train, feats_unatt_train = change_feats_label([data_SI, feats_att, feats_unatt], fs, TRUELABEL_PERCENT, resolu, RANDSEED)
    if KEEP_TRAIN_PERCENT is not None:
        T_train = data_train.shape[0]
        data_train = data_train[:int(T_train*KEEP_TRAIN_PERCENT),:]
        feats_att_train = feats_att_train[:int(T_train*KEEP_TRAIN_PERCENT),:]
        feats_unatt_train = feats_unatt_train[:int(T_train*KEEP_TRAIN_PERCENT),:]
    # Use the SO data for validation and testing
    test_list_folds, val_list_folds = utils.split_multi_mod_LVO([data_onesubj_SO_list, feat_SO_list], leave_out=2)
    data_test, feats_test = test_list_folds[0]
    data_val, feats_val = val_list_folds[0]
    views_train = [process_data_per_view(data_train, L_data, offset_data, NORMALIZE=True), process_data_per_view(feats_att_train, L_feats, offset_feats, NORMALIZE=True), process_data_per_view(feats_unatt_train, L_feats, offset_feats, NORMALIZE=True)]
    views_val = [process_data_per_view(data_val, L_data, offset_data, NORMALIZE=True), process_data_per_view(feats_val, L_feats, offset_feats, NORMALIZE=True)]
    views_test = [process_data_per_view(data_test, L_data, offset_data, NORMALIZE=True), process_data_per_view(feats_test, L_feats, offset_feats, NORMALIZE=True)]
    return views_train, views_val, views_test


def cal_corr_sum(corr, range_into_account=5, nb_comp_into_account=2):
    corr_ranked = np.sort(corr[:range_into_account])[::-1]
    corr_sum = np.sum(corr_ranked[:nb_comp_into_account])
    return corr_sum


def train_cca_model(views_train, views_val, LWCOV=False, latent_dimensions=5):
    data_train, feats_att_train, _ = views_train
    if LWCOV:
        best_model = MCCA_LW(latent_dimensions=latent_dimensions)
        best_model.fit([data_train, feats_att_train])
        best_corr_val = None
    else:
        param_grid = {'c': [(a, b) for a, b in product([1e-8, 1e-7, 1e-6, 1e-5], repeat=2) if a > b]}
        best_corr_sum = -np.inf
        for c in param_grid['c']:
            model = MCCA(latent_dimensions=latent_dimensions, pca=False, eps=0, c=c) 
            model.fit([data_train, feats_att_train])
            corr = model.average_pairwise_correlations(views_val)
            corr_sum = cal_corr_sum(corr)
            print(f'c: {c}, corr: {corr}')
            if corr_sum > best_corr_sum:
                best_corr_sum = corr_sum
                best_c = c
                best_model = model
                best_corr_val = corr
        print(f'Best c: {best_c}')
    best_corr_train = best_model.average_pairwise_correlations([data_train, feats_att_train])
    return best_model, best_corr_val, best_corr_train


def get_corr_pair(seg_views, model):
    data, feats_att, feats_unatt = seg_views
    corr_att = model.average_pairwise_correlations([data, feats_att])
    corr_unatt = model.average_pairwise_correlations([data, feats_unatt])
    corr_sum_att = cal_corr_sum(corr_att)
    corr_sum_unatt = cal_corr_sum(corr_unatt)
    corr_sum_pair = np.stack([corr_sum_att, corr_sum_unatt], axis=0)
    return corr_sum_pair


def get_mask_from_corr(views_train, model, fs, track_resolu, ITER, coe=1):
    # Convert views into trials with overlap
    views_in_segs = [utils.into_trials_with_overlap(view, fs, track_resolu, overlap=0) for view in views_train]
    nb_views = len(views_in_segs)
    nb_segs = len(views_in_segs[0])
    # Get views in each segment
    segs_views = [[views_in_segs[i][j] for i in range(nb_views)] for j in range(nb_segs)]
    corr_sum_pairs = [get_corr_pair(seg_views, model) for seg_views in segs_views]
    corr_sum_pairs = np.stack(corr_sum_pairs, axis=0)
    corr_diff = corr_sum_pairs[:, 0] - corr_sum_pairs[:, 1]
    mask = corr_diff > 0
    nb_detected_seg = np.sum(corr_diff < 0)
    # sort the influence difference from the smallest to the largest
    idx_sort = np.argsort(corr_diff)
    coe = 0.5**(ITER+1) if coe is None else coe
    idx_keep = idx_sort[int(coe*nb_detected_seg):]
    mask[idx_keep] = True
    rt = 1 - nb_detected_seg / len(corr_diff)
    print(f'Ratio of True: {rt}')
    return corr_sum_pairs, mask, rt, views_in_segs


def update_training_views(mask, views_in_segs):
    for i, indicator in enumerate(mask):
        if not indicator:
            feats_att_seg_i = views_in_segs[1][i].copy()
            views_in_segs[1][i] = views_in_segs[2][i]
            views_in_segs[2][i] = feats_att_seg_i
    data_h_train_updated = np.concatenate(tuple(views_in_segs[0]), axis=0)
    feats_h_att_train_updated = np.concatenate(tuple(views_in_segs[1]), axis=0)
    feats_h_unatt_train_updated = np.concatenate(tuple(views_in_segs[2]), axis=0)
    views_train_updated = [data_h_train_updated, feats_h_att_train_updated, feats_h_unatt_train_updated]
    return views_train_updated


def match_mismatch(views, model, fs, trial_len, overlap=0.9, BOOTSTRAP=False):
    data, feats = views
    T = data.shape[0]
    if not BOOTSTRAP:
        data_seg = utils.into_trials_with_overlap(data, fs, trial_len, overlap=overlap)
        feats_seg = utils.into_trials_with_overlap(feats, fs, trial_len, overlap=overlap)
        mismatch_seg = utils.into_trials_with_overlap(feats, fs, trial_len, overlap=overlap, PERMUTE=True)
    else:
        nb_trials = int(T/fs/trial_len/(1-overlap))
        start_points = np.random.randint(0, T-trial_len*fs, size=nb_trials)
        data_seg = utils.into_trials(data, fs, trial_len, start_points=start_points)
        feats_seg = utils.into_trials(feats, fs, trial_len, start_points=start_points)
        mismatch_seg = [utils.select_distractors(feats, fs, trial_len, start_point) for start_point in start_points]
    corr_match = [model.average_pairwise_correlations([d, f]) for d, f in zip(data_seg, feats_seg)]
    corr_mismatch = [model.average_pairwise_correlations([d, f]) for d, f in zip(data_seg, mismatch_seg)]
    acc = utils.eval_compete(np.stack(corr_match, axis=0), np.stack(corr_mismatch, axis=0), TRAIN_WITH_ATT=True)
    return acc


def iterate(views_train_ori, views_val, views_test, fs, track_resolu, mm_resolu, L_data, L_feats, MAX_ITER=10, LWCOV=True, coe=1, latent_dimensions=5, BOOTSTRAP=False):
    views_train = copy.deepcopy(views_train_ori)
    model_list = []
    corr_pair_list = []
    mask_list = []
    rt_list = []
    corr_sum_list = []
    acc_list = []
    for i in range(MAX_ITER):
        model, corr_val, corr_train = train_cca_model(views_train, views_val, LWCOV, latent_dimensions=latent_dimensions)
        model_list.append(model)
        print(f'Corr_sum_train: {cal_corr_sum(corr_train)}')
        if corr_val is not None:
            print(f'Corr_sum_val: {cal_corr_sum(corr_val)}')
        corr_sum_pairs, mask, rt, views_in_segs = get_mask_from_corr(views_train, model, fs, track_resolu, ITER=i, coe=coe)
        corr_pair_list.append(corr_sum_pairs)
        mask_list.append(mask)
        rt_list.append(rt)
        corr_test = model.average_pairwise_correlations(views_test)
        corr_sum = cal_corr_sum(corr_test)
        corr_sum_list.append(corr_sum)
        print(f'Corr_sum_test: {corr_sum}')
        acc = match_mismatch(views_test, model, fs, mm_resolu, BOOTSTRAP=BOOTSTRAP)
        acc_list.append(acc)
        views_train = update_training_views(mask, views_in_segs)
    return model_list, corr_pair_list, mask_list, rt_list, corr_sum_list, acc_list
