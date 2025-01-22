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


def prepare_train_val_test_data_svad(Subj_ID, MOD, modal_dict, feat_att_list, feat_unatt_list, L_data, offset_data, L_feats, offset_feats, fs, TRUELABEL_PERCENT, resolu, RANDSEED, leave_out=2, KEEP_TRAIN_PERCENT=None, LWCOV=False):
    # Load data for the specific subject
    data_onesubj_list = [data[:,:,Subj_ID] for data in modal_dict[MOD]]
    # Split the SI data into training and validation sets
    train_list_folds, test_list_folds, val_list_folds = utils.split_multi_mod_withval_LVO([data_onesubj_list, feat_att_list, feat_unatt_list], leave_out=leave_out, VAL=not LWCOV)
    nb_videos = len(feat_att_list)
    views_train_folds = []
    views_val_folds = []
    views_test_folds = []
    for fold in range(nb_videos//leave_out):
        data_train, feats_att_train, feats_unatt_train = change_feats_label(train_list_folds[fold], fs, TRUELABEL_PERCENT, resolu, RANDSEED)
        if KEEP_TRAIN_PERCENT is not None:
            T_train = data_train.shape[0]
            data_train = data_train[:int(T_train*KEEP_TRAIN_PERCENT),:]
            feats_att_train = feats_att_train[:int(T_train*KEEP_TRAIN_PERCENT),:]
            feats_unatt_train = feats_unatt_train[:int(T_train*KEEP_TRAIN_PERCENT),:]
        data_test, feats_att_test, feats_unatt_test = test_list_folds[fold]
        views_train = [process_data_per_view(data_train, L_data, offset_data, NORMALIZE=True), process_data_per_view(feats_att_train, L_feats, offset_feats, NORMALIZE=True), process_data_per_view(feats_unatt_train, L_feats, offset_feats, NORMALIZE=True)]
        views_test = [process_data_per_view(data_test, L_data, offset_data, NORMALIZE=True), process_data_per_view(feats_att_test, L_feats, offset_feats, NORMALIZE=True), process_data_per_view(feats_unatt_test, L_feats, offset_feats, NORMALIZE=True)]
        views_train_folds.append(views_train)
        views_test_folds.append(views_test)
        if not LWCOV:
            data_val, feats_att_val, feats_unatt_val = val_list_folds[fold] 
            views_val = [process_data_per_view(data_val, L_data, offset_data, NORMALIZE=True), process_data_per_view(feats_att_val, L_feats, offset_feats, NORMALIZE=True), process_data_per_view(feats_unatt_val, L_feats, offset_feats, NORMALIZE=True)]
            views_val_folds.append(views_val)
        else:
            views_val_folds.append(None)
    return views_train_folds, views_val_folds, views_test_folds


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
            corr = model.average_pairwise_correlations(views_val[:2])
            corr_sum = cal_corr_sum(corr)
            print(f'c: {c}, corr_val: {corr}')
            if corr_sum > best_corr_sum:
                best_corr_sum = corr_sum
                best_c = c
                best_model = model
                best_corr_val = corr
        print(f'Best c: {best_c}')
    best_corr_train = best_model.average_pairwise_correlations([data_train, feats_att_train])
    print(f'Corr_train: {best_corr_train}')
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
    coe = 0.5**(ITER) if coe is None else coe
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
        nb_trials = int(T/fs*3)
        start_points = np.random.randint(0, T-trial_len*fs, size=nb_trials)
        data_seg = utils.into_trials(data, fs, trial_len, start_points=start_points)
        feats_seg = utils.into_trials(feats, fs, trial_len, start_points=start_points)
        mismatch_seg = [utils.select_distractors(feats, fs, trial_len, start_point) for start_point in start_points]
    nb_trials = len(data_seg)
    corr_match = [model.average_pairwise_correlations([d, f]) for d, f in zip(data_seg, feats_seg)]
    corr_mismatch = [model.average_pairwise_correlations([d, f]) for d, f in zip(data_seg, mismatch_seg)]
    acc = utils.eval_compete(np.stack(corr_match, axis=0), np.stack(corr_mismatch, axis=0), TRAIN_WITH_ATT=True)
    nb_correct = round(acc * nb_trials)
    return nb_correct, nb_trials


def svad(views, model, fs, trial_len, overlap=0.9, BOOTSTRAP=False):
    data, att, unatt = views
    T = data.shape[0]
    if not BOOTSTRAP:
        data_seg = utils.into_trials_with_overlap(data, fs, trial_len, overlap=overlap)
        att_seg = utils.into_trials_with_overlap(att, fs, trial_len, overlap=overlap)
        unatt_seg = utils.into_trials_with_overlap(unatt, fs, trial_len, overlap=overlap)
    else:
        nb_trials = int(T/fs*3)
        start_points = np.random.randint(0, T-trial_len*fs, size=nb_trials)
        data_seg = utils.into_trials(data, fs, trial_len, start_points=start_points)
        att_seg  = utils.into_trials(att, fs, trial_len, start_points=start_points)
        unatt_seg = utils.into_trials(unatt, fs, trial_len, start_points=start_points)
    nb_trials = len(data_seg)
    corr_a = [model.average_pairwise_correlations([d, f]) for d, f in zip(data_seg, att_seg)]
    corr_u = [model.average_pairwise_correlations([d, f]) for d, f in zip(data_seg, unatt_seg)]
    acc = utils.eval_compete(np.stack(corr_a, axis=0), np.stack(corr_u, axis=0), TRAIN_WITH_ATT=True)
    nb_correct = round(acc * nb_trials)
    return nb_correct, nb_trials


def iterate(views_train_ori, views_val, views_test, fs, track_resolu, compete_resolu, SVAD=False, MAX_ITER=10, LWCOV=True, coe=1, latent_dimensions=5, BOOTSTRAP=False):
    views_train = copy.deepcopy(views_train_ori)
    model_list = []
    corr_pair_list = []
    mask_list = []
    rt_list = []
    corr_sum_att_list = []
    corr_sum_unatt_list = []
    nb_correct_list = []
    nb_trials_list = []
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
        if SVAD:
            corr_att_test = model.average_pairwise_correlations(views_test[:2])
            corr_sum_att_list.append(cal_corr_sum(corr_att_test))
            print(f'Corr_sum_att_test: {cal_corr_sum(corr_att_test)}')
            corr_unatt_test = model.average_pairwise_correlations([views_test[0], views_test[2]])
            corr_sum_unatt_list.append(cal_corr_sum(corr_unatt_test))
            print(f'Corr_sum_unatt_test: {cal_corr_sum(corr_unatt_test)}')
            nb_correct, nb_trials = svad(views_test, model, fs, compete_resolu, BOOTSTRAP=BOOTSTRAP)
        else:
            corr_att_test = model.average_pairwise_correlations(views_test)
            corr_sum_att_list.append(cal_corr_sum(corr_att_test))
            print(f'Corr_sum_att_test: {cal_corr_sum(corr_att_test)}')
            corr_sum_unatt_list.append(None)
            nb_correct, nb_trials = match_mismatch(views_test, model, fs, compete_resolu, BOOTSTRAP=BOOTSTRAP)
        nb_correct_list.append(nb_correct)
        nb_trials_list.append(nb_trials)
        views_train = update_training_views(mask, views_in_segs)
    return model_list, corr_pair_list, mask_list, rt_list, corr_sum_att_list, corr_sum_unatt_list, nb_correct_list, nb_trials_list
