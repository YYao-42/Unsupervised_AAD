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
    # Use SI data for training
    data_SI = np.concatenate(tuple(data_onesubj_list), axis=0)
    feats_SI = np.concatenate(tuple(feat_att_unatt_list), axis=0)
    data_train, feats_train = change_feats_label([data_SI, feats_SI], fs, TRUELABEL_PERCENT, resolu, RANDSEED)
    if KEEP_TRAIN_PERCENT is not None:
        T_train = data_train.shape[0]
        data_train = data_train[:int(T_train*KEEP_TRAIN_PERCENT),:]
        feats_train = feats_train[:int(T_train*KEEP_TRAIN_PERCENT),:]
    # Use the SO data for validation and testing
    test_list_folds, val_list_folds = utils.split_multi_mod_LVO([data_onesubj_SO_list, feat_SO_list], leave_out=2)
    data_test, feats_test = test_list_folds[0]
    data_val, feats_val = val_list_folds[0]
    views_train = [process_data_per_view(data_train, L_data, offset_data, NORMALIZE=True), process_data_per_view(feats_train, L_feats, offset_feats, NORMALIZE=True)]
    views_val = [process_data_per_view(data_val, L_data, offset_data, NORMALIZE=True), process_data_per_view(feats_val, L_feats, offset_feats, NORMALIZE=True)]
    views_test = [process_data_per_view(data_test, L_data, offset_data, NORMALIZE=True), process_data_per_view(feats_test, L_feats, offset_feats, NORMALIZE=True)]
    return views_train, views_val, views_test


def prepare_train_val_test_data_svad(Subj_ID, MOD, modal_dict, feat_att_unatt_list, L_data, offset_data, L_feats, offset_feats, fs, TRUELABEL_PERCENT, resolu, RANDSEED, leave_out=2, KEEP_TRAIN_PERCENT=None, LWCOV=False):
    # Load data for the specific subject
    data_onesubj_list = [data[:,:,Subj_ID] for data in modal_dict[MOD]]
    # Split the SI data into training and validation sets
    train_list_folds, test_list_folds, val_list_folds = utils.split_multi_mod_withval_LVO([data_onesubj_list, feat_att_unatt_list], leave_out=leave_out, VAL=not LWCOV)
    nb_videos = len(feat_att_unatt_list)
    views_train_folds = []
    views_val_folds = []
    views_test_folds = []
    for fold in range(nb_videos//leave_out):
        data_train, feats_train = change_feats_label(train_list_folds[fold], fs, TRUELABEL_PERCENT, resolu, RANDSEED)
        if KEEP_TRAIN_PERCENT is not None:
            T_train = data_train.shape[0]
            data_train = data_train[:int(T_train*KEEP_TRAIN_PERCENT),:]
            feats_train = feats_train[:int(T_train*KEEP_TRAIN_PERCENT),:]
        data_test, feats_test = test_list_folds[fold]
        views_train = [process_data_per_view(data_train, L_data, offset_data, NORMALIZE=True), process_data_per_view(feats_train, L_feats, offset_feats, NORMALIZE=True)]
        views_test = [process_data_per_view(data_test, L_data, offset_data, NORMALIZE=True), process_data_per_view(feats_test, L_feats, offset_feats, NORMALIZE=True)]
        views_train_folds.append(views_train)
        views_test_folds.append(views_test)
        if not LWCOV:
            data_val, feats_val = val_list_folds[fold] 
            views_val = [process_data_per_view(data_val, L_data, offset_data, NORMALIZE=True), process_data_per_view(feats_val, L_feats, offset_feats, NORMALIZE=True)]
            views_val_folds.append(views_val)
        else:
            views_val_folds.append(None)
    return views_train_folds, views_val_folds, views_test_folds


def cal_corr_sum(corr, range_into_account=5, nb_comp_into_account=2):
    corr_ranked = np.sort(corr[:range_into_account])[::-1]
    corr_sum = np.sum(corr_ranked[:nb_comp_into_account])
    return corr_sum


def train_cca_model(views_train, views_val, L_feats, LWCOV=False, latent_dimensions=5):
    if LWCOV:
        best_model = MCCA_LW(latent_dimensions=latent_dimensions)
        best_model.fit(views_train)
        best_corr_val = None
    else:
        d_feats_val = views_val[1].shape[1]
        param_grid = {'c': [(a, b) for a, b in product([1e-8, 1e-7, 1e-6, 1e-5], repeat=2) if a > b]}
        best_corr_sum = -np.inf
        for c in param_grid['c']:
            model = MCCA(latent_dimensions=latent_dimensions, pca=False, eps=0, c=c)  
            model.fit(views_train)
            if d_feats_val == L_feats:
                model_single = copy.deepcopy(model)
                model_single.weights_[1] = model.weights_[1][:L_feats, :]
                corr = model_single.average_pairwise_correlations(views_val)
            else:
                corr = model.average_pairwise_correlations(views_val)
            corr_sum = cal_corr_sum(corr)
            print(f'c: {c}, corr_val: {corr}')
            if corr_sum > best_corr_sum:
                best_corr_sum = corr_sum
                best_c = c
                best_model = model
                best_corr_val = corr
        print(f'Best c: {best_c}')
    best_corr_train = best_model.average_pairwise_correlations(views_train)
    print(f'Corr_train: {best_corr_train}')
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
    coe = 0.5**(ITER) if coe is None else coe
    idx_keep = idx_sort[int(coe*nb_detected_seg):]
    mask[idx_keep] = True
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


def svad(views, model, fs, trial_len, overlap=0.9, BOOTSTRAP=False, MIXPAIR=False):
    data, att_unatt = views
    T = data.shape[0]
    L_feats = att_unatt.shape[1]//2
    att = att_unatt[:, :L_feats]
    unatt = att_unatt[:, L_feats:]
    if not BOOTSTRAP:
        data_seg = utils.into_trials_with_overlap(data, fs, trial_len, overlap=overlap)
        att_seg = utils.into_trials_with_overlap(att, fs, trial_len, overlap=overlap)
        unatt_seg = utils.into_trials_with_overlap(unatt, fs, trial_len, overlap=overlap)
    else:
        nb_trials = int(T/fs*3)
        if not MIXPAIR:    
            start_points = np.random.randint(0, T-trial_len*fs, size=nb_trials)
            data_seg = utils.into_trials(data, fs, trial_len, start_points=start_points)
            att_seg  = utils.into_trials(att, fs, trial_len, start_points=start_points)
            unatt_seg = utils.into_trials(unatt, fs, trial_len, start_points=start_points)
        else:
            # Note that this part is only written for the case when leave_out=2
            start_points_v1 = np.random.randint(0, T//2-trial_len//2*fs, size=nb_trials)
            start_points_v2 = start_points_v1 + T//2
            data_seg_v1 = utils.into_trials(data, fs, trial_len//2, start_points=start_points_v1)
            data_seg_v2 = utils.into_trials(data, fs, trial_len//2, start_points=start_points_v2)
            data_seg = [np.concatenate([d1, d2], axis=0) for d1, d2 in zip(data_seg_v1, data_seg_v2)]
            att_seg_v1 = utils.into_trials(att, fs, trial_len//2, start_points=start_points_v1)
            att_seg_v2 = utils.into_trials(att, fs, trial_len//2, start_points=start_points_v2)
            att_seg = [np.concatenate([a1, a2], axis=0) for a1, a2 in zip(att_seg_v1, att_seg_v2)]
            unatt_seg_v1 = utils.into_trials(unatt, fs, trial_len//2, start_points=start_points_v1)
            unatt_seg_v2 = utils.into_trials(unatt, fs, trial_len//2, start_points=start_points_v2)
            unatt_seg = [np.concatenate([u1, u2], axis=0) for u1, u2 in zip(unatt_seg_v1, unatt_seg_v2)]
    nb_trials = len(data_seg)
    corr_a = [model.average_pairwise_correlations([d, f]) for d, f in zip(data_seg, att_seg)]
    corr_u = [model.average_pairwise_correlations([d, f]) for d, f in zip(data_seg, unatt_seg)]
    acc = utils.eval_compete(np.stack(corr_a, axis=0), np.stack(corr_u, axis=0), TRAIN_WITH_ATT=True, nb_comp_into_account=2)
    nb_correct = round(acc * nb_trials)
    return nb_correct, nb_trials


def iterate(views_train_ori, views_val, views_test, fs, track_resolu, compete_resolu, L_data, L_feats, SVAD=False, MAX_ITER=10, LWCOV=True, CROSSVIEW=True, coe=1, SAMEWEIGHT=False, latent_dimensions=5, BOOTSTRAP=False, MIXPAIR=False):
    views_train = copy.deepcopy(views_train_ori)
    model_list = []
    influence_list = []
    mask_list = []
    rt_list = []
    corr_sum_att_list = []
    corr_sum_unatt_list = []
    nb_correct_list = []
    nb_trials_list = []
    for i in range(MAX_ITER):
        model, corr_val, corr_train = train_cca_model(views_train, views_val, L_feats, LWCOV, latent_dimensions=latent_dimensions)
        model_list.append(model)
        print(f'Corr_sum_train: {cal_corr_sum(corr_train)}')
        if corr_val is not None:
            print(f'Corr_sum_val: {cal_corr_sum(corr_val)}')
        idx = np.argmax(corr_val)
        influence_views, mask, rt, views_in_segs = get_mask_from_influence(views_train, model, fs, track_resolu, L_data, L_feats, idx, ITER=i, CROSSVIEW=CROSSVIEW, coe=coe, SAMEWEIGHT=SAMEWEIGHT)
        influence_list.append(influence_views[1][:,idx,:])
        mask_list.append(mask)
        rt_list.append(rt)
        model.weights_[1] = model.weights_[1][:L_feats, :]
        if SVAD:
            data_test, att_unatt_test = views_test
            att_test = att_unatt_test[:, :L_feats]
            unatt_test = att_unatt_test[:, L_feats:]
            corr_att_test = model.average_pairwise_correlations([data_test, att_test])
            corr_sum_att_list.append(cal_corr_sum(corr_att_test))
            print(f'Corr_sum_att_test: {cal_corr_sum(corr_att_test)}')
            corr_unatt_test = model.average_pairwise_correlations([data_test, unatt_test])
            corr_sum_unatt_list.append(cal_corr_sum(corr_unatt_test))
            print(f'Corr_sum_unatt_test: {cal_corr_sum(corr_unatt_test)}')
            nb_correct, nb_trials = svad(views_test, model, fs, compete_resolu, BOOTSTRAP=BOOTSTRAP, MIXPAIR=MIXPAIR)
        else:
            corr_att_test = model.average_pairwise_correlations(views_test)
            corr_sum_att_list.append(cal_corr_sum(corr_att_test))
            print(f'Corr_sum_att_test: {cal_corr_sum(corr_att_test)}')
            corr_sum_unatt_list.append(None)
            nb_correct, nb_trials = match_mismatch(views_test, model, fs, compete_resolu, BOOTSTRAP=BOOTSTRAP)
        nb_correct_list.append(nb_correct)
        nb_trials_list.append(nb_trials)
        views_train = update_training_views(mask, views_in_segs, L_feats)
    return model_list, influence_list, mask_list, rt_list, corr_sum_att_list, corr_sum_unatt_list, nb_correct_list, nb_trials_list
