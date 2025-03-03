import utils
import numpy as np
import copy
import utils_unsup_single_enc
from cca_zoo.linear import MCCA, rCCA
from algo_suppl import MCCA_LW
from itertools import product


def process_data_per_view(view, L, offset, NORMALIZE=True):
    view_hankelized = utils.block_Hankel(view, L, offset)
    if NORMALIZE:
        view_hankelized = utils.normalize_per_view(view_hankelized)
    return view_hankelized


def prepare_train_val_test_data(Subj_ID, MOD, modal_dict, modal_dict_SO, feat_att_unatt_list, feat_SO_list, hparas, fs, trainmin=None):
    # Load data for the specific subject
    data_onesubj_list = [data[:,:,Subj_ID] for data in modal_dict[MOD]]
    data_onesubj_SO_list = [data[:,:,Subj_ID] for data in modal_dict_SO[MOD]]
    # Use SI data for training
    data_train = np.concatenate(tuple(data_onesubj_list), axis=0)
    feats_train = np.concatenate(tuple(feat_att_unatt_list), axis=0)
    if trainmin is not None:
        nb_samples = int(fs*60*trainmin)
        data_train = data_train[:nb_samples,:]
        feats_train = feats_train[:nb_samples,:]
    # Use the SO data for validation and testing
    test_list_folds, val_list_folds = utils.split_multi_mod_LVO([data_onesubj_SO_list, feat_SO_list], leave_out=2)
    data_test, feats_test = test_list_folds[0]
    data_val, feats_val = val_list_folds[0]
    L_data, offset_data = hparas[0]
    L_feats, offset_feats = hparas[1]
    views_train = [process_data_per_view(data_train, L_data, offset_data, NORMALIZE=True), process_data_per_view(feats_train, L_feats, offset_feats, NORMALIZE=True)]
    views_val = [process_data_per_view(data_val, L_data, offset_data, NORMALIZE=True), process_data_per_view(feats_val, L_feats, offset_feats, NORMALIZE=True)]
    views_test = [process_data_per_view(data_test, L_data, offset_data, NORMALIZE=True), process_data_per_view(feats_test, L_feats, offset_feats, NORMALIZE=True)]
    return views_train, views_val, views_test


def prepare_train_val_test_data_svad(Subj_ID, MOD, modal_dict, feat_att_unatt_list, hparas, fs, leave_out=2, trainmin=None, LWCOV=False):
    # Load data for the specific subject
    data_onesubj_list = [data[:,:,Subj_ID] for data in modal_dict[MOD]]
    # Split the SI data into training and validation sets
    train_list_folds, test_list_folds, val_list_folds = utils.split_multi_mod_withval_LVO([data_onesubj_list, feat_att_unatt_list], leave_out=leave_out, VAL=not LWCOV)
    nb_videos = len(feat_att_unatt_list)
    L_data, offset_data = hparas[0]
    L_feats, offset_feats = hparas[1]
    views_train_folds = []
    views_val_folds = []
    views_test_folds = []
    for fold in range(nb_videos//leave_out):
        data_train, feats_train = train_list_folds[fold]
        if trainmin is not None:
            nb_samples = int(fs*60*trainmin)
            data_train = data_train[:nb_samples,:]
            feats_train = feats_train[:nb_samples,:]
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


def cal_corr_sum(corr, range_into_account=3, nb_comp_into_account=2):
    corr_ranked = np.sort(corr[:range_into_account])[::-1]
    corr_sum = np.sum(corr_ranked[:nb_comp_into_account])
    return corr_sum


def get_rand_model(model, SEED):
    rng = np.random.RandomState(SEED)
    rand_model = copy.deepcopy(model)
    rand_model.weights_ = [rng.randn(*W.shape) for W in model.weights_]
    return rand_model


def train_cca_model(views_train, views_val, L_feats, LWCOV=False, latent_dimensions=5, evalpara=[3, 2], RANDMODEL=False, SEED=None):
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
            corr_sum = cal_corr_sum(corr, range_into_account=evalpara[0], nb_comp_into_account=evalpara[1])
            print(f'c: {c}, corr_val: {corr}')
            if corr_sum > best_corr_sum:
                best_corr_sum = corr_sum
                best_c = c
                best_model = model
                best_corr_val = corr
        print(f'Best c: {best_c}')
    if RANDMODEL:
        assert SEED is not None, 'SEED must be provided for random initialization'
        best_model = get_rand_model(best_model, SEED)
        best_corr_val = None
    best_corr_train = best_model.average_pairwise_correlations(views_train)
    print(f'Corr_train: {best_corr_train}')
    return best_model, best_corr_val, best_corr_train


def train_cca_model_adaptive(views_train, Rinit, Dinit, latent_dimensions=5, weightpara=[0.0, 0.0], RANDMODEL=False, SEED=None):
    best_model = MCCA_LW(latent_dimensions=latent_dimensions, alpha=weightpara[0], beta=weightpara[1])
    best_model.fit(views_train, Rinit=Rinit, Dinit=Dinit)
    if RANDMODEL:
        assert SEED is not None, 'SEED must be provided for random initialization'
        best_model = get_rand_model(best_model, SEED)
    best_corr_train = best_model.average_pairwise_correlations(views_train)
    print(f'Corr_train: {best_corr_train}')
    return best_model


def get_segments(view, fs, resolu, overlap=0, MIXPAIR=False, start_points=None):
    '''
    if MIXPAIR:
    Assume that one subvideo in the experiment is the superimposed version of vidA and vidB. The participants watch the superimposed video twice, once with vidA attended and the other time with vidB attended.
    When dividing data into segments (either in the training or the test set), the first half of one segment is when vidA is attended and the second half is when vidB is attended.
    If there are multiple video pairs {(vidA1, vidB1), (vidA2, vidB2), ...}, the view is the concatenation of(data_vidA1_att, data_vidA2_att, ..., data_vidB1_att, data_vidB2_att, ...).
    '''
    T = view.shape[0]
    if not MIXPAIR:
        if start_points is None:
            segs = utils.into_trials_with_overlap(view, fs, resolu, overlap=overlap)
        else:
            segs = utils.into_trials(view, fs, resolu, start_points=start_points)
    else:
        if start_points is None:
            segs_part1 = utils.into_trials_with_overlap(view[:T//2], fs, resolu//2, overlap=overlap) 
            segs_part2 = utils.into_trials_with_overlap(view[T//2:], fs, resolu//2, overlap=overlap)
        else:
            start_points_v1 = start_points
            start_points_v2 = start_points_v1 + T//2
            segs_part1 = utils.into_trials(view, fs, resolu//2, start_points=start_points_v1)
            segs_part2 = utils.into_trials(view, fs, resolu//2, start_points=start_points_v2)
        segs = [np.concatenate([s1, s2], axis=0) for s1, s2 in zip(segs_part1, segs_part2)]
    return segs


def get_mask_from_influence(views_train, model, fs, track_resolu, L_data, L_feats, idx, ITER, CROSSVIEW=True, coe=1, SAMEWEIGHT=False, MIXPAIR=False):
    # Convert views into trials with overlap
    views_in_segs = [get_segments(view, fs, track_resolu, MIXPAIR=MIXPAIR) for view in views_train]
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
    print(f'Ratio of Predicted Att: {rt}')
    return influence_views, mask, rt, views_in_segs


def get_mask_unbiased(views_train, fs, track_resolu, L_data, L_feats, idx, ITER, CROSSVIEW=True, coe=1, evalpara=[3, 2], latent_dimensions=5, RANDINIT=False, SEED=None, MIXPAIR=False):
    # Convert views into trials with overlap
    views_in_segs = views_in_segs = [get_segments(view, fs, track_resolu, MIXPAIR=MIXPAIR) for view in views_train]
    nb_views = len(views_in_segs)
    nb_segs = len(views_in_segs[0])
    segs_influence_views = []
    for i in range(nb_segs):
        views_test = [views_in_segs[j][i] for j in range(nb_views)]
        views_train = [np.concatenate([views_in_segs[j][k] for k in range(nb_segs) if k != i], axis=0) for j in range(nb_views)]
        views_val = None
        LWCOV = True
        model, _, _ = train_cca_model(views_train, views_val, L_feats, LWCOV, latent_dimensions=latent_dimensions, evalpara=evalpara, RANDMODEL=RANDINIT, SEED=SEED)
        weights = copy.deepcopy(model.weights_)
        weights[1][L_feats:, :] = model.weights_[1][:L_feats, :]
        seg_influence_views = utils.get_influence_all_views(views_test, weights, [L_data, L_feats], 'SDP', CROSSVIEW=CROSSVIEW, NORMALIZATION=False)
        segs_influence_views.append(copy.deepcopy(seg_influence_views))
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
    print(f'Ratio of Predicted Att: {rt}')
    return influence_views, mask, rt, views_in_segs


def update_training_views(mask, views_in_segs, L_feats, true_label):
    updated_label = true_label==mask if true_label is not None else mask
    print('Acc (train): ', np.sum(updated_label)/len(updated_label))
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
    return updated_label, views_train_updated


def match_mismatch(views, model, fs, trial_len, overlap=0.9, BOOTSTRAP=False, evalpara=[3, 2]):
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
    acc = utils.eval_compete(np.stack(corr_match, axis=0), np.stack(corr_mismatch, axis=0), TRAIN_WITH_ATT=True, range_into_account=evalpara[0], nb_comp_into_account=evalpara[1])
    nb_correct = round(acc * nb_trials)
    return nb_correct, nb_trials


def svad(views, model, fs, trial_len, overlap=0, BOOTSTRAP=False, MIXPAIR=False, TWOENC=False, evalpara=[3, 2]):
    data, att_unatt = views
    T = data.shape[0]
    L_feats = att_unatt.shape[1]//2
    att = att_unatt[:, :L_feats]
    unatt = att_unatt[:, L_feats:]
    if BOOTSTRAP:
        nb_trials = int(T/fs*3)
        if MIXPAIR:
            start_points = np.random.randint(0, T//2-trial_len//2*fs, size=nb_trials)
        else:
            start_points = np.random.randint(0, T-trial_len*fs, size=nb_trials)
    else:
        start_points = None
    data_seg, att_seg, unatt_seg = [get_segments(view, fs, trial_len, overlap=overlap, MIXPAIR=MIXPAIR, start_points=start_points) for view in [data, att, unatt]]
    if TWOENC:
        assert model.weights_[1].shape[0] == att_unatt.shape[1], 'The model is not in two-encoder mode'
        att_unatt_seg = [np.concatenate([a, u], axis=1) for a, u in zip(att_seg, unatt_seg)]
        unatt_att_seg = [np.concatenate([u, a], axis=1) for a, u in zip(att_seg, unatt_seg)]
        corr_a = [model.average_pairwise_correlations([d, f]) for d, f in zip(data_seg, att_unatt_seg)]
        corr_u = [model.average_pairwise_correlations([d, f]) for d, f in zip(data_seg, unatt_att_seg)]
    else:
        corr_a = [model.average_pairwise_correlations([d, f]) for d, f in zip(data_seg, att_seg)]
        corr_u = [model.average_pairwise_correlations([d, f]) for d, f in zip(data_seg, unatt_seg)]
    acc = utils.eval_compete(np.stack(corr_a, axis=0), np.stack(corr_u, axis=0), TRAIN_WITH_ATT=True, range_into_account=evalpara[0], nb_comp_into_account=evalpara[1])
    nb_trials = len(data_seg)
    nb_correct = round(acc * nb_trials)
    return nb_correct, nb_trials


def iterate(views_train_ori, views_val_ori, views_test_ori, fs, track_resolu, compete_resolu, L_data, L_feats, SEED, SVAD=False, MAX_ITER=10, LWCOV=True, CROSSVIEW=True, coe=1, SAMEWEIGHT=False, latent_dimensions=5, evalpara=[3, 2], BOOTSTRAP=False, MIXPAIR=False, TWOENC=False, RANDINIT=False, UNBIASED=False, BREAK=False):
    views_train = copy.deepcopy(views_train_ori)
    views_val = copy.deepcopy(views_val_ori)
    views_test = copy.deepcopy(views_test_ori)
    model_list = []
    influence_list = []
    mask_list = []
    updated_label_list = []
    rt_list = []
    corr_sum_att_list = []
    corr_sum_unatt_list = []
    nb_correct_train_list = []
    nb_trials_train_list = []
    nb_correct_list = []
    nb_trials_list = []
    true_label = None
    for i in range(MAX_ITER):
        model, corr_val, corr_train = train_cca_model(views_train, views_val, L_feats, LWCOV, latent_dimensions=latent_dimensions, evalpara=evalpara, RANDMODEL=RANDINIT, SEED=SEED)
        model_two_enc = copy.deepcopy(model)
        model_list.append(model_two_enc)
        print(f'Corr_sum_train: {cal_corr_sum(corr_train, range_into_account=evalpara[0], nb_comp_into_account=evalpara[1])}')
        if corr_val is not None:
            print(f'Corr_sum_val: {cal_corr_sum(corr_val, range_into_account=evalpara[0], nb_comp_into_account=evalpara[1])}')
        idx = np.argmax(corr_val) # always 0 if corr_val is None
        flip_coe = coe if not RANDINIT else 1
        if UNBIASED:
            influence_views, mask, rt, views_in_segs = get_mask_unbiased(views_train, fs, track_resolu, L_data, L_feats, idx, ITER=i, CROSSVIEW=CROSSVIEW, coe=flip_coe, evalpara=evalpara, latent_dimensions=latent_dimensions, RANDINIT=RANDINIT, SEED=SEED, MIXPAIR=MIXPAIR)
        else:
            influence_views, mask, rt, views_in_segs = get_mask_from_influence(views_train, model, fs, track_resolu, L_data, L_feats, idx, ITER=i, CROSSVIEW=CROSSVIEW, coe=flip_coe, SAMEWEIGHT=SAMEWEIGHT, MIXPAIR=MIXPAIR)
        influence_list.append(influence_views[1][:,idx,:])
        mask_list.append(mask)
        rt_list.append(rt)
        model.weights_[1] = model.weights_[1][:L_feats, :]
        if SVAD:
            data_test, att_unatt_test = views_test
            att_test = att_unatt_test[:, :L_feats]
            unatt_test = att_unatt_test[:, L_feats:]
            corr_att_test = model.average_pairwise_correlations([data_test, att_test])
            corr_sum_att_list.append(cal_corr_sum(corr_att_test, range_into_account=evalpara[0], nb_comp_into_account=evalpara[1]))
            print(f'Corr_att_test: {corr_att_test}')
            corr_unatt_test = model.average_pairwise_correlations([data_test, unatt_test])
            corr_sum_unatt_list.append(cal_corr_sum(corr_unatt_test, range_into_account=evalpara[0], nb_comp_into_account=evalpara[1]))
            print(f'Corr_unatt_test: {corr_unatt_test}')
            if TWOENC:
                nb_correct, nb_trials = svad(views_test, model_two_enc, fs, compete_resolu, BOOTSTRAP=BOOTSTRAP, MIXPAIR=MIXPAIR, TWOENC=TWOENC, evalpara=evalpara)
            else:
                nb_correct, nb_trials = svad(views_test, model, fs, compete_resolu, BOOTSTRAP=BOOTSTRAP, MIXPAIR=MIXPAIR, TWOENC=TWOENC, evalpara=evalpara)
        else:
            corr_att_test = model.average_pairwise_correlations(views_test)
            corr_sum_att_list.append(cal_corr_sum(corr_att_test, range_into_account=evalpara[0], nb_comp_into_account=evalpara[1]))
            print(f'Corr_sum_att_test: {cal_corr_sum(corr_att_test, range_into_account=evalpara[0], nb_comp_into_account=evalpara[1])}')
            corr_sum_unatt_list.append(None)
            nb_correct, nb_trials = match_mismatch(views_test, model, fs, compete_resolu, BOOTSTRAP=BOOTSTRAP, evalpara=evalpara)
        nb_correct_list.append(nb_correct)
        nb_trials_list.append(nb_trials)
        updated_label, views_train = update_training_views(mask, views_in_segs, L_feats, true_label)
        updated_label_list.append(updated_label)
        nb_correct_train_list.append(np.sum(updated_label))
        nb_trials_train_list.append(len(updated_label))
        true_label = updated_label
        RANDINIT = False
        if rt > 0.95 and BREAK:
            break
    return model_list, influence_list, mask_list, updated_label_list, rt_list, corr_sum_att_list, corr_sum_unatt_list, nb_correct_train_list, nb_trials_train_list, nb_correct_list, nb_trials_list


def iterate_switch(views_train_ori, views_val_ori, views_test_ori, fs, track_resolu, compete_resolu, L_data, L_feats, SEED, SVAD=False, MAX_ITER=10, LWCOV=True, CROSSVIEW=True, coe=1, SAMEWEIGHT=False, latent_dimensions=5, evalpara=[3, 2], BOOTSTRAP=False, MIXPAIR=False, TWOENC=False, RANDINIT=False, UNBIASED_SE=False):
    views_train = copy.deepcopy(views_train_ori)
    views_val = copy.deepcopy(views_val_ori)
    views_test = copy.deepcopy(views_test_ori)
    rt_list = []
    corr_sum_att_list = []
    corr_sum_unatt_list = []
    nb_correct_train_list = []
    nb_trials_train_list = []
    nb_correct_list = []
    nb_trials_list = []
    true_label = None
    for i in range(MAX_ITER):
        model, corr_val, corr_train = train_cca_model(views_train, views_val, L_feats, LWCOV, latent_dimensions=latent_dimensions, evalpara=evalpara, RANDMODEL=RANDINIT, SEED=SEED)
        model_two_enc = copy.deepcopy(model)
        print(f'Corr_sum_train: {cal_corr_sum(corr_train, range_into_account=evalpara[0], nb_comp_into_account=evalpara[1])}')
        if corr_val is not None:
            print(f'Corr_sum_val: {cal_corr_sum(corr_val, range_into_account=evalpara[0], nb_comp_into_account=evalpara[1])}')
        idx = np.argmax(corr_val)
        flip_coe = coe if not RANDINIT else 1
        influence_views, mask, rt, views_in_segs = get_mask_from_influence(views_train, model, fs, track_resolu, L_data, L_feats, idx, ITER=i, CROSSVIEW=CROSSVIEW, coe=flip_coe, SAMEWEIGHT=SAMEWEIGHT, MIXPAIR=MIXPAIR)
        rt_list.append(rt)
        model.weights_[1] = model.weights_[1][:L_feats, :]
        if SVAD:
            data_test, att_unatt_test = views_test
            att_test = att_unatt_test[:, :L_feats]
            unatt_test = att_unatt_test[:, L_feats:]
            corr_att_test = model.average_pairwise_correlations([data_test, att_test])
            corr_sum_att_list.append(cal_corr_sum(corr_att_test, range_into_account=evalpara[0], nb_comp_into_account=evalpara[1]))
            print(f'Corr_att_test: {corr_att_test}')
            corr_unatt_test = model.average_pairwise_correlations([data_test, unatt_test])
            corr_sum_unatt_list.append(cal_corr_sum(corr_unatt_test, range_into_account=evalpara[0], nb_comp_into_account=evalpara[1]))
            print(f'Corr_unatt_test: {corr_unatt_test}')
            if TWOENC:
                nb_correct, nb_trials = svad(views_test, model_two_enc, fs, compete_resolu, BOOTSTRAP=BOOTSTRAP, MIXPAIR=MIXPAIR, TWOENC=TWOENC, evalpara=evalpara)
            else:
                nb_correct, nb_trials = svad(views_test, model, fs, compete_resolu, BOOTSTRAP=BOOTSTRAP, MIXPAIR=MIXPAIR, TWOENC=TWOENC, evalpara=evalpara)
        else:
            corr_att_test = model.average_pairwise_correlations(views_test)
            corr_sum_att_list.append(cal_corr_sum(corr_att_test, range_into_account=evalpara[0], nb_comp_into_account=evalpara[1]))
            print(f'Corr_sum_att_test: {cal_corr_sum(corr_att_test, range_into_account=evalpara[0], nb_comp_into_account=evalpara[1])}')
            corr_sum_unatt_list.append(None)
            nb_correct, nb_trials = match_mismatch(views_test, model, fs, compete_resolu, BOOTSTRAP=BOOTSTRAP, evalpara=evalpara)
        nb_correct_list.append(nb_correct)
        nb_trials_list.append(nb_trials)
        updated_label, views_train = update_training_views(mask, views_in_segs, L_feats, true_label)
        nb_correct_train_list.append(np.sum(updated_label))
        nb_trials_train_list.append(len(updated_label))
        true_label = updated_label
        RANDINIT = False
        if rt > 0.95:
            break
    nb_iter = i+1
    if nb_iter < MAX_ITER:
        views_train_single = [views_train[0], views_train[1][:, :L_feats], views_train[1][:, L_feats:]]
        views_val_single = [views_val_ori[0], views_val_ori[1][:, :L_feats], views_val_ori[1][:, L_feats:]] if views_val_ori is not None else None
        views_test_single = [views_test_ori[0], views_test_ori[1][:, :L_feats], views_test_ori[1][:, L_feats:]]
        _, _, _, _, _, corr_sum_att_clist, corr_sum_unatt_clist, nb_correct_train_clist, nb_trials_train_clist, nb_correct_clist, nb_trials_clist = utils_unsup_single_enc.iterate(views_train_single, views_val_single, views_test_single, fs, track_resolu, compete_resolu, SEED, SVAD=SVAD, MAX_ITER=(MAX_ITER-nb_iter), LWCOV=LWCOV, coe=coe, latent_dimensions=latent_dimensions, evalpara=evalpara, BOOTSTRAP=BOOTSTRAP, MIXPAIR=MIXPAIR, RANDINIT=RANDINIT, UNBIASED=UNBIASED_SE, true_label=true_label)
        corr_sum_att_list.extend(corr_sum_att_clist)
        corr_sum_unatt_list.extend(corr_sum_unatt_clist)
        nb_correct_train_list.extend(nb_correct_train_clist)
        nb_trials_train_list.extend(nb_trials_train_clist)
        nb_correct_list.extend(nb_correct_clist)
        nb_trials_list.extend(nb_trials_clist)
    return corr_sum_att_list, corr_sum_unatt_list, nb_correct_train_list, nb_trials_train_list, nb_correct_list, nb_trials_list


def recursive(views_train_ori, views_test_ori, fs, track_resolu, compete_resolu, L_data, L_feats, SEED, latent_dimensions=5, weightpara=[0.0, 0.0], evalpara=[3, 2], CROSSVIEW=True, BOOTSTRAP=False, MIXPAIR=False):
    T_track = fs*track_resolu
    views_train = copy.deepcopy(views_train_ori)
    views_test = copy.deepcopy(views_test_ori)
    views_in_segs_train = [get_segments(view, fs, track_resolu, MIXPAIR=MIXPAIR) for view in views_train]
    nb_views = len(views_in_segs_train)
    nb_segs_train = len(views_in_segs_train[0])
    segs_views_train = [[views_in_segs_train[i][j] for i in range(nb_views)] for j in range(nb_segs_train)]
    # generate a random encoder and decoder
    model = train_cca_model_adaptive(segs_views_train[0], None, None, latent_dimensions=latent_dimensions, weightpara=weightpara, RANDMODEL=True, SEED=SEED)
    Rinit = None
    Dinit = None
    labels = []
    nb_correct_list = []
    nb_trials_list = []
    for i in range(0, nb_segs_train):
        weights = copy.deepcopy(model.weights_)
        # get accuracy in the test set
        model.weights_[1] = model.weights_[1][:L_feats, :]
        nb_correct, nb_trials = svad(views_test, model, fs, compete_resolu, BOOTSTRAP=BOOTSTRAP, MIXPAIR=MIXPAIR, evalpara=evalpara)
        nb_correct_list.append(nb_correct)
        nb_trials_list.append(nb_trials)
        # Acquire next segment, predict the label
        seg_to_pred = segs_views_train[i]
        influence_views = utils.get_influence_all_views(seg_to_pred, weights, [L_data, L_feats], 'SDP', CROSSVIEW=CROSSVIEW, NORMALIZATION=False)
        idx = 0
        label = influence_views[1][0, idx] > influence_views[1][1, idx]
        labels = np.append(labels, label)
        if label:
            seg_predicted = seg_to_pred
        else:
            seg_predicted = [seg_to_pred[0], np.concatenate([seg_to_pred[1][:, L_feats:], seg_to_pred[1][:, :L_feats]], axis=1)]
        # update the model
        model = train_cca_model_adaptive(seg_predicted, Rinit, Dinit, latent_dimensions=latent_dimensions, weightpara=weightpara, RANDMODEL=False, SEED=SEED)
        Rinit = model.Rxx
        Dinit = model.Dxx
    model.weights_[1] = model.weights_[1][:L_feats, :]
    nb_correct, nb_trials = svad(views_test, model, fs, compete_resolu, BOOTSTRAP=BOOTSTRAP, MIXPAIR=MIXPAIR, evalpara=evalpara)
    nb_correct_list.append(nb_correct)
    nb_trials_list.append(nb_trials)
    return labels, nb_correct_list, nb_trials_list

