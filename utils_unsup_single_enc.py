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


def prepare_train_val_test_data(Subj_ID, MOD, modal_dict, modal_dict_SO, feat_att_list, feat_unatt_list, feat_SO_list, hparas, fs, trainmin=None):
    # Load data for the specific subject
    data_onesubj_list = [data[:,:,Subj_ID] for data in modal_dict[MOD]]
    data_onesubj_SO_list = [data[:,:,Subj_ID] for data in modal_dict_SO[MOD]]
    # Use SI data for training
    data_train = np.concatenate(tuple(data_onesubj_list), axis=0)
    feats_att_train = np.concatenate(tuple(feat_att_list), axis=0)
    feats_unatt_train = np.concatenate(tuple(feat_unatt_list), axis=0)
    if trainmin is not None:
        nb_samples = int(fs*60*trainmin)
        data_train = data_train[:nb_samples,:]
        feats_att_train = feats_att_train[:nb_samples,:]
        feats_unatt_train = feats_unatt_train[:nb_samples,:]
    # Use the SO data for validation and testing
    test_list_folds, val_list_folds = utils.split_multi_mod_LVO([data_onesubj_SO_list, feat_SO_list], leave_out=2)
    data_test, feats_test = test_list_folds[0]
    data_val, feats_val = val_list_folds[0]
    L_data, offset_data = hparas[0]
    L_feats, offset_feats = hparas[1]
    views_train = [process_data_per_view(data_train, L_data, offset_data, NORMALIZE=True), process_data_per_view(feats_att_train, L_feats, offset_feats, NORMALIZE=True), process_data_per_view(feats_unatt_train, L_feats, offset_feats, NORMALIZE=True)]
    views_val = [process_data_per_view(data_val, L_data, offset_data, NORMALIZE=True), process_data_per_view(feats_val, L_feats, offset_feats, NORMALIZE=True)]
    views_test = [process_data_per_view(data_test, L_data, offset_data, NORMALIZE=True), process_data_per_view(feats_test, L_feats, offset_feats, NORMALIZE=True)]
    return views_train, views_val, views_test


def prepare_train_val_test_data_svad(Subj_ID, MOD, modal_dict, feat_att_list, feat_unatt_list, hparas, fs, leave_out=2, trainmin=None, LWCOV=False):
    # Load data for the specific subject
    data_onesubj_list = [data[:,:,Subj_ID] for data in modal_dict[MOD]]
    # Split the SI data into training and validation sets
    train_list_folds, test_list_folds, val_list_folds = utils.split_multi_mod_withval_LVO([data_onesubj_list, feat_att_list, feat_unatt_list], leave_out=leave_out, VAL=not LWCOV)
    nb_videos = len(feat_att_list)
    L_data, offset_data = hparas[0]
    L_feats, offset_feats = hparas[1]
    views_train_folds = []
    views_val_folds = []
    views_test_folds = []
    for fold in range(nb_videos//leave_out):
        data_train, feats_att_train, feats_unatt_train = train_list_folds[fold]
        if trainmin is not None:
            nb_samples = int(fs*60*trainmin)
            data_train = data_train[:nb_samples,:]
            feats_att_train = feats_att_train[:nb_samples,:]
            feats_unatt_train = feats_unatt_train[:nb_samples,:]
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


def cal_corr_sum(corr, range_into_account=3, nb_comp_into_account=2):
    corr_ranked = np.sort(corr[:range_into_account])[::-1]
    corr_sum = np.sum(corr_ranked[:nb_comp_into_account])
    return corr_sum


def get_rand_model(model, SEED):
    rng = np.random.RandomState(SEED)
    rand_model = copy.deepcopy(model)
    rand_model.weights_ = [rng.randn(*W.shape) for W in model.weights_]
    return rand_model


def train_cca_model(views_train, views_val, LWCOV=False, latent_dimensions=5, evalpara=[3, 2], RANDMODEL=False, SEED=None):
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
    best_corr_train = best_model.average_pairwise_correlations([data_train, feats_att_train])
    print(f'Corr_train: {best_corr_train}')
    return best_model, best_corr_val, best_corr_train


def train_cca_model_adaptive(views_train, Rinit, Dinit, latent_dimensions=5, weightpara=[0.0, 0.0], RANDMODEL=False, SEED=None):
    data_train, feats_att_train, _ = views_train
    best_model = MCCA_LW(latent_dimensions=latent_dimensions, alpha=weightpara[0], beta=weightpara[1])
    best_model.fit([data_train, feats_att_train], Rinit=Rinit, Dinit=Dinit)
    if RANDMODEL:
        assert SEED is not None, 'SEED must be provided for random initialization'
        best_model = get_rand_model(best_model, SEED)
    best_corr_train = best_model.average_pairwise_correlations([data_train, feats_att_train])
    print(f'Corr_train: {best_corr_train}')
    return best_model


def get_corr_pair(seg_views, model, evalpara=[3, 2]):
    data, feats_att, feats_unatt = seg_views
    corr_att = model.average_pairwise_correlations([data, feats_att])
    corr_unatt = model.average_pairwise_correlations([data, feats_unatt])
    corr_sum_att = cal_corr_sum(corr_att, range_into_account=evalpara[0], nb_comp_into_account=evalpara[1])
    corr_sum_unatt = cal_corr_sum(corr_unatt, range_into_account=evalpara[0], nb_comp_into_account=evalpara[1])
    corr_sum_pair = np.stack([corr_sum_att, corr_sum_unatt], axis=0)
    return corr_sum_pair


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


def get_mask_from_corr(views_train, model, fs, track_resolu, ITER, coe=1, evalpara=[3, 2], MIXPAIR=False):
    # Convert views into trials with overlap
    views_in_segs = [get_segments(view, fs, track_resolu, MIXPAIR=MIXPAIR) for view in views_train]
    nb_views = len(views_in_segs)
    nb_segs = len(views_in_segs[0])
    # Get views in each segment
    segs_views = [[views_in_segs[i][j] for i in range(nb_views)] for j in range(nb_segs)]
    corr_sum_pairs = [get_corr_pair(seg_views, model, evalpara=evalpara) for seg_views in segs_views]
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
    print(f'Ratio of Predicted Att: {rt}')
    return corr_sum_pairs, mask, rt, views_in_segs


def get_mask_unbiased(views_train, fs, track_resolu, ITER, coe=1, evalpara=[3, 2], latent_dimensions=5, RANDINIT=False, SEED=None, MIXPAIR=False):
    # Convert views into trials with overlap
    views_in_segs = views_in_segs = [get_segments(view, fs, track_resolu, MIXPAIR=MIXPAIR) for view in views_train]
    nb_views = len(views_in_segs)
    nb_segs = len(views_in_segs[0])
    corr_sum_pairs = np.zeros((nb_segs, 2))
    for i in range(nb_segs):
        views_test = [views_in_segs[j][i] for j in range(nb_views)]
        views_train = [np.concatenate([views_in_segs[j][k] for k in range(nb_segs) if k != i], axis=0) for j in range(nb_views)]
        views_val = None
        LWCOV = True
        model, _, _ = train_cca_model(views_train, views_val, LWCOV, latent_dimensions=latent_dimensions, evalpara=evalpara, RANDMODEL=RANDINIT, SEED=SEED)
        corr_sum_pair = get_corr_pair(views_test, model, evalpara=evalpara)
        corr_sum_pairs[i,:] = corr_sum_pair
    corr_diff = corr_sum_pairs[:, 0] - corr_sum_pairs[:, 1]
    mask = corr_diff > 0
    nb_detected_seg = np.sum(corr_diff < 0)
    # sort the influence difference from the smallest to the largest
    idx_sort = np.argsort(corr_diff)
    coe = 0.5**(ITER) if coe is None else coe
    idx_keep = idx_sort[int(coe*nb_detected_seg):]
    mask[idx_keep] = True
    rt = 1 - nb_detected_seg / len(corr_diff)
    print(f'Ratio of Predicted Att: {rt}')
    return corr_sum_pairs, mask, rt, views_in_segs


def update_training_views(mask, views_in_segs, true_label):
    updated_label = true_label==mask if true_label is not None else mask
    print('Acc (train): ', np.sum(updated_label)/len(updated_label))
    for i, indicator in enumerate(mask):
        if not indicator:
            feats_att_seg_i = views_in_segs[1][i].copy()
            views_in_segs[1][i] = views_in_segs[2][i]
            views_in_segs[2][i] = feats_att_seg_i
    data_h_train_updated = np.concatenate(tuple(views_in_segs[0]), axis=0)
    feats_h_att_train_updated = np.concatenate(tuple(views_in_segs[1]), axis=0)
    feats_h_unatt_train_updated = np.concatenate(tuple(views_in_segs[2]), axis=0)
    views_train_updated = [data_h_train_updated, feats_h_att_train_updated, feats_h_unatt_train_updated]
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


def svad(views, model, fs, trial_len, overlap=0, BOOTSTRAP=False, MIXPAIR=False, evalpara=[3, 2]):
    if BOOTSTRAP:
        T = views[0].shape[0]
        nb_trials = int(T/fs*3)
        if MIXPAIR:
            start_points = np.random.randint(0, T//2-trial_len//2*fs, size=nb_trials)
        else:
            start_points = np.random.randint(0, T-trial_len*fs, size=nb_trials)
    else:
        start_points = None
    data_seg, att_seg, unatt_seg = [get_segments(view, fs, trial_len, overlap=overlap, MIXPAIR=MIXPAIR, start_points=start_points) for view in views]
    nb_trials = len(data_seg)
    corr_a = [model.average_pairwise_correlations([d, f]) for d, f in zip(data_seg, att_seg)]
    corr_u = [model.average_pairwise_correlations([d, f]) for d, f in zip(data_seg, unatt_seg)]
    acc = utils.eval_compete(np.stack(corr_a, axis=0), np.stack(corr_u, axis=0), TRAIN_WITH_ATT=True, range_into_account=evalpara[0], nb_comp_into_account=evalpara[1])
    nb_correct = round(acc * nb_trials)
    return nb_correct, nb_trials


def iterate(views_train_ori, views_val_ori, views_test_ori, fs, track_resolu, compete_resolu, SEED, SVAD=False, MAX_ITER=10, LWCOV=True, coe=1, latent_dimensions=5, evalpara=[3, 2], BOOTSTRAP=False, MIXPAIR=False, RANDINIT=False, UNBIASED=False, true_label=None):
    views_train = copy.deepcopy(views_train_ori)
    views_val = copy.deepcopy(views_val_ori)
    views_test = copy.deepcopy(views_test_ori)
    model_list = []
    corr_pair_list = []
    mask_list = []
    updated_label_list = []
    rt_list = []
    corr_sum_att_list = []
    corr_sum_unatt_list = []
    nb_correct_train_list = []
    nb_trials_train_list = []
    nb_correct_list = []
    nb_trials_list = []
    for i in range(MAX_ITER):
        model, corr_val, corr_train = train_cca_model(views_train, views_val, LWCOV, latent_dimensions=latent_dimensions, evalpara=evalpara, RANDMODEL=RANDINIT, SEED=SEED)
        model_list.append(model)
        print(f'Corr_sum_train: {cal_corr_sum(corr_train, range_into_account=evalpara[0], nb_comp_into_account=evalpara[1])}')
        if corr_val is not None:
            print(f'Corr_sum_val: {cal_corr_sum(corr_val, range_into_account=evalpara[0], nb_comp_into_account=evalpara[1])}')
        flip_coe = coe if not RANDINIT else 1
        if UNBIASED:
            corr_sum_pairs, mask, rt, views_in_segs = get_mask_unbiased(views_train, fs, track_resolu, ITER=i, coe=flip_coe, evalpara=evalpara, latent_dimensions=latent_dimensions, RANDINIT=RANDINIT, SEED=SEED, MIXPAIR=MIXPAIR)
        else:
            corr_sum_pairs, mask, rt, views_in_segs = get_mask_from_corr(views_train, model, fs, track_resolu, ITER=i, coe=flip_coe, evalpara=evalpara, MIXPAIR=MIXPAIR)
        corr_pair_list.append(corr_sum_pairs)
        mask_list.append(mask)
        rt_list.append(rt)
        if SVAD:
            corr_att_test = model.average_pairwise_correlations(views_test[:2])
            corr_sum_att_list.append(cal_corr_sum(corr_att_test, range_into_account=evalpara[0], nb_comp_into_account=evalpara[1]))
            print(f'Corr_att_test: {corr_att_test}')
            corr_unatt_test = model.average_pairwise_correlations([views_test[0], views_test[2]])
            corr_sum_unatt_list.append(cal_corr_sum(corr_unatt_test, range_into_account=evalpara[0], nb_comp_into_account=evalpara[1]))
            print(f'Corr_unatt_test: {corr_unatt_test}')
            nb_correct, nb_trials = svad(views_test, model, fs, compete_resolu, BOOTSTRAP=BOOTSTRAP, MIXPAIR=MIXPAIR, evalpara=evalpara)
        else:
            corr_att_test = model.average_pairwise_correlations(views_test)
            corr_sum_att_list.append(cal_corr_sum(corr_att_test, range_into_account=evalpara[0], nb_comp_into_account=evalpara[1]))
            print(f'Corr_att_test: {corr_att_test}')
            corr_sum_unatt_list.append(None)
            nb_correct, nb_trials = match_mismatch(views_test, model, fs, compete_resolu, BOOTSTRAP=BOOTSTRAP, evalpara=evalpara)
        nb_correct_list.append(nb_correct)
        nb_trials_list.append(nb_trials)
        updated_label, views_train = update_training_views(mask, views_in_segs, true_label)
        updated_label_list.append(updated_label)
        nb_correct_train_list.append(np.sum(updated_label))
        nb_trials_train_list.append(len(updated_label))
        true_label = updated_label
        RANDINIT = False
    return model_list, corr_pair_list, mask_list, updated_label_list, rt_list, corr_sum_att_list, corr_sum_unatt_list, nb_correct_train_list, nb_trials_train_list, nb_correct_list, nb_trials_list


def recursive(views_train_ori, views_test_ori, fs, track_resolu, compete_resolu, SEED, pool_size, latent_dimensions=5, weightpara=[0.0, 0.0], evalpara=[3, 2], BOOTSTRAP=False, MIXPAIR=False):
    T_track = fs*track_resolu
    views_train = copy.deepcopy(views_train_ori)
    views_test = copy.deepcopy(views_test_ori)
    views_in_segs_train = [get_segments(view, fs, track_resolu, MIXPAIR=MIXPAIR) for view in views_train]
    nb_views = len(views_in_segs_train)
    nb_segs_train = len(views_in_segs_train[0])
    assert nb_segs_train>pool_size, 'The number of segments in the training set must be larger than the pool size'
    segs_views_train = [[views_in_segs_train[i][j] for i in range(nb_views)] for j in range(nb_segs_train)]
    Rinit = None
    Dinit = None
    pool_segs = [segs[:pool_size] for segs in views_in_segs_train]
    # get random mask (len = pool_size) for the first pool
    rng = np.random.RandomState(SEED)
    mask = rng.choice([False, True], size=pool_size)
    true_label = [True]*pool_size
    labels, pool = update_training_views(mask, pool_segs, true_label)
    nb_correct_list = []
    nb_trials_list = []
    for i in range(pool_size, nb_segs_train):
        seg_to_pred = segs_views_train[i]
        model = train_cca_model_adaptive(pool, Rinit, Dinit, latent_dimensions=latent_dimensions, weightpara=weightpara, RANDMODEL=False, SEED=SEED)
        Rinit = model.Rxx
        Dinit = model.Dxx
        # predict the label of the next segment
        corr_sum_pair = get_corr_pair(seg_to_pred, model, evalpara=evalpara)
        label = corr_sum_pair[0] > corr_sum_pair[1]
        labels = np.append(labels, label)
        if label:
            att = np.concatenate([pool[1], seg_to_pred[1]], axis=0)
            unatt = np.concatenate([pool[2], seg_to_pred[2]], axis=0)
        else:
            att = np.concatenate([pool[1], seg_to_pred[2]], axis=0)
            unatt = np.concatenate([pool[2], seg_to_pred[1]], axis=0)
        eeg = np.concatenate([pool[0], seg_to_pred[0]], axis=0)
        pool = [eeg[T_track:,:], att[T_track:,:], unatt[T_track:,:]]
        # predict the labels of the segments in the test set
        nb_correct, nb_trials = svad(views_test, model, fs, compete_resolu, BOOTSTRAP=BOOTSTRAP, MIXPAIR=MIXPAIR, evalpara=evalpara)
        nb_correct_list.append(nb_correct)
        nb_trials_list.append(nb_trials)
    model = train_cca_model_adaptive(pool, Rinit, Dinit, latent_dimensions=latent_dimensions, weightpara=weightpara, RANDMODEL=False, SEED=SEED)
    nb_correct, nb_trials = svad(views_test, model, fs, compete_resolu, BOOTSTRAP=BOOTSTRAP, MIXPAIR=MIXPAIR, evalpara=evalpara)
    nb_correct_list.append(nb_correct)
    nb_trials_list.append(nb_trials)
    return labels, nb_correct_list, nb_trials_list

