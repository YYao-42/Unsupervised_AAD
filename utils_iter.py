import utils
import utils_prob
import numpy as np
import copy
from algo_suppl import MCCA_LW
from itertools import product


def further_split_and_shuffle(data_trials, feats_trials, labels_trials, sub_trial_length, fs, SHUFFLE=False, SEED=None):
    """
    Further split the data into smaller chunks and shuffle them.
    """
    nb_trials = len(data_trials)
    data = []
    feats = []
    labels = []
    T = data_trials[0].shape[0]
    nb_sub_segs = T // (sub_trial_length * fs)
    assert T % (sub_trial_length * fs) == 0, "The trial len must be divisible by the sub trial len."
    # Split the data into smaller chunks
    for i in range(nb_trials):
        data_subsegs = utils.into_trials(data_trials[i], fs, t=sub_trial_length)
        feats_subsegs = utils.into_trials(feats_trials[i], fs, t=sub_trial_length)
        labels_subsegs = np.repeat(labels_trials[i], nb_sub_segs)
        data += data_subsegs
        feats += feats_subsegs
        labels += list(labels_subsegs)
    if SHUFFLE:
        # Shuffle the data
        rng = np.random.default_rng(SEED)
        indices = np.arange(len(data))
        rng.shuffle(indices)
        data_shuffled = [data[i] for i in indices]
        feats_shuffled = [feats[i] for i in indices]
        labels_shuffled = [labels[i] for i in indices]
        return data_shuffled, feats_shuffled, labels_shuffled
    else:
        return data, feats, labels


def add_label_noise(label_list, noise_rate, rng):
    """
    Add noise to the labels
    """
    label_copy = copy.deepcopy(label_list)
    if noise_rate is not None:
        nb_trials = len(label_copy)
        nb_noisy_labels = int(nb_trials * noise_rate)
        indices = rng.choice(nb_trials, nb_noisy_labels, replace=False)
        for i in indices:
            label_copy[i] = 3 - label_copy[i]  # flip the label
    return label_copy


def process_data_per_view(view, L, offset, NORMALIZE=True):
    view_hankelized = utils.block_Hankel(view, L, offset)
    if NORMALIZE:
        view_hankelized = utils.normalize_per_view(view_hankelized)
    return view_hankelized


def get_feats_per_stream(feats):
    dim_feats = feats.shape[1]//2
    f1 = feats[:, :dim_feats]
    f2 = feats[:, dim_feats:]
    return f1, f2


def switch_f1_f2(feats):
    f1, f2 = get_feats_per_stream(feats)
    feats_switched = np.concatenate([f2, f1], axis=1)
    return feats_switched


def select_att_unatt_feats(feats_trials, label_trials):
    nb_trials = len(feats_trials)
    att_trials = []
    unatt_trials = []
    for i in range(nb_trials):
        feats = feats_trials[i]
        dim_hankel = feats.shape[1]//2
        att = feats[:, (label_trials[i]-1)*dim_hankel:label_trials[i]*dim_hankel]
        unatt = feats[:, (2-label_trials[i])*dim_hankel:(3-label_trials[i])*dim_hankel]
        att_trials.append(np.expand_dims(att, axis=1) if len(att.shape) == 1 else att)
        unatt_trials.append(np.expand_dims(unatt, axis=1) if len(unatt.shape) == 1 else unatt)
    return att_trials, unatt_trials


def cal_corr_sum(corr, range_into_account=3, nb_comp_into_account=2):
    corr_ranked = np.sort(corr[:range_into_account])[::-1]
    corr_sum = np.sum(corr_ranked[:nb_comp_into_account])
    return corr_sum


def get_rand_model(model, SEED):
    rng = np.random.RandomState(SEED)
    rand_model = copy.deepcopy(model)
    rand_model.weights_ = [rng.randn(*W.shape) for W in model.weights_]
    return rand_model


def train_cca_model(views_train, latent_dimensions=5, RANDMODEL=False, SEED=None, SINGLEENC=False):
    if SINGLEENC:
        data_train, att_unatt = views_train
        att, _ = get_feats_per_stream(att_unatt)
        views_train = [data_train, att]
    best_model = MCCA_LW(latent_dimensions=latent_dimensions)
    best_model.fit(views_train)
    if RANDMODEL:
        assert SEED is not None, 'SEED must be provided for random initialization'
        best_model = get_rand_model(best_model, SEED)
    best_corr_train = best_model.average_pairwise_correlations(views_train)
    # print(f'Corr_train: {best_corr_train}')
    return best_model


def train_cca_model_adaptive(views_train, Rinit, Dinit, latent_dimensions=5, weightpara=[0.0, 0.0], RANDMODEL=False, SEED=None, SINGLEENC=False):
    if SINGLEENC:
        data_train, att_unatt = views_train
        att, _ = get_feats_per_stream(att_unatt)
        views_train = [data_train, att]
    best_model = MCCA_LW(latent_dimensions=latent_dimensions, alpha=weightpara[0], beta=weightpara[1])
    best_model.fit(views_train, Rinit=Rinit, Dinit=Dinit)
    if RANDMODEL:
        assert SEED is not None, 'SEED must be provided for random initialization'
        best_model = get_rand_model(best_model, SEED)
    best_corr_train = best_model.average_pairwise_correlations(views_train)
    # print(f'Corr_train: {best_corr_train}')
    return best_model


def predict_labels(views, model, L_data, L_feats, evalpara, NEWSEG=False):
    data, feats = views
    if NEWSEG:
        f1, f2 = get_feats_per_stream(feats)
        model_single_enc = copy.deepcopy(model)
        model_single_enc.weights_[1] = model.weights_[1][:L_feats, :]
        corr_1 = model_single_enc.average_pairwise_correlations([data, f1])
        corr_sum_1 = cal_corr_sum(corr_1, evalpara[0], evalpara[1])
        corr_2 = model_single_enc.average_pairwise_correlations([data, f2])
        corr_sum_2 = cal_corr_sum(corr_2, evalpara[0], evalpara[1])
        pred_label = 1 if corr_sum_1 > corr_sum_2 else 2
    else:
        influ = utils.get_influence_all_views([data, feats], model.weights_, [L_data, L_feats], 'SDP', CROSSVIEW=True, NORMALIZATION=False)[1]
        pred_label = 1 if influ[0,0] > influ[1,0] else 2
    return pred_label


def predict_labels_single_enc(views, model, evalpara):
    data, feats = views
    f1, f2 = get_feats_per_stream(feats)
    corr_1 = model.average_pairwise_correlations([data, f1])
    corr_sum_1 = cal_corr_sum(corr_1, evalpara[0], evalpara[1])
    corr_2 = model.average_pairwise_correlations([data, f2])
    corr_sum_2 = cal_corr_sum(corr_2, evalpara[0], evalpara[1])
    pred_label = 1 if corr_sum_1 > corr_sum_2 else 2
    return pred_label


def predict_labels_unbiased(segs_views, labels, L_data, L_feats, evalpara, SINGLEENC=False):
    nb_segs = len(segs_views)
    pred_labels = []
    for i in range(nb_segs):
        data, feats = segs_views[i]
        data_train = np.concatenate([views[0] for j, views in enumerate(segs_views) if j != i], axis=0)
        feats_train_trials = [views[1] for j, views in enumerate(segs_views) if j != i]
        labels_train = [labels[j] for j in range(nb_segs) if j != i]
        att_train_trials, unatt_train_trials = select_att_unatt_feats(feats_train_trials, labels_train)
        feats_train_trials = [np.concatenate([att, unatt], axis=1) for att, unatt in zip(att_train_trials, unatt_train_trials)]
        feats_train = np.concatenate(feats_train_trials, axis=0)
        model = train_cca_model([data_train, feats_train], latent_dimensions=5, SINGLEENC=SINGLEENC)
        if SINGLEENC:
            pred_label = predict_labels_single_enc([data, feats], model, evalpara)
        else:
            pred_label = predict_labels([data, feats], model, L_data, L_feats, evalpara)
        pred_labels.append(pred_label)
    return pred_labels


def get_class_priors(labels, confi):
    # check if all elements in labels are None
    if all(label is None for label in labels):
        class_0 = 0.5
        class_1 = 0.5
    else:
        labels_sum = np.sum(labels)
        nb_labels = len(labels)
        class_1 = (3*confi - 1) - labels_sum*(2*confi - 1)/nb_labels
        class_0 = 1 - class_1
    return np.array([class_0, class_1])


def predict_labels_soft(views, model, gmm_0, gmm_1, evalpara, class_priors=None):
    data, feats = views
    f1, f2 = get_feats_per_stream(feats)
    corr_1 = model.average_pairwise_correlations([data, f1])
    corr_sum_1 = cal_corr_sum(corr_1, evalpara[0], evalpara[1])
    corr_2 = model.average_pairwise_correlations([data, f2])
    corr_sum_2 = cal_corr_sum(corr_2, evalpara[0], evalpara[1])
    probas = utils_prob.predict_proba(np.array([corr_sum_1, corr_sum_2]), gmm_0, gmm_1, class_priors=class_priors)
    pred_label = 1 if probas[0, 1] > probas[0, 0] else 2
    return probas, pred_label


def update_seg_soft(views, probas):
    data, feats = views
    f1, f2 = get_feats_per_stream(feats)
    att_predicted = f1*probas[0, 1] + f2*probas[0, 0]
    unatt_predicted = f2*probas[0, 1] + f1*probas[0, 0]
    feats_weighted = np.concatenate([att_predicted, unatt_predicted], axis=1)
    return [data, feats_weighted]


class ITERATIVE:
    def __init__(self, data_train_trials, data_test_trials, feats_train_trials, feats_test_trials, labels_test_trials, L_data, L_feats, latent_dimensions, SEED, evalpara, ITERS):
        self.data_train_trials = data_train_trials
        self.data_test_trials = data_test_trials
        self.feats_train_trials = feats_train_trials
        self.feats_test_trials = feats_test_trials
        self.labels_test_trials = labels_test_trials
        self.L_data = L_data
        self.L_feats = L_feats
        self.latent_dimensions = latent_dimensions
        self.SEED = SEED
        self.evalpara = evalpara
        self.ITERS = ITERS

    def supervised(self, labels_train_trials):
        att_train_trials, unatt_train_trials = select_att_unatt_feats(self.feats_train_trials, labels_train_trials)
        feats_train_trials = [np.concatenate([att, unatt], axis=1) for att, unatt in zip(att_train_trials, unatt_train_trials)]
        data_train = np.concatenate(self.data_train_trials, axis=0)
        feats_train = np.concatenate(feats_train_trials, axis=0)
        model = train_cca_model([data_train, feats_train], latent_dimensions=self.latent_dimensions, SINGLEENC=True)
        segs_views = [[data, feats] for data, feats in zip(self.data_test_trials, self.feats_test_trials)]
        pred_labels = [predict_labels_single_enc(views, model, self.evalpara) for views in segs_views]
        return pred_labels

    def unsupervised(self, model_init=None, SINGLEENC=True):
        data_train = np.concatenate(self.data_train_trials, axis=0)
        feats_train = np.concatenate(self.feats_train_trials, axis=0)
        segs_views = [[data, feats] for data, feats in zip(self.data_test_trials, self.feats_test_trials)]
        if model_init is not None:
            model = model_init
            Rinit = Rinit
            Dinit = Dinit
        else:
            model = train_cca_model([data_train, feats_train], latent_dimensions=self.latent_dimensions, RANDMODEL=True, SEED=self.SEED, SINGLEENC=SINGLEENC)
            Rinit = None 
            Dinit = None
        pred_labels_iters = []
        for i in range(self.ITERS):
            pred_labels = np.array([predict_labels_single_enc(views, model, self.evalpara) if SINGLEENC else predict_labels(views, model, self.L_data, self.L_feats, self.evalpara, NEWSEG=True) for views in segs_views])
            pred_labels_iters.append(pred_labels)
            # reprediction & retraining
            segs_views_train = [[data, feats] for data, feats in zip(self.data_train_trials, self.feats_train_trials)]
            repred_labels = [predict_labels_single_enc(views, model, self.evalpara) if SINGLEENC else predict_labels(views, model, self.L_data, self.L_feats, self.evalpara, NEWSEG=False) for views in segs_views_train]
            att_trials, unatt_trials = select_att_unatt_feats(self.feats_train_trials, repred_labels)
            feats_train = np.concatenate([np.concatenate([att, unatt], axis=1) for att, unatt in zip(att_trials, unatt_trials)], axis=0)
            model = train_cca_model([data_train, feats_train], latent_dimensions=self.latent_dimensions, RANDMODEL=False, SEED=self.SEED, SINGLEENC=SINGLEENC)
        pred_labels = np.array([predict_labels_single_enc(views, model, self.evalpara) if SINGLEENC else predict_labels(views, model, self.L_data, self.L_feats, self.evalpara, NEWSEG=True) for views in segs_views])
        pred_labels_iters.append(pred_labels)
        pred_labels_iters = np.stack(pred_labels_iters, axis=0)
        return pred_labels_iters

    def unbiased(self, model_init=None, SINGLEENC=True):
        data_train = np.concatenate(self.data_train_trials, axis=0)
        feats_train = np.concatenate(self.feats_train_trials, axis=0)
        segs_views = [[data, feats] for data, feats in zip(self.data_test_trials, self.feats_test_trials)]
        if model_init is not None:
            model = model_init
            Rinit = Rinit
            Dinit = Dinit
        else:
            model = train_cca_model([data_train, feats_train], latent_dimensions=self.latent_dimensions, RANDMODEL=True, SEED=self.SEED, SINGLEENC=SINGLEENC)
            Rinit = None 
            Dinit = None
        pred_labels_iters = []
        for i in range(self.ITERS):
            pred_labels = np.array([predict_labels_single_enc(views, model, self.evalpara) if SINGLEENC else predict_labels(views, model, self.L_data, self.L_feats, self.evalpara, NEWSEG=True) for views in segs_views])
            pred_labels_iters.append(pred_labels)
            # reprediction & retraining
            segs_views_train = [[data, feats] for data, feats in zip(self.data_train_trials, self.feats_train_trials)]
            pred_labels_train = np.array([predict_labels_single_enc(views, model, self.evalpara) if SINGLEENC else predict_labels(views, model, self.L_data, self.L_feats, self.evalpara, NEWSEG=True) for views in segs_views_train])
            repred_labels = predict_labels_unbiased(segs_views_train, pred_labels_train, self.L_data, self.L_feats, self.evalpara, SINGLEENC=SINGLEENC)
            att_trials, unatt_trials = select_att_unatt_feats(self.feats_train_trials, repred_labels)
            feats_train = np.concatenate([np.concatenate([att, unatt], axis=1) for att, unatt in zip(att_trials, unatt_trials)], axis=0)
            model = train_cca_model([data_train, feats_train], latent_dimensions=self.latent_dimensions, RANDMODEL=False, SEED=self.SEED, SINGLEENC=SINGLEENC)
        pred_labels = np.array([predict_labels_single_enc(views, model, self.evalpara) if SINGLEENC else predict_labels(views, model, self.L_data, self.L_feats, self.evalpara, NEWSEG=True) for views in segs_views])
        pred_labels_iters.append(pred_labels)
        pred_labels_iters = np.stack(pred_labels_iters, axis=0)
        return pred_labels_iters
    
    def soft(self, gmm_0, gmm_1, model_init=None):
        data_train = np.concatenate(self.data_train_trials, axis=0)
        feats_train = np.concatenate(self.feats_train_trials, axis=0)
        segs_views = [[data, feats] for data, feats in zip(self.data_test_trials, self.feats_test_trials)]
        if model_init is not None:
            model = model_init
            Rinit = Rinit
            Dinit = Dinit
        else:
            model = train_cca_model([data_train, feats_train], latent_dimensions=self.latent_dimensions, RANDMODEL=True, SEED=self.SEED, SINGLEENC=True)
            Rinit = None 
            Dinit = None
        pred_labels_iters = []
        for i in range(self.ITERS):
            pred_labels = np.array([predict_labels_soft(views, model, gmm_0, gmm_1, self.evalpara)[1] for views in segs_views])
            pred_labels_iters.append(pred_labels)
            # reprediction & retraining
            segs_views_train = [[data, feats] for data, feats in zip(self.data_train_trials, self.feats_train_trials)]
            repred_probas = np.array([predict_labels_soft(views, model, gmm_0, gmm_1, self.evalpara)[0] for views in segs_views_train])
            segs_pred = [update_seg_soft(view, probas) for view, probas in zip(segs_views_train, repred_probas)]
            feats_train = np.concatenate([views[1] for views in segs_pred], axis=0)
            model = train_cca_model([data_train, feats_train], latent_dimensions=self.latent_dimensions, RANDMODEL=False, SEED=self.SEED, SINGLEENC=True)
        pred_labels = np.array([predict_labels_soft(views, model, gmm_0, gmm_1, self.evalpara)[1] for views in segs_views])
        pred_labels_iters.append(pred_labels)
        pred_labels_iters = np.stack(pred_labels_iters, axis=0)
        return pred_labels_iters