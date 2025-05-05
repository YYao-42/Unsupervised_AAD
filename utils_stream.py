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


def change_order_of_trials(data_trials, feats_trials, labels_trials):
    # divide trials into two groups based on labels
    data_1 = [data for data, label in zip(data_trials, labels_trials) if label == 1]
    data_2 = [data for data, label in zip(data_trials, labels_trials) if label == 2]
    assert len(data_1) == len(data_2), "The number of trials in data_1 and data_2 should be equal."
    feats_1 = [feats for feats, label in zip(feats_trials, labels_trials) if label == 1]
    feats_2 = [feats for feats, label in zip(feats_trials, labels_trials) if label == 2]
    # make sure that one trials in data_1 is followed by one trial in data_2
    data_trials_ordered = []
    feats_trials_ordered = []
    labels_trials_ordered = []
    for i in range(len(data_1)):
        data_trials_ordered.append(data_1[i])
        feats_trials_ordered.append(feats_1[i])
        labels_trials_ordered.append(1)
        data_trials_ordered.append(data_2[i])
        feats_trials_ordered.append(feats_2[i])
        labels_trials_ordered.append(2)
    return data_trials_ordered, feats_trials_ordered, labels_trials_ordered


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


def train_cca_model(views_train, latent_dimensions=5, RANDMODEL=False, SEED=None):
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


def predict_labels_soft(views, model, gmm_0, gmm_1, evalpara):
    data, feats = views
    f1, f2 = get_feats_per_stream(feats)
    corr_1 = model.average_pairwise_correlations([data, f1])
    corr_sum_1 = cal_corr_sum(corr_1, evalpara[0], evalpara[1])
    corr_2 = model.average_pairwise_correlations([data, f2])
    corr_sum_2 = cal_corr_sum(corr_2, evalpara[0], evalpara[1])
    probas = utils_prob.predict_proba(np.array([corr_sum_1, corr_sum_2]), gmm_0, gmm_1)
    pred_label = 1 if probas[0, 1] > probas[0, 0] else 2
    return probas, pred_label


def calc_smooth_acc(pred_labels, true_labels, nb_trials_considered, nearby=14, nb_calibsessions=2):
    if np.ndim(pred_labels) == 2:
        pred_labels = pred_labels.reshape(-1)
        true_labels = true_labels.reshape(-1)
    if len(pred_labels) != len(true_labels):
        assert len(true_labels) == len(pred_labels) + nb_calibsessions*nb_trials_considered, "The length of pred_labels and true_labels do not match."
        right = pred_labels == true_labels[nb_calibsessions*nb_trials_considered:]
        acc_non_calib = np.sum(right)/len(right)
    else:
        right = pred_labels == true_labels
        acc_non_calib = np.sum(right[nb_calibsessions*nb_trials_considered:])/len(right[nb_calibsessions*nb_trials_considered:])
    print("Avg acc non-calib: ", acc_non_calib)
    acc = []
    for i in range(len(pred_labels)):
        idx_range = (max(0, i-nearby), min(len(pred_labels), i+nearby))
        acc.append(np.sum(right[idx_range[0]:idx_range[1]])/len(right[idx_range[0]:idx_range[1]]))
    return acc_non_calib, acc


class STREAM:
    def __init__(self, data_conditions_dict, feats_conditions_dict, L_data, L_feats, latent_dimensions, SEED, evalpara, nb_trials, UPDATE_STEP):
        self.data_conditions_dict = data_conditions_dict
        self.feats_conditions_dict = feats_conditions_dict
        self.L_data = L_data
        self.L_feats = L_feats
        self.latent_dimensions = latent_dimensions
        self.SEED = SEED
        self.evalpara = evalpara
        self.nb_trials = nb_trials
        self.UPDATE_STEP = UPDATE_STEP

    def fixed_supervised(self, labels_conditions_dict, conds_train, conds_test):
        # Train (single-enc) model with calibration sessions and predict labels for test sessions
        calib_data = []
        calib_feats = []
        calib_labels = []
        for cond in conds_train:
            calib_data += self.data_conditions_dict[cond]
            calib_feats += self.feats_conditions_dict[cond]
            calib_labels += labels_conditions_dict[cond]
        calib_att, _ = select_att_unatt_feats(calib_feats, calib_labels)
        train_data = np.concatenate(calib_data, axis=0)
        train_feats = np.concatenate(calib_att, axis=0)
        model = train_cca_model([train_data, train_feats], latent_dimensions=self.latent_dimensions)
        pred_labels_dict = {}
        for cond in conds_test:
            pred_labels = []
            segs_views = [[data, feats] for data, feats in zip(self.data_conditions_dict[cond], self.feats_conditions_dict[cond])]
            for i in range(self.nb_trials*self.UPDATE_STEP):
                pred_labels.append(predict_labels_single_enc(segs_views[i], model, self.evalpara))
            pred_labels_dict[cond] = pred_labels
        return pred_labels_dict

    def adaptive_supervised(self, labels_conditions_dict, weightpara, PARATRANS=True, SINGLEENC=True, conds_sorted=None):
        # Adaptive training with known labels
        model_init = None
        pred_labels_dict = {}
        conds = labels_conditions_dict.keys() if conds_sorted is None else conds_sorted
        for cond in conds:
            pred_labels = []
            segs_views = [[data, feats] for data, feats in zip(self.data_conditions_dict[cond], self.feats_conditions_dict[cond])]
            if model_init is not None:
                model = model_init
                Rinit = Rinit
                Dinit = Dinit
            else:
                model = train_cca_model_adaptive(segs_views[0], None, None, latent_dimensions=self.latent_dimensions, weightpara=weightpara, RANDMODEL=True, SEED=self.SEED, SINGLEENC=SINGLEENC)
                Rinit = None 
                Dinit = None
            for k in range(self.UPDATE_STEP):
                pred_labels.append(predict_labels_single_enc(segs_views[k], model, self.evalpara) if SINGLEENC else predict_labels(segs_views[k], model, self.L_data, self.L_feats, self.evalpara, NEWSEG=True))
            for i in range(0, self.nb_trials*self.UPDATE_STEP, self.UPDATE_STEP):
                data_segs = []
                feats_segs = []
                label_segs = []
                for k in range(self.UPDATE_STEP):
                    data_segs.append(segs_views[i+k][0])
                    feats_segs.append(segs_views[i+k][1])
                    label_segs.append(labels_conditions_dict[cond][i+k])
                att_segs, unatt_segs = select_att_unatt_feats(feats_segs, label_segs)
                att_unatt_segs = [np.concatenate((att, unatt), axis=1) for att, unatt in zip(att_segs, unatt_segs)]
                seg_train = [np.concatenate(data_segs, axis=0), np.concatenate(att_unatt_segs, axis=0)]
                model = train_cca_model_adaptive(seg_train, Rinit, Dinit, latent_dimensions=self.latent_dimensions, weightpara=weightpara, RANDMODEL=False, SEED=self.SEED, SINGLEENC=SINGLEENC)
                Rinit = model.Rxx 
                Dinit = model.Dxx
                if i < (self.nb_trials - 1) * self.UPDATE_STEP:
                    for k in range(self.UPDATE_STEP):
                        # predict labels of the next trials
                        pred_labels.append(predict_labels_single_enc(segs_views[i+self.UPDATE_STEP+k], model, self.evalpara) if SINGLEENC else predict_labels(segs_views[i+self.UPDATE_STEP+k], model, self.L_data, self.L_feats, self.evalpara, NEWSEG=True))
            pred_labels_dict[cond] = pred_labels
            if PARATRANS:
                model_init = model
        return pred_labels_dict

    def recursive(self, weightpara, PARATRANS=True, SINGLEENC=True, conds_sorted=None):
        model_init = None
        pred_labels_dict = {}
        conds = self.data_conditions_dict.keys() if conds_sorted is None else conds_sorted
        for cond in conds:
            pred_labels = []
            segs_views = [[data, feats] for data, feats in zip(self.data_conditions_dict[cond], self.feats_conditions_dict[cond])]
            if model_init is not None:
                model = model_init
                Rinit = Rinit
                Dinit = Dinit
            else:
                model = train_cca_model_adaptive(segs_views[0], None, None, latent_dimensions=self.latent_dimensions, weightpara=weightpara, RANDMODEL=True, SEED=self.SEED, SINGLEENC=SINGLEENC)
                Rinit = None 
                Dinit = None
            for k in range(self.UPDATE_STEP):
                pred_labels.append(predict_labels_single_enc(segs_views[k], model, self.evalpara) if SINGLEENC else predict_labels(segs_views[k], model, self.L_data, self.L_feats, self.evalpara, NEWSEG=True))
            for i in range(0, self.nb_trials*self.UPDATE_STEP, self.UPDATE_STEP):
                data_segs = []
                feats_segs = []
                for k in range(self.UPDATE_STEP):
                    data_segs.append(segs_views[i+k][0])
                    feats_segs.append(segs_views[i+k][1])
                seg_to_pred = [np.concatenate(data_segs, axis=0), np.concatenate(feats_segs, axis=0)]
                label = predict_labels_single_enc(seg_to_pred, model, self.evalpara) if SINGLEENC else predict_labels(seg_to_pred, model, self.L_data, self.L_feats, self.evalpara, NEWSEG=False) 
                if label == 1:
                    seg_predicted = seg_to_pred
                else:
                    seg_predicted = [seg_to_pred[0], np.concatenate([seg_to_pred[1][:, self.L_feats:], seg_to_pred[1][:, :self.L_feats]], axis=1)]
                model = train_cca_model_adaptive(seg_predicted, Rinit, Dinit, latent_dimensions=self.latent_dimensions, weightpara=weightpara, RANDMODEL=False, SEED=self.SEED, SINGLEENC=SINGLEENC)
                Rinit = model.Rxx 
                Dinit = model.Dxx
                if i < (self.nb_trials - 1) * self.UPDATE_STEP:
                    for k in range(self.UPDATE_STEP):
                        # predict labels of the next trials
                        pred_labels.append(predict_labels_single_enc(segs_views[i+self.UPDATE_STEP+k], model, self.evalpara) if SINGLEENC else predict_labels(segs_views[i+self.UPDATE_STEP+k], model, self.L_data, self.L_feats, self.evalpara, NEWSEG=True))
            pred_labels_dict[cond] = pred_labels
            if PARATRANS:
                model_init = model
        return pred_labels_dict

    def recursive_soft(self, weightpara, gmm_0, gmm_1, PARATRANS=True, conds_sorted=None):
        model_init = None
        pred_labels_dict = {}
        conds = self.data_conditions_dict.keys() if conds_sorted is None else conds_sorted
        for cond in conds:
            pred_labels = []
            segs_views = [[data, feats] for data, feats in zip(self.data_conditions_dict[cond], self.feats_conditions_dict[cond])]
            if model_init is not None:
                model = model_init
                Rinit = Rinit
                Dinit = Dinit
            else:
                model = train_cca_model_adaptive(segs_views[0], None, None, latent_dimensions=self.latent_dimensions, weightpara=weightpara, RANDMODEL=True, SEED=self.SEED, SINGLEENC=True)
                Rinit = None 
                Dinit = None
            for k in range(self.UPDATE_STEP):
                pred_labels.append(predict_labels_soft(segs_views[k], model, gmm_0, gmm_1, self.evalpara)[1])
            for i in range(0, self.nb_trials*self.UPDATE_STEP, self.UPDATE_STEP):
                data_segs = []
                feats_segs = []
                for k in range(self.UPDATE_STEP):
                    data_segs.append(segs_views[i+k][0])
                    feats_segs.append(segs_views[i+k][1])
                seg_to_pred = [np.concatenate(data_segs, axis=0), np.concatenate(feats_segs, axis=0)]
                probas, _ = predict_labels_soft(seg_to_pred, model, gmm_0, gmm_1, self.evalpara)
                eeg_seg, feats_seg = seg_to_pred
                f1, f2 = get_feats_per_stream(feats_seg)
                att_predicted = f1*probas[0, 1] + f2*probas[0, 0]
                unatt_predicted = f2*probas[0, 1] + f1*probas[0, 0]
                seg_predicted = [eeg_seg, np.concatenate([att_predicted, unatt_predicted], axis=1)]
                model = train_cca_model_adaptive(seg_predicted, Rinit, Dinit, latent_dimensions=self.latent_dimensions, weightpara=weightpara, RANDMODEL=False, SEED=self.SEED, SINGLEENC=True)
                Rinit = model.Rxx 
                Dinit = model.Dxx
                if i < (self.nb_trials - 1) * self.UPDATE_STEP:
                    for k in range(self.UPDATE_STEP):
                        # predict labels of the next trials
                        pred_labels.append(predict_labels_soft(segs_views[i+self.UPDATE_STEP+k], model, gmm_0, gmm_1, self.evalpara)[1])
            pred_labels_dict[cond] = pred_labels
            if PARATRANS:
                model_init = model
        return pred_labels_dict