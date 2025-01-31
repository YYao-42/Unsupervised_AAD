import numpy as np
import copy
import random
import itertools
from sklearn.covariance import LedoitWolf
from tqdm import tqdm
from numpy import linalg as LA
from scipy.linalg import eig, eigh, sqrtm, lstsq
from scipy.stats import pearsonr
from cca_zoo.linear import rCCA, PCACCA, PartialCCA, GCCA, MCCA
from cca_zoo.nonparametric import KGCCA, NCCA
from cca_zoo.probabilistic import ProbabilisticCCA
from algo_suppl import CorrCA_LW, MCCA_LW
from sklearn.kernel_approximation import RBFSampler, Nystroem
import utils


class CorrelationAnalysis:
    def __init__(self, views_list, method, fs, params_hankel, CONTAIN_PARTIALS=False, views_aug_list=None, n_components=3, leave_out=2, VALSET=True, CROSSVAL=True, SELECTVIEWS=None, message=True, n_permu=500, p_value=0.05, random_state=42, **kwargs):
        '''
        views: list of data views, [[v1_video1, v2_video2, ...], [v2_video1, v2_video2, ...], ...] each element is a T(#sample)xDx(#channel) array 
        fs: Sampling rate
        params_hankel: Parameters for Hankelization [(L, offset), ...]
        leave_out: Number of pairs to leave out for leave-one-pair-out cross-validation
        n_components: Number of components to be returned
        message: If print message
        n_permu: Number of permutations for the significance level
        p_value: Significance level
        kwargs: Other parameters for the method
        '''
        self.views_list = views_list
        self.nb_views = len(self.views_list)
        self.nb_videos = len(self.views_list[0])
        self.method = method
        self.fs = fs
        self.params_hankel = params_hankel
        self.CONTAIN_PARTIALS = CONTAIN_PARTIALS
        self.views_aug_list = views_aug_list
        if self.views_aug_list is not None:
            assert len(self.views_aug_list) == len(self.views_list), "The number of augmented views should be the same as the original views."        
        self.n_components = n_components
        self.leave_out = leave_out
        self.VALSET = VALSET
        self.CROSSVAL = CROSSVAL
        if self.CROSSVAL and self.views_aug_list is None:
            print("Performing cross-validation.")
            assert self.nb_videos%self.leave_out == 0, "The number of videos should be a multiple of the leave_out parameter."
            self.nb_folds = self.nb_videos//self.leave_out
        elif self.views_aug_list is not None:
            print("Not performing cross-validation because addtional data is provided for training.")
            self.nb_folds = 1
        else:
            print("Not performing cross-validation because CROSSVAL is set to False.")
            self.nb_folds = 1
        self.SELECTVIEWS = SELECTVIEWS
        self.message = message
        self.n_permu = n_permu
        self.p_value = p_value
        self.random_state = random_state
        self._assign_params(kwargs)

        self.best_models = None

    def _assign_params(self, kwargs):
        '''
        Assign parameters for the method
        '''
        kwargs = kwargs or {}
        self.data_process_params = {key: value for key, value in kwargs.items() if key in {'rbf', 'rbf_n_comp', 'rbf_gamma'}}
        self.fit_params = {key: value for key, value in kwargs.items() if key in {'c', 'eps', 'pca', 'learning_rate'}}

    def _process_data_per_view(self, view, view_idx, L, offset, NORMALIZE=True):
        view_hankelized = utils.block_Hankel(view, L, offset)
        if self.data_process_params:
            if self.data_process_params['rbf']:
                rbf_n_comp_view = self.data_process_params['rbf_n_comp'][view_idx] if 'rbf_n_comp' in self.data_process_params else 100
                rbf_gamma_view = self.data_process_params['rbf_gamma'][view_idx] if 'rbf_gamma' in self.data_process_params else 1
                rbf_samp = RBFSampler(gamma=rbf_gamma_view, n_components=rbf_n_comp_view, random_state=self.random_state)
                view_hankelized = rbf_samp.fit_transform(view_hankelized)
        if NORMALIZE:
            view_hankelized = utils.normalize_per_view(view_hankelized)
        return view_hankelized
            
    def _process_data(self, views, NORMALIZE=True):
        '''
        Process the data for CCA
        '''
        views_hankelized = []
        for i, view in enumerate(views):
            L, offset = self.params_hankel[i]
            view_hankelized = self._process_data_per_view(view, i, L, offset, NORMALIZE)
            views_hankelized.append(view_hankelized)
        if self.CONTAIN_PARTIALS:
            self.partials_hankelized = views_hankelized[-1]
            self.views_hankelized = views_hankelized[:-1]
        else:
            self.partials_hankelized = None
            self.views_hankelized = views_hankelized

    def _fit(self):
        '''
        Fit the data with the specified method
        '''

        if self.method == 'rCCA':
            model = rCCA(latent_dimensions=self.n_components, pca=False, eps=0, **self.fit_params)
        elif self.method == 'MCCA':
            model = MCCA(latent_dimensions=self.n_components, pca=False, eps=0, **self.fit_params)
        elif self.method == 'PCACCA':
            model = PCACCA(latent_dimensions=self.n_components, **self.fit_params)
        elif self.method == 'PartialCCA':
            model = PartialCCA(latent_dimensions=self.n_components, pca=False, **self.fit_params)
        elif self.method == 'NCCA':
            model = NCCA(latent_dimensions=self.n_components, **self.fit_params)
        elif self.method == 'ProbabilisticCCA':
            model = ProbabilisticCCA(latent_dimensions=self.n_components, **self.fit_params)
        elif self.method == 'MCCA_LW':
            model = MCCA_LW(latent_dimensions=self.n_components)
        else:
            raise ValueError('Method not supported.')
        model.fit(self.views_hankelized) if self.method != 'PartialCCA' else model.fit(self.views_hankelized, partials=self.partials_hankelized)
        return model

    def _get_transformed_data(self, model):
        '''
        Get the transformed data
        '''
        if self.method == 'PartialCCA':
            views_trans = model.transform(self.views_hankelized, partials=self.partials_hankelized)
        else:
            views_trans = model.transform(self.views_hankelized)
        return views_trans

    def _get_influence(self, model, views, TRAIN_MODE, resolution=10, TRACKMODE='ST', aggcomp=None, overlap=0, CROSSVIEW=False):
        lag_views = [param[0] for param in self.params_hankel]
        self._process_data(views)
        # divide the views into segments
        views_in_segs = [utils.into_trials_with_overlap(view, self.fs, resolution, overlap=overlap) for view in self.views_hankelized]
        nb_views = len(views_in_segs)
        nb_segs = len(views_in_segs[0])
        # get views in each segment
        segs_views = [[views_in_segs[i][j] for i in range(nb_views)] for j in range(nb_segs)]
        # get influence of views for each segment
        segs_influence_views = [utils.get_influence_all_views(views, model.weights_, lag_views, TRAIN_MODE, TRACKMODE, aggcomp, CROSSVIEW) for views in segs_views]
        # stack the influence of views along axis 2; shape of elements in influence_views: (dim_view, nb_components, nb_segs)
        influence_views = [np.stack([segs_influence_views[j][i] for j in range(nb_segs)], axis=2) for i in range(nb_views)]
        return influence_views
    
    def _get_influence_data_two_layers(self, model, data_h, feats, w1_data, lag1_data, TRAIN_MODE, resolution=10, TRACKMODE='ST', aggcomp=None, overlap=0, CROSSVIEW=False):
        w2_data, w2_feats = model.weights_
        lag2_views = [param[0] for param in self.params_hankel]
        assert lag2_views[0] == 1, "In the second layer, the data side should not have time-lagged copies."
        lag2_feats = lag2_views[1]
        data_h = np.expand_dims(data_h, axis=2) if len(data_h.shape) == 2 else data_h
        w1_data = np.expand_dims(w1_data, axis=2) if len(w1_data.shape) == 2 else w1_data
        nb_subj = data_h.shape[2] 
        views_persubj_feats = [data_h[:,:,n] for n in range(nb_subj)] + [feats]
        self.params_hankel = [self.params_hankel[0]] * nb_subj + [self.params_hankel[1]]
        self._process_data(views_persubj_feats)
        self.params_hankel = [self.params_hankel[0], self.params_hankel[-1]]
        views_h = [np.stack(self.views_hankelized[:-1], axis=2), self.views_hankelized[-1]]
        views_in_segs = [utils.into_trials_with_overlap(view, self.fs, resolution, overlap=overlap) for view in views_h]
        nb_views = len(views_in_segs)
        nb_segs = len(views_in_segs[0])
        segs_views = [[views_in_segs[i][j] for i in range(nb_views)] for j in range(nb_segs)]
        segs_influence_views = [utils.get_influence_all_views_two_layers(views, w1_data, w2_data, w2_feats, lag1_data, lag2_feats, TRAIN_MODE, TRACKMODE, aggcomp, CROSSVIEW) for views in segs_views]
        influence_views = [np.stack([segs_influence_views[j][i] for j in range(nb_segs)], axis=2) for i in range(nb_views)]
        return influence_views
            
    def _get_inter_view_corr(self, model):
        '''
        Get the inter-view correlation coefficients
        '''
        assert self.SELECTVIEWS is None, "SELECTVIEWS is not supported for cca-zoo-based inter-view correlation calculation."
        if self.method == 'PartialCCA':
            ivc = model.average_pairwise_correlations(self.views_hankelized, partials=self.partials_hankelized)
        else:
            ivc = model.average_pairwise_correlations(self.views_hankelized)
        return ivc

    def _get_inter_view_corr_mv(self, views_trans):
        '''
        Get the inter-view correlation coefficients from the transformed data; Is able to select views taken into account; The views_trans can be sth other than self.views_hankelized
        '''
        if self.SELECTVIEWS is not None:
            views_trans = [views_trans[i] for i in self.SELECTVIEWS]
        views_tensor = np.stack(views_trans, axis=2)
        nb_views = len(views_trans)
        n_components = self.n_components
        corr_tensor = np.zeros((nb_views, nb_views, n_components))
        ivc = np.zeros(n_components)
        for component in range(n_components):
            corr_tensor[:,:,component] = np.corrcoef(views_tensor[:,component,:], rowvar=False)
            ivc[component] = np.sum(corr_tensor[:,:,component]-np.eye(nb_views))/nb_views/(nb_views-1)
        return ivc
        
    def _permutation_test(self, model, PHASE_SCRAMBLE=True, block_len=None):
        '''
        Permutation test for the correlation coefficients. Use phase scrambling or block shuffling.
        '''
        views_trans = self._get_transformed_data(model)
        ivc_permu = np.zeros((self.n_permu, self.n_components))
        for i in tqdm(range(self.n_permu)):
            views_shuffled = [utils.phase_scramble_2D(view) if PHASE_SCRAMBLE else utils.shuffle_2D(view, block_len) for view in views_trans]
            ivc_shuffled = self._get_inter_view_corr_mv(views_shuffled)
            ivc_permu[i,:] = ivc_shuffled
        return ivc_permu
    
    def _calcu_sig_ivc(self, ivc_permu_fold):
        nb_folds = len(ivc_permu_fold)
        ivc_permu = np.concatenate(tuple(ivc_permu_fold), axis=0)
        ivc_permu = ivc_permu.reshape(-1)
        ivc_permu = np.sort(abs(ivc_permu))
        assert self.n_components*self.n_permu*nb_folds == ivc_permu.shape[0]
        sig_idx = -int(self.n_permu*self.p_value*self.n_components*nb_folds)
        return ivc_permu[sig_idx]

    def _split_data(self, extra_views_list=None):
        '''
        Split the data into training, testing, and validation sets
        '''
        views = self.views_list if extra_views_list is None else self.views_list + extra_views_list
        if self.views_aug_list is None:
            train_list_folds, test_list_folds, val_list_folds = utils.split_multi_mod_withval_LVO(views, self.leave_out, self.VALSET, self.CROSSVAL)
        else:
            train_list, test_list, val_list = utils.split_multi_mod_withval_withaug(views, self.views_aug_list, self.VALSET)
            train_list_folds, test_list_folds, val_list_folds = [train_list], [test_list], [val_list]
        assert len(train_list_folds) == len(test_list_folds) == self.nb_folds
        assert len(val_list_folds) == self.nb_folds if self.VALSET else val_list_folds is None
        return train_list_folds, test_list_folds, val_list_folds

    def retrain_dec_or_enc(self, MODE='DECODER'):
        assert self.best_models is not None, "The best models should be provided for retraining."
        train_list_folds, _, _ = self._split_data()
        for idx in range(self.nb_folds):
            best_model = self.best_models[idx]
            dec, enc = best_model.weights_
            views_train = train_list_folds[idx]
            self._process_data(views_train)
            data_train, feats_train = self.views_hankelized
            views_trans_train = self._get_transformed_data(best_model)
            data_trans_train, feat_trans_train = views_trans_train
            if MODE == 'DECODER':
                R_data = LedoitWolf().fit(data_train).covariance_
                R_df = data_train.T @ feat_trans_train / data_train.shape[0]
                R_fd = R_df.T
                dec_new = LA.inv(R_data) @ R_df @ sqrtm(R_fd @ LA.inv(R_data) @ R_df)
                assert dec_new.shape == dec.shape, "The shape of the new decoder should be the same as the original decoder."
                self.best_models[idx].weights_[0] = dec_new
            if MODE == 'ENCODER':
                R_feats = LedoitWolf().fit(feats_train).covariance_
                R_df = data_trans_train.T @ feats_train / data_trans_train.shape[0]
                R_fd = R_df.T
                enc_new = LA.inv(R_feats) @ R_fd @ sqrtm(R_df @ LA.inv(R_feats) @ R_fd)
                assert enc_new.shape == enc.shape, "The shape of the new encoder should be the same as the original encoder."
                self.best_models[idx].weights_[1] = enc_new

    def inter_view_corr(self, train_list_folds=None, test_list_folds=None, SIGNIFI_LEVEL=True):
        '''
        Calculate inter-view correlation 
        '''
        train_list_folds, test_list_folds, _ = (self._split_data() if train_list_folds is None else (train_list_folds, test_list_folds, None))
        ivc_train_fold = np.zeros((self.nb_folds, self.n_components))
        ivc_test_fold = np.zeros((self.nb_folds, self.n_components))
        model_fold = []
        ivc_permu_fold = []
        for idx in range(0, self.nb_folds):
            views_train, views_test = train_list_folds[idx], test_list_folds[idx]
            self._process_data(views_train)
            model = self._fit() if self.best_models is None else self.best_models[idx]
            views_trans_train = self._get_transformed_data(model)
            ivc_train_fold[idx,:] = self._get_inter_view_corr_mv(views_trans_train)
            self._process_data(views_test)
            views_trans_test = self._get_transformed_data(model)
            ivc_test_fold[idx,:] = self._get_inter_view_corr_mv(views_trans_test)
            model_fold.append(model)
            if SIGNIFI_LEVEL:
                ivc_permu_fold.append(self._permutation_test(model))
        if SIGNIFI_LEVEL:
            sig_ivc = self._calcu_sig_ivc(ivc_permu_fold)
        else:
            sig_ivc = None
        if self.message:
            print('Average inter-view correlation coefficients of the top {} components on the training sets: {}'.format(self.n_components, np.average(ivc_train_fold, axis=0)))
            print('Average inter-view correlation coefficients of the top {} components on the test sets: {}'.format(self.n_components, np.average(ivc_test_fold, axis=0)))
            print('Significance level: {}'.format(sig_ivc))
        return ivc_train_fold, ivc_test_fold, sig_ivc, model_fold

    def vad_mm(self, trial_len, train_list_folds=None, test_list_folds=None, feat_unatt_list=None, MM=True):
        '''
        Perform visual attention decoding and/or match-mismatch tasks
        '''
        train_list_folds, test_list_folds, _ = self._split_data(extra_views_list=[feat_unatt_list]) if train_list_folds is None else (train_list_folds, test_list_folds, None)
        corr_att_fold = []
        corr_unatt_fold = []
        corr_mm_fold = []
        model_fold = []
        for idx in range(0, self.nb_folds):
            views_train, views_test = train_list_folds[idx], test_list_folds[idx]
            data_train, feat_att_train, _ = views_train
            data_test, feat_att_test, feat_unatt_test = views_test
            self._process_data([data_train, feat_att_train])
            model = self._fit() if self.best_models is None else self.best_models[idx]
            model_fold.append(model)
            self._process_data([data_test, feat_att_test])
            data_trans, feat_att_trans = self._get_transformed_data(model)
            data_trans_trials = utils.into_trials_with_overlap(data_trans, self.fs, trial_len, overlap=0.9)
            att_trans_trials = utils.into_trials_with_overlap(feat_att_trans, self.fs, trial_len, overlap=0.9)
            corr_att = [self._get_inter_view_corr_mv([data, att]) for data, att in zip (data_trans_trials, att_trans_trials)]
            if feat_unatt_list is not None:
                self._process_data([data_test, feat_unatt_test])
                _, feat_unatt_trans = self._get_transformed_data(model)
                Unatt_trans_trials = utils.into_trials_with_overlap(feat_unatt_trans, self.fs, trial_len, overlap=0.9)
                corr_unatt = [self._get_inter_view_corr_mv([data, unatt]) for data, unatt in zip (data_trans_trials, Unatt_trans_trials)]
                corr_unatt_fold.append(np.array(corr_unatt))
            if MM:
                mm_trans_trials = utils.into_trials_with_overlap(feat_att_trans, self.fs, trial_len, overlap=0.9, PERMUTE=True)
                corr_mm = [self._get_inter_view_corr_mv([data, mm]) for data, mm in zip (data_trans_trials, mm_trans_trials)]
                corr_mm_fold.append(np.array(corr_mm))
            corr_att_fold.append(np.array(corr_att))
            corr_att, corr_unatt, corr_mm = [np.concatenate(tuple(corr_fold), axis=0) if len(corr_fold)!=0 else None for corr_fold in [corr_att_fold, corr_unatt_fold, corr_mm_fold]]
        return corr_att, corr_unatt, corr_mm, model_fold
    
    def hyperparam_search_ivc(self, param_grid, train_list_folds=None, val_list_folds=None, DATAFEATS=False, SAVE_INFO=None, range=3, nb_comp_into_account=2):
        '''
        Hyperparameter search, based on the inter-view correlation coefficients
        '''
        self.best_models = None
        if train_list_folds is None:
            train_list_folds, _, val_list_folds = self._split_data()
        keys, values = zip(*param_grid.items())
        best_ivc = -np.inf
        best_params = None
        best_models = None
        for combination in itertools.product(*values):
            kwargs = dict(zip(keys, combination))
            self._assign_params(kwargs)
            try:
                ivc_train_folds, ivc_test_folds, _, model_folds = self.inter_view_corr(train_list_folds, val_list_folds, SIGNIFI_LEVEL=False)
            except:
                print('Error in hyperparameter search. Skip combination:', kwargs)
                continue
            ivc_test = np.average(ivc_test_folds, axis=0)
            ivc_train = np.average(ivc_train_folds, axis=0)
            if SAVE_INFO is not None:
                res = {'ivc_train': ivc_train, 'ivc_test': ivc_test}
                if DATAFEATS:
                    utils.save_results_data_feats(SAVE_INFO[0], res, kwargs, SAVE_INFO[1], OVERWRITE=False)
                else:
                    utils.save_results_data_only(SAVE_INFO[0], res, kwargs, SAVE_INFO[1], OVERWRITE=False)
            ivc_ranked = np.sort(ivc_test[:range])[::-1]
            ivc = np.sum(ivc_ranked[:nb_comp_into_account])
            if ivc > best_ivc:
                best_ivc = ivc
                best_params = kwargs
                best_models = model_folds
        self._assign_params(best_params)
        self.best_models = best_models
        return best_params

    def hyperparam_search_acc(self, param_grid, trial_len, train_list_folds=None, val_list_folds=None, feat_unatt_list=None, SAVE_INFO=None):
        '''
        Hyperparameter search, based on the decoding accuracy
        '''
        self.best_models = None
        if train_list_folds is None:
            train_list_folds, _, val_list_folds = self._split_data(extra_views_list=[feat_unatt_list])
        keys, values = zip(*param_grid.items())
        best_acc = -np.inf
        best_params = None
        best_models = None
        for combination in itertools.product(*values):
            kwargs = dict(zip(keys, combination))
            self._assign_params(kwargs)
            try:
                corr_att_trials, _, corr_mm_trials, model_folds = self.vad_mm(trial_len, train_list_folds=train_list_folds, test_list_folds=val_list_folds, MM=True)
            except:
                print('Error in hyperparameter search. Skip combination:', kwargs)
                continue
            acc, _, _, corr_att, corr_mm  = utils.eval_compete(corr_att_trials, corr_mm_trials, TRAIN_WITH_ATT=True)
            if SAVE_INFO is not None:
                res = {'acc': acc, 'corr_att': corr_att, 'corr_mm': corr_mm}
                utils.save_results_data_feats(SAVE_INFO[0], res, kwargs, SAVE_INFO[1], OVERWRITE=False)
            if acc > best_acc:
                best_acc = acc
                best_params = kwargs
                best_models = model_folds
        self._assign_params(best_params)
        self.best_models = best_models
        return best_params
    
    def track_influence(self, TRAINMODE, resolution=10, TRACKMODE='ST', train_list_folds=None, val_list_folds=None, aggcomp=None, CROSSVIEW=False):
        '''
        Track the influence of views
        '''
        if TRACKMODE == 'NO':
            return None, None, None
        else:
            assert self.best_models is not None, "The best models should be provided for influence tracking."
            SPLIT_HOMO = train_list_folds is None
            if SPLIT_HOMO:
                train_list_folds, test_list_folds, val_list_folds = self._split_data()
            infl_train_folds = []
            infl_test_folds = []
            infl_val_folds = []
            for idx in range(self.nb_folds):
                model = self.best_models[idx]
                infl_train_folds.append(self._get_influence(model, train_list_folds[idx], TRAINMODE, resolution, TRACKMODE, aggcomp, CROSSVIEW=CROSSVIEW))
                if self.VALSET:
                    infl_val_folds.append(self._get_influence(model, val_list_folds[idx], TRAINMODE, resolution, TRACKMODE, aggcomp, CROSSVIEW=CROSSVIEW))
                if SPLIT_HOMO:
                    infl_test_folds.append(self._get_influence(model, test_list_folds[idx], TRAINMODE, resolution, TRACKMODE, aggcomp, CROSSVIEW=CROSSVIEW))
        return infl_train_folds, infl_test_folds, infl_val_folds
    
    def track_influence_data_two_layers(self, data_ori_h, w1_data, lag1_data, TRAINMODE, resolution=10, TRACKMODE='ST', train_list_folds=None, val_list_folds=None, aggcomp=None, CROSSVIEW=False):
        '''
        Track the influence of views
        '''
        if TRACKMODE == 'NO':
            return None, None, None
        else:
            assert self.best_models is not None, "The best models should be provided for influence tracking."
            SPLIT_HOMO = train_list_folds is None
            if SPLIT_HOMO:
                train_list_folds, test_list_folds, val_list_folds = self._split_data(extra_views_list=[data_ori_h])
            infl_train_folds = []
            infl_test_folds = []
            infl_val_folds = []
            for idx in range(self.nb_folds):
                model = self.best_models[idx]
                _, feats_train, data_h_train = train_list_folds[idx]
                infl_train_folds.append(self._get_influence_data_two_layers(model, data_h_train, feats_train, w1_data, lag1_data, TRAINMODE, resolution, TRACKMODE, aggcomp, CROSSVIEW=CROSSVIEW))
                if self.VALSET:
                    _, feats_val, data_h_val = val_list_folds[idx]
                    infl_val_folds.append(self._get_influence_data_two_layers(model, data_h_val, feats_val, w1_data, lag1_data, TRAINMODE, resolution, TRACKMODE, aggcomp, CROSSVIEW=CROSSVIEW))
                if SPLIT_HOMO:
                    _, feats_test, data_h_test = test_list_folds[idx]
                    infl_test_folds.append(self._get_influence_data_two_layers(model, data_h_test, feats_test, w1_data, lag1_data, TRAINMODE, resolution, TRACKMODE, aggcomp, CROSSVIEW=CROSSVIEW))
        return infl_train_folds, infl_test_folds, infl_val_folds


class MultiViewCorrelationAnalysis(CorrelationAnalysis):
    def __init__(self, views_list, method, fs, params_hankel, CONTAIN_PARTIALS=False, views_aug_list=None, n_components=3, leave_out=2, VALSET=True, CROSSVAL=True, SELECTVIEWS=None, message=True, n_permu=500, p_value=0.05, random_state=42, **kwargs):
        '''
        views: list of data views, [[v1_video1, v2_video2, ...], [v2_video1, v2_video2, ...], ...] each element is a T(#sample)xDx(#channel) array 
        fs: Sampling rate
        params_hankel: Parameters for Hankelization [(L, offset), ...]
        leave_out: Number of pairs to leave out for leave-one-pair-out cross-validation
        n_components: Number of components to be returned
        message: If print message
        n_permu: Number of permutations for the significance level
        p_value: Significance level
        kwargs: Other parameters for the method
        '''
        self.views_list = views_list
        self.nb_videos = len(self.views_list[0])
        self.method = method
        self.fs = fs
        self.params_hankel = params_hankel
        self.CONTAIN_PARTIALS = CONTAIN_PARTIALS
        self.views_aug_list = views_aug_list
        if self.views_aug_list is not None:
            assert len(self.views_aug_list) == len(self.views_list), "The number of augmented views should be the same as the original views."        
        self.n_components = n_components
        self.leave_out = leave_out
        self.VALSET = VALSET
        self.CROSSVAL = CROSSVAL
        if self.CROSSVAL and self.views_aug_list is None:
            print("Performing cross-validation.")
            assert self.nb_videos%self.leave_out == 0, "The number of videos should be a multiple of the leave_out parameter."
            self.nb_folds = self.nb_videos//self.leave_out
        elif self.views_aug_list is not None:
            print("Not performing cross-validation because addtional data is provided for training.")
            self.nb_folds = 1
        else:
            print("Not performing cross-validation because CROSSVAL is set to False.")
            self.nb_folds = 1
        self.SELECTVIEWS = SELECTVIEWS
        self.message = message
        self.n_permu = n_permu
        self.p_value = p_value
        self.random_state = random_state
        self._assign_params(kwargs)

        self.best_models = None

    def _fit(self):
        '''
        Fit the data with the specified method
        '''

        if self.method == 'GCCA':
            model = GCCA(latent_dimensions=self.n_components, **self.fit_params)
        elif self.method == 'MCCA':
            model = MCCA(latent_dimensions=self.n_components, pca=False, eps=0, **self.fit_params)
        elif self.method == 'KGCCA':
            model = KGCCA(latent_dimensions=self.n_components, **self.fit_params)
        elif self.method == 'CorrCA':
            model = CorrCA_LW(latent_dimensions=self.n_components)
        elif self.method == 'MCCA_LW':
            model = MCCA_LW(latent_dimensions=self.n_components)
        else:
            raise ValueError('Method not supported.')
        model.fit(self.views_hankelized) if self.method != 'PartialCCA' else model.fit(self.views_hankelized, partials=self.partials_hankelized)
        return model
    
    def get_gcca_preprocessed_views(self, views_complete=None, views_aug_complete=None):
        # This method is only used when views_aug_list exists, i.e., when using the single-object data for training
        assert self.views_aug_list is not None 
        train_list, test_list, val_list = utils.split_multi_mod_withval_withaug(self.views_list, self.views_aug_list, VAL=self.VALSET)
        model = self._fit()
        if views_complete is not None:
            train_list, test_list, val_list = utils.split_multi_mod_withval_withaug(views_complete, views_aug_complete, VAL=self.VALSET)
            model.weights_ = (model.weights_).append(model.weights_[-1])
        self._process_data(train_list)
        views_trans_train = self._get_transformed_data(model)
        self._process_data(test_list)
        views_trans_test = self._get_transformed_data(model)
        if self.VALSET:
            self._process_data(val_list)
            views_trans_val = self._get_transformed_data(model)
        else:
            views_trans_val = None
        views_datasets = [views_trans_train, views_trans_test, views_trans_val]
        return views_datasets, model.weights_

    def get_gcca_preprocessed_views_not_divided(self, views_complete=None):
        # This method is only used when views_aug_list exists, i.e., when using the single-object data for training
        # Compared to get_gcca_preprocessed_views, this method uses single-object data for training, and returns only the transformed superimposed-object data
        # The returned data is in the original form, not divided into testing and validation sets
        assert self.views_aug_list is not None 
        SINDP = True if views_complete is not None else False
        train_list, _, _ = utils.split_multi_mod_withval_withaug(self.views_list, self.views_aug_list, VAL=self.VALSET)
        self._process_data(train_list)
        model = self._fit()
        nb_views = len(self.views_list)
        nb_videos = len(self.views_list[0])
        views_transformed_temp = []
        if SINDP:
            views_to_trans = views_complete
            (model.weights_).append(model.weights_[-1])
        else:
            views_to_trans = self.views_list
        for i in range(nb_videos):
            views = [view[i] for view in views_to_trans]
            self._process_data(views)
            views_transformed_temp.append(self._get_transformed_data(model))
        views_transformed_list = [np.stack(views, axis=2) for views in views_transformed_temp]
        return views_transformed_list, model.weights_