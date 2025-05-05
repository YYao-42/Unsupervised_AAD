import numpy as np
import utils
import utils_stream
import matplotlib.pyplot as plt
from algo_ccazoo import CorrelationAnalysis
from sklearn.mixture import GaussianMixture


def cal_corr_sum_multi_trials(corr, range_into_account=2, nb_comp_into_account=1):
    corr_ranked = np.sort(corr[:,:range_into_account], axis=1)[:,::-1]
    corr_sum = np.sum(corr_ranked[:,:nb_comp_into_account], axis=1)
    return corr_sum


def estimate_distribution_corr(eeg_dict, feats_dict, labels_dict, est_subs, fs, hparadata, hparafeats, leave_out_persubj, trial_len=60, range_into_account=2, nb_comp_into_account=1):
    """
    Estimate the distribution of correlation coefficients for the given subjects.
    
    Parameters:
    - eeg_dict: Dictionary containing EEG data for each subject.
    - feats_dict: Dictionary containing features for each subject.
    - labels_dict: Dictionary containing labels for each subject.
    - subest_subsjects: List of subjects to be used for estimation.
    - fs: Sampling frequency.
    - hparadata: Hyperparameters for data analysis.
    - hparafeats: Hyperparameters for feature analysis.
    
    Returns:
    - corr_att: Correlation coefficients for attended trials.
    - corr_unatt: Correlation coefficients for unattended trials.
    """
    # take 80% percent of the trials per subject for training and 20% for testing, 5-fold cross-validation
    eeg_est_trials = []
    att_est_trials = []
    unatt_est_trials = []

    for subj in est_subs:
        eeg_est_trials += eeg_dict[subj]
        att_trials, unatt_trials = utils_stream.select_att_unatt_feats(feats_dict[subj], labels_dict[subj])
        att_est_trials += att_trials
        unatt_est_trials += unatt_trials

    CCA = CorrelationAnalysis([eeg_est_trials, att_est_trials], 'MCCA_LW', fs, [hparadata, hparafeats], n_components=3, leave_out=leave_out_persubj*len(est_subs), VALSET=False, CROSSVAL=True)
    corr_att, corr_unatt, _, _ = CCA.vad_mm(trial_len=trial_len, feat_unatt_list=unatt_est_trials, MM=False)
    corr_att_sum = cal_corr_sum_multi_trials(corr_att, range_into_account=range_into_account, nb_comp_into_account=nb_comp_into_account)
    corr_unatt_sum = cal_corr_sum_multi_trials(corr_unatt, range_into_account=range_into_account, nb_comp_into_account=nb_comp_into_account)
    return corr_att_sum, corr_unatt_sum


def fit_gmm(corr_att_sum, corr_unatt_sum, n_components_per_class=1):
    """
    Fit Gaussian Mixture Model (GMM) to the correlation coefficients.
    
    Parameters:
    - corr_att_sum: Correlation coefficients for attended trials.
    - corr_unatt_sum: Correlation coefficients for unattended trials.
    
    Returns:
    - gmm_0: GMM fitted to unattended-attended data.
    - gmm_1: GMM fitted to attended-unattended data.
    """
    corr_att_unatt = np.stack((corr_att_sum, corr_unatt_sum), axis=1)
    corr_unatt_att = np.stack((corr_unatt_sum, corr_att_sum), axis=1)
    # Train GMM for class 0 
    gmm_0 = GaussianMixture(n_components=n_components_per_class, 
                            covariance_type='full', 
                            random_state=42)
    gmm_0.fit(corr_unatt_att)
    # Train GMM for class 1 
    gmm_1 = GaussianMixture(n_components=n_components_per_class, 
                            covariance_type='full', 
                            random_state=42)
    gmm_1.fit(corr_att_unatt)
    return gmm_0, gmm_1


def predict_proba(X, gmm_0, gmm_1, class_priors=None):
    """
    Compute class probabilities for each point in X.
    
    Parameters:
    -----------
    X : array-like, shape (n_samples, 2)
        The input data points
    gmm_red : fitted GaussianMixture for class 0
    gmm_blue : fitted GaussianMixture for class 1
    class_priors : array-like, shape (2,), optional
        Prior probabilities of the classes. If None, class priors
        are calculated based on the proportion of samples.
        
    Returns:
    --------
    probas : array-like, shape (n_samples, 2)
        The class probabilities for each sample
    """
    if X.ndim == 1:
        X = X.reshape(1, -1)
    # Default priors based on class sizes
    if class_priors is None:
        class_priors = np.array([0.5, 0.5])
    # Get log likelihood for each class
    ll_0 = gmm_0.score_samples(X)
    ll_1 = gmm_1.score_samples(X)
    # Apply log priors
    ll_0 += np.log(class_priors[0])
    ll_1 += np.log(class_priors[1])
    # Convert to probabilities (softmax)
    ll_combined = np.column_stack([ll_0, ll_1])
    ll_max = np.max(ll_combined, axis=1, keepdims=True)
    exp_ll = np.exp(ll_combined - ll_max)
    probas = exp_ll / np.sum(exp_ll, axis=1, keepdims=True)
    return probas


