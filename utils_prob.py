import numpy as np
import utils
import utils_stream
import matplotlib.pyplot as plt
from algo_ccazoo import CorrelationAnalysis
from sklearn.mixture import GaussianMixture
from scipy.special import erf, erfc
from scipy.optimize import fsolve
from scipy.stats import norm


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
    # Train GMM for class 0 -> (corr_1, corr_2) f1 is unattended, f2 is attended
    gmm_0 = GaussianMixture(n_components=n_components_per_class, 
                            covariance_type='full', 
                            random_state=42)
    gmm_0.fit(corr_unatt_att)
    # Train GMM for class 1 -> (corr_1, corr_2) f1 is attended, f2 is unattended
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
    gmm_0 : fitted GaussianMixture for class 0
    gmm_1 : fitted GaussianMixture for class 1
    class_priors : array-like, shape (2,), optional
        Prior probabilities of the classes.
        
    Returns:
    --------
    probas : array-like, shape (n_samples, 2)
        The class probabilities for each sample
        first column is class 0 (f2 is att), second column is class 1 (f1 is att)
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


def predict_acc_bpsk(corr_pairs):
    """
    Unsupervised accuracy estimation from correlations
    M. A. Lopez-Gordo, S. Geirnaert and A. Bertrand, "Unsupervised Accuracy Estimation for Brain-Computer Interfaces Based on Selective Auditory Attention Decoding," in IEEE Transactions on Biomedical Engineering, doi: 10.1109/TBME.2025.3542253
    Input:
    corr_pairs : array-like, shape (n_samples, 2). The correlations between (EEG, feat1) and (EEG, feat2) on different windows.
    Output:
    acc: predicted accuracy
    """
    Z_s = np.sum(corr_pairs, axis=1)
    mu_s = np.mean(Z_s)
    sigma_d = np.std(Z_s)
    def equation_to_solve(x):
        """The equation whose positive root we need to find"""
        if x <= 0:
            return float('inf') 
        term1 = np.sqrt(2 / np.pi) * sigma_d * np.exp(-x**2 / 2 / sigma_d**2)
        term2 = x * erf(x / np.sqrt(2 * sigma_d**2))
        term3 = np.mean(np.abs(corr_pairs[:, 0] - corr_pairs[:, 1]))
        return term1 + term2 - term3
    x = fsolve(equation_to_solve, 1)[0]
    assert x > 0, "The root of the equation should be positive"
    BER = 0.5 * erfc(x / np.sqrt(2) / sigma_d)
    acc = 1 - BER
    mu_a = (mu_s + x) / 2
    mu_u = (mu_s - x) / 2
    sigma = sigma_d / np.sqrt(2)
    return acc, (mu_a, mu_u, sigma)


def predict_proba_bpsk(X, stats):
    """
    Compute class probabilities for each point in X, with statistics estimated under bpsk framework.
    
    Parameters:
    -----------
    X : array-like, shape (n_samples, 2)
    stats: (mu_a, mu_u, sigma), the mean and standard deviation of the distributions of p(rho|feat=att) and p(rho|feat=unatt)
        
    Returns:
    --------
    probas : array-like, shape (n_samples, 2)
        The class probabilities for each sample
        first column is class 0 (f2 is att), second column is class 1 (f1 is att)
    """
    if X.ndim == 1:
        X = X.reshape(1, -1)
    mu_a, mu_u, sigma = stats
    norm_att = norm(loc=mu_a, scale=sigma)
    norm_unatt = norm(loc=mu_u, scale=sigma)
    probas = np.zeros_like(X)
    for i in range(X.shape[0]):
        rho_1 = X[i, 0]
        rho_2 = X[i, 1]
        # Calculate log probabilities to avoid underflow
        log_term1 = norm_att.logpdf(rho_1) + norm_unatt.logpdf(rho_2)
        log_term2 = norm_att.logpdf(rho_2) + norm_unatt.logpdf(rho_1)
        # Use log-sum-exp trick
        max_log = max(log_term1, log_term2)
        # Subtract max to prevent overflow, then exp
        exp_term1 = np.exp(log_term1 - max_log)
        exp_term2 = np.exp(log_term2 - max_log)
        # Calculate probabilities
        sum_terms = exp_term1 + exp_term2
        p_1a = exp_term1 / sum_terms
        p_2a = exp_term2 / sum_terms
        probas[i, 1] = p_1a
        probas[i, 0] = p_2a
    return probas