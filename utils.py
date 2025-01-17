import numpy as np
import random
import mne
import scipy.io
import matplotlib.pyplot as plt
import copy
import pickle
import os
import pandas as pd
import seaborn as sns
from numpy import linalg as LA
from scipy import signal
from scipy.linalg import toeplitz, eig, eigh, sqrtm, lstsq, block_diag
from scipy.sparse.linalg import eigs
from scipy.stats import zscore, pearsonr, binomtest, binom
from sklearn.covariance import LedoitWolf


def cross_corrcoef(A, B, rowvar=True):
    """Cross correlation of two matrices.

    Args:
        A (np.ndarray or torch.Tensor): Matrix of size (n, p).
        B (np.ndarray or torch.Tensor): Matrix of size (n, q).
        rowvar (bool, optional): Whether to calculate the correlation along the rows. 

    Returns:
        np.ndarray or torch.Tensor: Matrix of size (p, q) containing the cross correlation of A and B.
    """
    if rowvar is False:
        A = A.T
        B = B.T

    A = A - A.mean(axis=1, keepdims=True)
    B = B - B.mean(axis=1, keepdims=True)

    C = A @ B.T

    A = np.sqrt(np.sum(A**2, axis=1))
    B = np.sqrt(np.sum(B**2, axis=1))

    return C / np.outer(A, B)


def expand_data_to_3D(data):
    '''
    Expand 1D or 2D data to 3D data
    '''
    if np.ndim(data) == 1:
        data = np.expand_dims(np.expand_dims(data, axis=1), axis=2)
    elif np.ndim(data) == 2:
        data = np.expand_dims(data, axis=2)
    return data


def regress_out(X, Y):
    '''
    Regress out Y from X
    X: T x Dx or T x Dx x N
    Y: T x Dy or T,
    '''
    if np.ndim(Y) == 1:
        Y = np.expand_dims(Y, axis=1)
    if np.ndim(X) == 3:
        X_res = copy.deepcopy(X)
        for i in range(X.shape[2]):
            W = lstsq(Y, X[:,:,i])[0]
            X_res[:,:,i] = X[:,:,i] - Y @ W
    elif np.ndim(X) == 2:
        W = lstsq(Y, X)[0]
        X_res = X - Y @ W
    else:
        raise ValueError('Check the dimension of X')
    return X_res


def further_regress_out(data_3D, confound_3D, L_d, L_c, offset_d, offset_c):
    N = data_3D.shape[2]
    data_clean_list = []
    for n in range(N):
        data = block_Hankel(data_3D[:,:,n], L_d, offset_d)
        confound = block_Hankel(confound_3D[:,:,n], L_c, offset_c)
        data_clean = regress_out(data, confound)
        data_clean_list.append(data_clean)
    data_clean_3D = np.stack(data_clean_list, axis=2)
    return data_clean_3D


def further_regress_out_list(X_list, confound_list, L_d, L_c, offset_d, offset_c):
    X_list = [expand_data_to_3D(X) for X in X_list]
    confound_list = [expand_data_to_3D(confound) for confound in confound_list]
    N = max(X_list[0].shape[2], confound_list[0].shape[2])
    X_list = [np.tile(X,(1,1,N)) if X.shape[2] == 1 else X for X in X_list]
    confound_list = [np.tile(confound,(1,1,N)) if confound.shape[2] == 1 else confound for confound in confound_list]
    # Do regression per video
    X_reg_list = [further_regress_out(X, confound, L_d, L_c, offset_d, offset_c) for X, confound in zip(X_list, confound_list)]
    return X_reg_list


def stack_modal(modal_nested_list):
    nb_video = len(modal_nested_list[0])
    dim_list = [modal[0].shape[1] for modal in modal_nested_list]
    stacked_list = []
    for i in range(nb_video):
        modal_list = [modal[i] for modal in modal_nested_list]
        modal_stacked = np.concatenate(tuple(modal_list), axis=1)
        stacked_list.append(modal_stacked)
    return stacked_list, dim_list


def get_cov_mtx(X, dim_list, regularization=None):
    '''
    Get the covariance matrix of X (T x dimX) with or without regularization
    dim_ilst is a list of dimensions of each modality (data from different subjects can also be viewed as different modalities)
    sum(dim_list) = dimX
    '''
    Rxx = np.cov(X, rowvar=False)
    Dxx = np.zeros_like(Rxx)
    dim_accumu = 0
    for dim in dim_list:
        if regularization == 'lwcov':
            Rxx[dim_accumu:dim_accumu+dim, dim_accumu:dim_accumu+dim] = LedoitWolf().fit(X[:, dim_accumu:dim_accumu+dim]).covariance_
        Dxx[dim_accumu:dim_accumu+dim, dim_accumu:dim_accumu+dim] = Rxx[dim_accumu:dim_accumu+dim, dim_accumu:dim_accumu+dim]
        dim_accumu += dim
    return Rxx, Dxx


def bandpass(data, fs, band):
    '''
    Bandpass filter
    Inputs:
        data: T x D
        fs: sampling frequency
        band: frequency band
    Outputs:
        filtered data: T x D        
    '''
    b, a = scipy.signal.butter(5, np.array(band), btype='bandpass', fs=fs)
    filtered = scipy.signal.filtfilt(b, a, data, axis=0)
    return filtered


def extract_freq_band(eeg, fs, band, normalize=False):
    '''
    Extract frequency band from EEG data
    Inputs:
        eeg: EEG data
        fs: sampling frequency
        band: frequency band
        normalize: whether to normalize the bandpassed data
    Outputs:
        eeg_band: bandpassed EEG data
    '''
    if eeg.ndim < 3:
        eeg_band = bandpass(eeg, fs, band)
        eeg_band = eeg_band / np.linalg.norm(eeg_band, 'fro') if normalize else eeg_band
    else:
        N = eeg.shape[2]
        eeg_band =np.zeros_like(eeg)
        for n in range(N):
            eeg_band[:,:,n] = bandpass(eeg[:,:,n], fs, band)
            eeg_band[:,:,n] = eeg_band[:,:,n] / np.linalg.norm(eeg_band[:,:,n], 'fro') if normalize else eeg_band[:,:,n]
    return eeg_band


def Hankel_mtx(L_timefilter, x, offset=0, mask=None):
    '''
    Calculate the Hankel matrix
    Convolution: y(t)=x(t)*h(t)
    In matrix form: y=Xh E.g. time lag = 3
    If offset=0,
    h = h(0); h(1); h(2)
    X = 
    x(0)   x(-1)  x(-2)
    x(1)   x(0)   x(-1)
            ...
    x(T-1) x(T-2) x(T-3)
    If offset !=0, e.g., offset=1,
    h = h(-1); h(0); h(1)
    X = 
    x(1)   x(0)   x(-1)
    x(2)   x(1)   x(0)
            ...
    x(T)   x(T-1) x(T-2)
    Unknown values are set as 0
    If mask is not None, then discard the rows indicated by mask
    This is useful when we want to remove segments (e.g., blinks, saccades) in the signals.
    '''
    first_col = np.zeros(L_timefilter)
    first_col[0] = x[0]
    if offset != 0:
        x = np.append(x, [np.zeros((1,offset))])
    hankel_mtx = np.transpose(toeplitz(first_col, x))
    if offset != 0:
        hankel_mtx = hankel_mtx[offset:,:]
    if mask is not None:
        hankel_mtx = hankel_mtx[mask,:]
    return hankel_mtx


def block_Hankel(X, L, offset=0, mask=None):
    '''
    For spatial-temporal filter, calculate the block Hankel matrix
    Inputs:
    X: T(#sample)xD(#channel)
    L: number of time lags; 
    offset: offset of time lags; from -(L-1) to 0 (offset=0) or offset-(L-1) to offset
    '''
    if np.ndim(X) == 1:
        X = np.expand_dims(X, axis=1)
    Hankel_list = [Hankel_mtx(L, X[:,i], offset, mask) for i in range(X.shape[1])]
    blockHankel = np.concatenate(tuple(Hankel_list), axis=1)
    return blockHankel


def hankelize_data_multisub(data_multisub, L, offset):
    N = data_multisub.shape[2]
    X_list = [block_Hankel(data_multisub[:,:,n], L, offset) for n in range(N)]
    X_list = [np.expand_dims(X, axis=2) for X in X_list]
    X = np.concatenate(tuple(X_list), axis=2)
    return X


def normalize_per_view(data):
    data_center = data - np.mean(data, axis=0)
    scale = np.sqrt(np.sqrt(data.shape[1]) / LA.norm(np.transpose(data_center) @ data_center))
    data_norm = data_center * scale
    return data_norm


def normalize_multi_views(data_list):
    data_norm_list = [normalize_per_view(data) for data in data_list]
    return data_norm_list


def random_combination(iterable, r):
    "Random selection from itertools.combinations(iterable, r)"
    pool = tuple(iterable)
    n = len(pool)
    indices = sorted(random.sample(range(n), r))
    return [pool[i] for i in indices]


def get_possible_offset(offset_list, timeline, min_shift):
    '''
    Create a list of possible offsets [ofs_1, ofs_2, ...]
    ofs_i is not in the range of [ofs_j-min_shift, ofs_j+min_shift] for any j!=i
    '''
    T = len(timeline)
    exclude_range = set(range(min_shift))
    for offset in offset_list:
        add_exclude_range = range(max(0, offset - min_shift), min(offset + min_shift, T))
        exclude_range = set(exclude_range).union(set(add_exclude_range))
    return list(set(range(T)) - exclude_range)


def random_shift_3D(X, min_shift):
    '''
    Randomly shift the data of each subject
    The offset with respect to each other must be at least min_shift
    '''
    T, _, N = X.shape
    X_shifted = np.zeros(X.shape)
    offset_list = []
    for n in range(N):
        possible_offset = get_possible_offset(offset_list, range(T), min_shift)
        offset_list.append(random.sample(possible_offset, 1)[0])
        X_shifted[:,:,n] = np.concatenate((X[offset_list[n]:,:,n], X[:offset_list[n],:,n]), axis=0)
    return X_shifted, offset_list


def transformed_GEVD(Dxx, Rxx, rho, dimStim, n_components):
    Rxx_hat = copy.deepcopy(Rxx)
    Rxx_hat[:, -dimStim:] = np.sqrt(rho) * Rxx_hat[:, -dimStim:]
    Rxx_hat[-dimStim:, :] = np.sqrt(rho) * Rxx_hat[-dimStim:, :]
    Rxx_hat = (Rxx_hat + Rxx_hat.T)/2
    lam, W = eigh(Dxx, Rxx_hat, subset_by_index=[0,n_components-1]) # automatically ascend
    W[-dimStim:, :] = np.sqrt(1/rho) * W[-dimStim:, :]
    # Alternatively:
    # Rxx_hat = copy.deepcopy(Rxx)
    # Rxx_hat[:,-D_stim*L_Stim:] = Rxx_hat[:,-D_stim*L_Stim:]*rho
    # Rxx_hat[-D_stim*L_Stim:,:] = Rxx_hat[-D_stim*L_Stim:,:]*rho
    # Dxx_hat = copy.deepcopy(Dxx)
    # Dxx_hat[:,-D_stim*L_Stim:] = Dxx_hat[:,-D_stim*L_Stim:]*rho
    # lam, W = eigh(Dxx_hat, Rxx_hat, subset_by_index=[0,self.n_components-1])
    return lam, W


def into_trials(data, fs, t=60, start_points=None):
    if np.ndim(data)==1:
        data = np.expand_dims(data, axis=1)
    T = data.shape[0]
    if start_points is not None:
        # has a target number of trials with specified start points, then randomly select nb_trials trials
        # select data from start_points along axis 0
        data_trials = [data[start:start+fs*t, ...] for start in start_points]
    else:
        # does not have a target number of trials, then divide data into t s trials without overlap
        # if T is not a multiple of t sec, then discard the last few samples
        T_trunc = T - T%(fs*t)
        data_intmin = data[:T_trunc, ...]
        # segment X_intmin into 1 min trials along axis 0
        data_trials = np.split(data_intmin, int(T/(fs*t)), axis=0)
    return data_trials


def into_trials_with_overlap(data, fs, t=30, overlap=0.9, PERMUTE=False):
    if np.ndim(data) == 1:
        data = np.expand_dims(data, axis=1)
    T = data.shape[0]  
    window_size = int(t * fs)
    if T < window_size:
        print("Data is shorter than the specified window size. Returning an empty list.")
        return []
    if PERMUTE:
        shift_len = max(int(window_size / 2), int(T/3))
        data = np.concatenate((data[shift_len:], data[:shift_len]), axis=0)
    step = max(int(window_size * (1 - overlap)), fs)
    data_trials = [data[i:i + window_size, ...] for i in range(0, T - window_size + 1, step)]
    return data_trials


def select_distractors(data, fs, t, start_point):
    adjacent_start = max(start_point - fs, 0)
    adjacent_end = min(start_point + (t+1)*fs, data.shape[0])
    # remove the target trial from the data
    data_distractor = np.delete(data, range(adjacent_start, adjacent_end), axis=0)
    # randomly select one trial from the rest of the data
    start_points_distractor = np.random.randint(0, len(data_distractor)-t*fs, size=1)[0]
    seg_distractor = data_distractor[start_points_distractor:start_points_distractor+t*fs, ...]
    return seg_distractor


def select_distractors_list(data_list, fs, t, start_point):
    assert all(data.shape[0] == data_list[0].shape[0] for data in data_list)
    adjacent_start = max(start_point - fs, 0)
    adjacent_end = min(start_point + (t+1)*fs, data_list[0].shape[0])
    # remove the target trial from the data
    data_distractor_list = [np.delete(data, range(adjacent_start, adjacent_end), axis=0) for data in data_list]
    # randomly select one trial from the rest of the data
    start_points_distractor = np.random.randint(0, len(data_distractor_list[0])-t*fs, size=1)[0]
    seg_distractor_list = [data_distractor[start_points_distractor:start_points_distractor+t*fs, ...] for data_distractor in data_distractor_list]
    return seg_distractor_list


def shift_trials(data_trials, shift=None):
    '''
    Given a list of trials, move half of the trials to the end of the list
    '''
    nb_trials = len(data_trials)
    if shift is None:
        shift = nb_trials//2
    trials_shifted = [data_trials[(n+shift)%nb_trials] for n in range(nb_trials)]
    return trials_shifted


def split(EEG, Sti, fold=10, fold_idx=1):
    '''
    Split datasets as one fold specified by fold_idx (test set), and the rest folds (training set). 
    '''
    T = EEG.shape[0]
    len_test = T // fold
    if np.ndim(EEG)==2:
        EEG_test = EEG[len_test*(fold_idx-1):len_test*fold_idx,:]
        EEG_train = np.delete(EEG, range(len_test*(fold_idx-1), len_test*fold_idx), axis=0)
    elif np.ndim(EEG)==3:
        EEG_test = EEG[len_test*(fold_idx-1):len_test*fold_idx,:,:]
        EEG_train = np.delete(EEG, range(len_test*(fold_idx-1), len_test*fold_idx), axis=0)
    else:
        print('Warning: Check the dimension of EEG data')
    if np.ndim(Sti)==1:
        Sti = np.expand_dims(Sti, axis=1)
    Sti_test = Sti[len_test*(fold_idx-1):len_test*fold_idx,:]
    Sti_train = np.delete(Sti, range(len_test*(fold_idx-1), len_test*fold_idx), axis=0)
    return EEG_train, EEG_test, Sti_train, Sti_test


def split_multi_mod(datalist, fold=10, fold_idx=1):
    '''
    Split datasets as one fold specified by fold_idx (test set), and the rest folds (training set). 
    Datasets are organized in datalist.
    '''
    train_list = []
    test_list = []
    for data in datalist:
        T = data.shape[0]
        len_test = T // fold
        if np.ndim(data)==1:
            data_test = np.expand_dims(data[len_test*(fold_idx-1):len_test*fold_idx], axis=1)
            data_train = np.expand_dims(np.delete(data, range(len_test*(fold_idx-1), len_test*fold_idx)), axis=1)
        elif np.ndim(data)==2:
            data_test = data[len_test*(fold_idx-1):len_test*fold_idx,:]
            data_train = np.delete(data, range(len_test*(fold_idx-1), len_test*fold_idx), axis=0)
        elif np.ndim(data)==3:
            data_test = data[len_test*(fold_idx-1):len_test*fold_idx,:,:]
            data_train = np.delete(data, range(len_test*(fold_idx-1), len_test*fold_idx), axis=0)
        else:
            print('Warning: Check the dimension of data')
        train_list.append(data_train)
        test_list.append(data_test)
    return train_list, test_list


def split_multi_mod_LVO(nested_datalist, leave_out=2):
    '''
    Datasets are organized in nested datalist: [[EEG_1, EEG_2, ... ], [Vis_1, Vis_2, ... ], [Sd_1, Sd_2, ... ]]
    Create training and test lists for leave-one-video-out cross-validation
    - A training list: [[EEG_train, Vis_train, Sd_train]_1, [EEG_train, Vis_train, Sd_train]_2, ...]
    - A test list: [[EEG_test, Vis_test, Sd_test]_1, [EEG_test, Vis_test, Sd_test]_2, ...]
    '''
    nb_videos = len(nested_datalist[0])
    train_list_folds = []
    test_list_folds = []
    for i in range(0, nb_videos, leave_out):
        indices_train = [j for j in range(nb_videos) if j not in range(i, i+leave_out)]
        indices_test = [j for j in range(i, i+leave_out)]
        train_list_folds.append([np.concatenate(tuple([mod[i] for i in indices_train]), axis=0) if mod is not None else None for mod in nested_datalist])
        test_list_folds.append([np.concatenate(tuple([mod[i] for i in indices_test]), axis=0) if mod is not None else None for mod in nested_datalist])
    return train_list_folds, test_list_folds


def split_multi_mod_withval_LVO(nested_datalist, leave_out=2, VAL=True, CV=True):
    '''
    Datasets are organized in nested datalist: [[EEG_1, EEG_2, ... ], [Vis_1, Vis_2, ... ], [Sd_1, Sd_2, ... ]]
    Create training and test lists for leave-one-video-out cross-validation
    - A training list: [[EEG_train, Vis_train, Sd_train]_1, [EEG_train, Vis_train, Sd_train]_2, ...]
    - A test list: [[EEG_test, Vis_test, Sd_test]_1, [EEG_test, Vis_test, Sd_test]_2, ...]
    - A validation list: [[EEG_val, Vis_val, Sd_val]_1, [EEG_val, Vis_val, Sd_val]_2, ...]
    '''
    nb_videos = len(nested_datalist[0])
    train_list_folds = []
    test_list_folds = []
    val_list_folds = [] if VAL else None
    for i in range(0, nb_videos, leave_out):
        indices_train = [j for j in range(nb_videos) if j not in range(i, i+leave_out)]
        indices_test = [j for j in range(i, i+leave_out-1)] if VAL else [j for j in range(i, i+leave_out)]
        train_list_folds.append([np.concatenate(tuple([mod[i] for i in indices_train]), axis=0) if mod is not None else None for mod in nested_datalist])
        test_list_folds.append([np.concatenate(tuple([mod[i] for i in indices_test]), axis=0) if mod is not None else None for mod in nested_datalist])
        if VAL:
            index_val = [i+leave_out-1]
            val_list_folds.append([np.concatenate(tuple([mod[i] for i in index_val]), axis=0) if mod is not None else None for mod in nested_datalist])
        if not CV:
            break             
    return train_list_folds, test_list_folds, val_list_folds


def sig_level_binomial_test(p_value, total_trials, p=0.5):
    critical_value = binom.ppf(1-p_value, total_trials, p=p)
    critical_value = critical_value+1 if (1 - binom.cdf(critical_value, total_trials, p=p) > p_value) else critical_value # in case ppf returns a value that leads to a closer but larger p-value
    sig_level = int(critical_value+1)/total_trials
    return sig_level


def eval_compete(corr_att, corr_unatt, TRAIN_WITH_ATT, range=5, nb_comp_into_account=2):
    nb_test = corr_att.shape[0]
    corr_att = np.array([np.sort(row[:range])[::-1] for row in corr_att])
    corr_unatt = np.array([np.sort(row[:range])[::-1] for row in corr_unatt])
    nb_correct = sum(corr_att[:,:nb_comp_into_account].sum(axis=1)>corr_unatt[:,:nb_comp_into_account].sum(axis=1))
    if not TRAIN_WITH_ATT:
        nb_correct = nb_test - nb_correct
    acc = nb_correct/nb_test
    print('Accuracy: ', acc, 'Number of tests: ', nb_test)
    return acc


def W_organize(W, datalist, Llist=None):
    '''
    Input: 
    W generated by GCCA_multi_modal
    Output:
    Organized W list containing W of each modality 
    '''
    W_list = []
    dim_start = 0
    for i in range(len(datalist)):
        rawdata = datalist[i]
        L = Llist[i] if Llist is not None else 1
        if np.ndim(rawdata) == 3:
            _, D, N = rawdata.shape
            dim_end = dim_start + D*L*N
            W_temp = W[dim_start:dim_end,:]
            W_stack = np.reshape(W_temp, (N,D*L,-1))
            W_list.append(np.transpose(W_stack, [1,0,2]))
        elif np.ndim(rawdata) == 2:
            _, D = rawdata.shape
            dim_end = dim_start + D*L
            W_list.append(W[dim_start:dim_end,:])
        else:
            print('Warning: Check the dim of data')
        dim_start = dim_end
    return W_list


def F_organize(F_redun, L, offset, avg=True):
    '''
    Extract the forward model corresponding to the correct time points from the redundant forward model F_redun
    Input: 
    F_redun: DLxNxK or DLxK
    Output:
    Forward model DxNxK or DxK
    '''
    if np.ndim(F_redun) == 3:
        DL, _, _ = F_redun.shape
    else:
        DL, _ = F_redun.shape
    D = int(DL/L)
    indices = [i*L+offset for i in range(D)]
    if np.ndim(F_redun) == 3:
        F = F_redun[indices,:,:]
        if avg:
            F = np.average(F, axis=1)
    else:
        F = F_redun[indices,:]
    return F


def forward_model(X, W_Hankel, L=1, offset=0):
    '''
    Reference: On the interpretation of weight vectors of linear models in multivariate neuroimaging https://www.sciencedirect.com/science/article/pii/S1053811913010914
    Backward models: Extract latent factors as functions of the observed data s(t) = W^T x(t)
    Forward models: Reconstruct observations from latent factors x(t) = As(t) + n(t)
    x(t): D-dimensional observations
    s(t): K-dimensional latent factors
    W: backward model
    A: forward model

    In our use case the backward model can be found using (G)CCA. Latent factors are the representations generated by different components.
    X_Hankel W_Hankel = S     X:TxDL W:DLxK S:TxK
    S F.T = X                 F: DxK
    F = X.T X_Hankel W_Hankel inv(W_Hankel.T X_Hankel.T X_Hankel W_Hankel)

    Inputs:
    X: observations (one subject) TxD
    W_Hankel: filters/backward models DLxK
    L: time lag (if temporal-spatial)

    Output:
    F: forward model
    '''
    if L == 1:
        Rxx = np.cov(X, rowvar=False)
        F = Rxx@W_Hankel@LA.inv(W_Hankel.T@Rxx@W_Hankel)
    else:
        X_block_Hankel = block_Hankel(X, L, offset)
        F = X.T@X_block_Hankel@W_Hankel@LA.inv(W_Hankel.T@X_block_Hankel.T@X_block_Hankel@W_Hankel)
    return F


def aggregated_space_time(data, weights, lag, KEEPDIM='S'):
    assert np.ndim(data) == np.ndim(weights)
    if np.ndim(data) == 2:
        data = np.expand_dims(data, axis=2)
        weights = np.expand_dims(weights, axis=2)
    dim = int(data.shape[1]/lag)
    nb_subj = data.shape[2] 
    aggregated_per_subj = []
    trans_per_subj = []
    for i in range(nb_subj):
        data_subj = data[:,:,i]
        weights_subj = weights[:,:,i]
        trans_per_subj.append(data_subj @ weights_subj)
        if KEEPDIM == 'S':
            # divide weights into D L x K matrices, and construct a block diagonal matrix from them
            w_per_dim = np.vsplit(weights_subj, dim)
            w_block = block_diag(*w_per_dim) # DL x DK
            aggregated_per_subj.append(data_subj @ w_block)
        elif KEEPDIM == 'T':
            w_reorg = weights_subj.reshape((lag, dim, -1), order='F')
            w_reorg = np.transpose(w_reorg, (1, 0, 2))
            w_reorg = w_reorg.reshape((dim*lag, -1), order='F')
            w_per_lag = np.vsplit(w_reorg, lag)
            w_block = block_diag(*w_per_lag) # DL x LK
            aggregated_per_subj.append(data_subj @ w_block)
        elif KEEPDIM =='ST':
            w_per_element = np.vsplit(weights_subj, dim*lag)
            w_block = block_diag(*w_per_element) # DL x DLK
            aggregated_per_subj.append(data_subj @ w_block)
        else:
            raise ValueError('Invalid mode')
    aggregated_3D = np.stack(aggregated_per_subj, axis=2) if nb_subj > 1 else np.expand_dims(aggregated_per_subj[0], axis=2)
    trans_3D = np.stack(trans_per_subj, axis=2) if nb_subj > 1 else np.expand_dims(trans_per_subj[0], axis=2)
    return aggregated_3D, trans_3D


def get_influence_all_views(data_views_h, weights_views, lag_views, TRAINMODE, TRACKMODE='S', aggcomp=None, CROSSVIEW=False, NORMALIZATION=True):
    '''
    Calculate the influence of each channel/each lag/each channel-lag
    '''
    influence_views = []
    aggregated_3D_views = []
    trans_3D_views = []
    for data, weights, lag in zip(data_views_h, weights_views, lag_views):
        aggregated_3D, trans_3D = aggregated_space_time(data, weights, lag, KEEPDIM=TRACKMODE)
        aggregated_3D_views.append(aggregated_3D)
        trans_3D_views.append(trans_3D)
    if CROSSVIEW:
        assert len(data_views_h) == 2, 'Cross-view influence tracking can only be applied to two views'
        trans_3D_views = [trans_3D_views[1], trans_3D_views[0]]
    for aggregated_3D, trans_3D in zip(aggregated_3D_views, trans_3D_views):
        nb_subj = trans_3D.shape[2]
        if TRAINMODE == 'ViaShared' or TRAINMODE == 'SINDP_avg':
            data_trans = np.mean(trans_3D, axis=2)
            data_aggregated = np.mean(aggregated_3D, axis=2)
        elif TRAINMODE == 'ConcatSubj':
            data_trans = np.stack([trans_3D[:,:,i] for i in range(nb_subj)], axis=0)
            data_aggregated = np.stack([aggregated_3D[:,:,i] for i in range(nb_subj)], axis=0)
        else:
            if nb_subj != 1:
                raise ValueError('Influence tracking is not implemented or tested for training mode: ', TRAINMODE)
            else:
                data_trans = trans_3D[:,:,0]
                data_aggregated = aggregated_3D[:,:,0]
        rt = data_aggregated.shape[1]//data_trans.shape[1]
        data_rep = np.tile(data_trans, rt)
        corr = np.array([np.corrcoef(data_rep[:, i], data_aggregated[:, i])[0, 1] for i in range(data_rep.shape[1])])
        influence = np.reshape(np.abs(corr), (rt, -1))
        if aggcomp is not None:
            influence = np.sum(influence[:,:aggcomp], axis=1, keepdims=True)
        if NORMALIZATION:
            influence = influence / LA.norm(influence, axis=0)
        influence_views.append(influence)
    return influence_views


def get_influence_all_views_two_layers(data_views_h, w1_data, w2_data, w2_feats, lag1_data, lag2_feats, TRAINMODE, TRACKMODE='S', aggcomp=None, CROSSVIEW=False):
    if np.ndim(w1_data) == 2:
        w12_data = w1_data @ w2_data
    else:
        assert np.ndim(w1_data) == 3
        w12_data = np.stack([w1_data[:,:,i] @ w2_data for i in range(w1_data.shape[2])], axis=2)
    w_views = [w12_data, w2_feats]
    lag_views = [lag1_data, lag2_feats]
    influence_views = get_influence_all_views(data_views_h, w_views, lag_views, TRAINMODE, TRACKMODE, aggcomp=aggcomp, CROSSVIEW=CROSSVIEW)
    return influence_views


def phase_scramble_2D(data):
    # Initialize an array to hold the scrambled data
    scrambled_data = np.zeros_like(data, dtype=complex)
    # Loop over channels
    for i in range(data.shape[1]):  # Assuming data.shape = (time, channel)
        # Perform FFT on each channel independently
        fft_result = np.fft.fft(data[:, i])
        amplitude = np.abs(fft_result)
        T = len(data[:, i])
        # Generate random phase shifts for half of the spectrum
        half_T = T // 2 if T % 2 == 0 else (T + 1) // 2
        random_phase_half = np.exp(1j * np.random.uniform(0, 2*np.pi, size=half_T))
        # Ensure conjugate symmetry
        random_phase_full = np.concatenate(([1], random_phase_half[1:half_T], [1], np.conj(random_phase_half[1:half_T][::-1]))) if T%2==0 else np.concatenate(([1], random_phase_half[1:half_T], np.conj(random_phase_half[1:half_T][::-1])))
        # Apply the random phase shifts
        scrambled_fft = amplitude * random_phase_full
        scrambled_data[:, i] = np.fft.ifft(scrambled_fft)
    return scrambled_data.real


def phase_scramble_3D(data):
    _, _, N = data.shape
    scrambled_data = np.zeros_like(data)
    for n in range(N):
        scrambled_data[:,:,n] = phase_scramble_2D(data[:,:,n])
    return scrambled_data


def shuffle_block(X, block_len):
    '''
    Shuffle the blocks of X along the time axis for each subject.
    '''
    T, D, N = X.shape
    if T%block_len != 0:
        append_arr = np.zeros((block_len-T%block_len, D, N))
        X = np.concatenate((X, append_arr), axis=0)
    T_appended = X.shape[0]
    X_shuffled = np.zeros_like(X)
    for n in range(N):
        blocks = [X[i:i+block_len, :, n] for i in range(0, T_appended, block_len)]
        random.shuffle(blocks)
        X_shuffled[:,:,n] = np.concatenate(tuple(blocks), axis=0)
    return X_shuffled


def shuffle_2D(X, block_len):
    T, D = X.shape
    if T%block_len != 0:
        append_arr = np.zeros((block_len-T%block_len, D))
        X = np.concatenate((X, append_arr), axis=0)
        T, _ = X.shape
    X_block = X.reshape((T//block_len, block_len, D))
    X_shuffle_block = np.random.permutation(X_block)
    X_shuffle = X_shuffle_block.reshape((T, D))
    return X_shuffle


def shuffle_3D(X, block_len):
    '''
    Same as shuffle_block(X, block_len)
    '''
    T, D, N = X.shape
    if T%block_len != 0:
        append_arr = np.zeros((block_len-T%block_len, D, N))
        X = np.concatenate((X, append_arr), axis=0)
    X_shuffled = np.zeros_like(X)
    for n in range(N):
        X_shuffled[:,:,n] = shuffle_2D(X[:,:,n], block_len)
    return X_shuffled


def shuffle_datalist(datalist, block_len):
    '''
    Shuffle the blocks of X along the time axis for each subject.
    '''
    datalist_shuffled = []
    for data in datalist:
        if np.ndim(data) == 2:
            datalist_shuffled.append(shuffle_2D(data, block_len))
        elif np.ndim(data) == 3:
            datalist_shuffled.append(shuffle_3D(data, block_len))
    return datalist_shuffled


def EEG_normalization(data, len_seg):
    '''
    Normalize the EEG data.
    Subtract data of each channel by the mean of it
    Divide data into several segments, and for each segment, divide the data matrix by its Frobenius norm.
    Inputs:
    data: EEG data D x T
    len_seg: length of the segments
    Output:
    normalized_data
    '''
    _, T = data.shape
    n_blocks = T // len_seg + 1
    data_blocks = np.array_split(data, n_blocks, axis=1)
    data_zeromean = [db - np.mean(db, axis=1, keepdims=True) for db in data_blocks]
    normalized_blocks = [db/LA.norm(db) for db in data_zeromean]
    normalized_data = np.concatenate(tuple(normalized_blocks), axis=1)
    return normalized_data


def extract_highfreq(EEG, resamp_freqs, band=[15,20], ch_eog=None, regression=False, normalize=True):
    '''
    EEG signals -> band-pass filter -> high-frequency signals -> Hilbert transform -> signal envelope -> low-pass filter -> down-sampled envelope -> noramalized envelope
    Inputs:
    EEG: EEG signals with original sampling rate
    resamp_freqs: resampling frequency
    band: the frequency band to be kept
    Outputs:
    envelope: the envelope of high-frequency signals
    '''
    # EOG channels are marked as 'eeg' now
    # Filter both eeg and eog channels with a band-pass filter
    EEG_band = EEG.filter(l_freq=band[0], h_freq=band[1], picks=['eeg'])
    # Extract the envelope of signals
    envelope = EEG_band.copy().apply_hilbert(picks=['eeg'], envelope=True)
    # Mark EOG channels as 'eog'
    if ch_eog is not None:
        type_true = ['eog']*len(ch_eog)
        change_type_dict = dict(zip(ch_eog, type_true))
        envelope.set_channel_types(change_type_dict)
        # Regress out the filtered EOG signals before extracting the envelope of high-frequency signals
        if regression:
            EOGweights = mne.preprocessing.EOGRegression(picks='eeg', proj=False).fit(envelope)
            envelope = EOGweights.apply(envelope, copy=False)
    envelope = envelope.resample(sfreq=resamp_freqs)
    if normalize:
        eeg_channel_indices = mne.pick_types(envelope.info, eeg=True)
        eegdata, _ = envelope[eeg_channel_indices]
        envelope._data[eeg_channel_indices, :] = EEG_normalization(eegdata, resamp_freqs*60)
    return envelope


def preprocessing(file_path, HP_cutoff = 0.5, AC_freqs=50, band=None, resamp_freqs=None, bads=[], eog=True, regression=False, normalize=False):
    '''
    Preprocessing of the raw signal
    Re-reference -> Highpass filter (-> downsample)
    No artifact removal technique has been applied yet
    Inputs:
    file_path: location of the eeg dataset
    HP_cutoff: cut off frequency of the high pass filter (for removing DC components and slow drifts)
    AC_freqs: AC power line frequency
    resamp_freqs: resampling frequency (if None then resampling is not needed)
    bads: list of bad channels
    eog: if contains 4 eog channels
    regression: whether regresses eog out
    Output:
    preprocessed: preprocessed eeg
    fs: the sample frequency of the EEG signal (original or down sampled)
    '''
    raw_lab = mne.io.read_raw_eeglab(file_path, preload=True)
    raw_lab.info['bads'] = bads
    fsEEG = raw_lab.info['sfreq']
    # Rename channels and set montages
    biosemi_layout = mne.channels.read_layout('biosemi')
    ch_names_map = dict(zip(raw_lab.info['ch_names'], biosemi_layout.names))
    raw_lab.rename_channels(ch_names_map)
    montage = mne.channels.make_standard_montage('biosemi64')
    raw_lab.set_montage(montage)
    if len(bads)>0:
        # Interpolate bad channels
        raw_lab.interpolate_bads()
    # Re-reference
    # raw_lab.set_eeg_reference(ref_channels=['Cz']) # Select the reference channel to be Cz
    raw_lab.set_eeg_reference(ref_channels='average')
    # If there are EOG channels, first treat them as EEG channels and do re-referencing, filtering and resampling.
    if eog:
        misc_names = [raw_lab.info.ch_names[i] for i in mne.pick_types(raw_lab.info, misc=True)]
        # eog_data, _ = raw_lab[misc_names]
        # eog_channel_indices = mne.pick_channels(raw_lab.info['ch_names'], include=misc_names)
        type_eeg = ['eeg']*len(misc_names)
        change_type_dict = dict(zip(misc_names, type_eeg))
        raw_lab.set_channel_types(change_type_dict)
        # Take the average of four EOG channels as the reference
        # raw_lab._data[eog_channel_indices, :] = eog_data - np.average(eog_data, axis=0)
    else:
        misc_names = None
    # Highpass filter - remove DC components and slow drifts
    raw_highpass = raw_lab.copy().filter(l_freq=HP_cutoff, h_freq=None)
    # raw_highpass.compute_psd().plot(average=True)
    # Remove power line noise
    raw_notch = raw_highpass.copy().notch_filter(freqs=AC_freqs)
    # raw_notch.compute_psd().plot(average=True)
    # Resampling:
    # Anti-aliasing has been implemented in mne.io.Raw.resample before decimation
    # https://mne.tools/stable/generated/mne.io.Raw.html#mne.io.Raw.resample
    if resamp_freqs is not None:
        if band is not None:
            highfreq = extract_highfreq(raw_notch.copy(), resamp_freqs, band, misc_names, regression, normalize)
        else:
            highfreq = None
        raw_downsampled = raw_notch.copy().resample(sfreq=resamp_freqs)
        # raw_downsampled.compute_psd().plot(average=True)
        preprocessed = raw_downsampled
        fs = resamp_freqs
    else:
        highfreq = None
        preprocessed = raw_notch
        fs = fsEEG
    # Then set EOG channels to their true type
    if eog:
        type_true = ['eog']*len(misc_names)
        change_type_dict = dict(zip(misc_names, type_true))
        preprocessed.set_channel_types(change_type_dict)
    if regression:
        EOGweights = mne.preprocessing.EOGRegression(picks='eeg', proj=False).fit(preprocessed)
        preprocessed = EOGweights.apply(preprocessed, copy=False)
    if normalize:
        eeg_channel_indices = mne.pick_types(preprocessed.info, eeg=True)
        eegdata, _ = preprocessed[eeg_channel_indices]
        preprocessed._data[eeg_channel_indices, :] = EEG_normalization(eegdata, fs*60)
    return preprocessed, fs, highfreq


def clean_features(feats, smooth=True):
    y = copy.deepcopy(feats)
    T, nb_feature = y.shape
    for i in range(nb_feature):
        # interpolate NaN values (linearly)
        nans, x= np.isnan(y[:,i]), lambda z: z.nonzero()[0]
        if any(nans):
            f1 = scipy.interpolate.interp1d(x(~nans), y[:,i][~nans], fill_value='extrapolate')
            y[:,i][nans] = f1(x(nans))
        if smooth:
            # extract envelope by finding peaks and interpolating peaks with spline
            idx_peaks = scipy.signal.find_peaks(y[:,i])[0]
            idx_rest = np.setdiff1d(np.array(range(T)), idx_peaks)
            # consider use quadratic instead
            f2 = scipy.interpolate.interp1d(idx_peaks, y[:,i][idx_peaks], kind='cubic', fill_value='extrapolate')
            y[:,i][idx_rest] = f2(idx_rest)
    return y


def create_dir(path, CLEAR=False):
    if not os.path.exists(path):
        os.makedirs(path)
    if CLEAR:
        for file in os.listdir(path):
            os.remove(path + file)


def get_features(feats_path_folder, video_id, len_seg, offset=None, smooth=True):
    with open(feats_path_folder + video_id + '_mask.pkl', 'rb') as f:
        feats = pickle.load(f)
    feats = np.concatenate(tuple(feats), axis=0)
    feats = clean_features(feats, smooth=smooth)
    if offset is not None:
        end_idx = min(offset + len_seg, feats.shape[0])
        start_idx = end_idx - len_seg
        feats = feats[start_idx:end_idx, ...]
    else:
        feats = feats[:len_seg, ...]
    return feats


def get_gaze(gaze_path, len_seg, offset=None):
    gaze = np.load(gaze_path, allow_pickle=True)
    # interpolate missing values
    gaze = np.array([np.nan if x is None else x for x in gaze])
    gaze_clean = clean_features(gaze.astype(np.float64), smooth=False)
    if offset is not None:
        end_idx = min(offset + len_seg, gaze_clean.shape[0])
        start_idx = end_idx - len_seg
        gaze_clean = gaze_clean[start_idx:end_idx, :]
    else:
        gaze_clean = gaze_clean[:len_seg, :]
    return gaze_clean


def get_eeg_eog(eeg_path, fsStim, bads, expdim=True):
    eeg_prepro, fs, _ = preprocessing(eeg_path, HP_cutoff = 0.5, AC_freqs=50, band=None, resamp_freqs=fsStim, bads=bads, eog=True, regression=False, normalize=True)
    eeg_channel_indices = mne.pick_types(eeg_prepro.info, eeg=True)
    eog_channel_indices = mne.pick_types(eeg_prepro.info, eog=True)
    eeg_downsampled, _ = eeg_prepro[eeg_channel_indices]
    eog_downsampled, _ = eeg_prepro[eog_channel_indices]
    if expdim:
        eeg_downsampled = np.expand_dims(eeg_downsampled.T, axis=2)
        eog_downsampled = np.expand_dims(eog_downsampled.T, axis=2)
    return eeg_downsampled, eog_downsampled, fs


def data_per_subj(eeg_folder, fsStim, bads, feats_path_folder=None, expdim=True):
    eeg_files_all = [file for file in os.listdir(eeg_folder) if file.endswith('.set')]
    files = [file for file in eeg_files_all if len(file.split('_')) == 3]
    files.sort()
    nb_files = len(files)
    eeg_list = []
    eog_list = []
    len_seg_list = []
    gaze_list = []
    for file in files:
        eeg_downsampled, eog_downsampled, fs = get_eeg_eog(eeg_folder + file, fsStim, bads, expdim)
        eeg_list.append(eeg_downsampled)
        eog_list.append(eog_downsampled)
        len_seg_list.append(eeg_downsampled.shape[0])
        id_att = file[:-4].split('_')[-1]
        gaze_file = [file for file in os.listdir(eeg_folder) if file.endswith('.npy') and file.split('_')[-2]==id_att]
        if len(gaze_file) == 1:
            gaze = get_gaze(eeg_folder + gaze_file[0], len_seg_list[-1])
            gaze = np.expand_dims(gaze, axis=2)
        else:
            gaze = np.zeros((len_seg_list[-1], 4, 1))
        gaze_list.append(gaze)
    if feats_path_folder is not None:
        feat_att_list = []
        feat_unatt_list = []
        for i in range(len(files)):
            file = files[i]
            len_seg = len_seg_list[i]
            name = file[:-4]
            id_att = name.split('_')[-1]
            ids = set(name.split('_'))
            ids.remove(id_att)
            id_unatt = ids.pop()
            feats_att = get_features(feats_path_folder, id_att, len_seg, smooth=True)
            feats_unatt = get_features(feats_path_folder, id_unatt, len_seg, smooth=True)
            feat_att_list.append(feats_att)
            feat_unatt_list.append(feats_unatt)
    else:
        feat_att_list = None
        feat_unatt_list = None
    return eeg_list, eog_list, feat_att_list, feat_unatt_list, gaze_list, fs, nb_files, len_seg_list


def data_multi_subj(subj_path, fsStim, bads, feats_path_folder, SAVE=True):
    data_path = 'data/'
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    nb_subj = len(subj_path)
    eeg_multisubj_list, eog_multisubj_list, feat_att_list, feat_unatt_list, gaze_multisubj_list, fs, nb_files, len_seg_list = data_per_subj(subj_path[0], fsStim, bads[0], feats_path_folder)
    for n in range(1,nb_subj):
        eeg_list, eog_list, _, _, gaze_list, _, nb_files_sub, _ = data_per_subj(subj_path[n], fsStim, bads[n])
        assert nb_files == nb_files_sub
        eeg_multisubj_list = [np.concatenate((eeg_multisubj_list[i], eeg_list[i]), axis=2) for i in range(nb_files)]
        eog_multisubj_list = [np.concatenate((eog_multisubj_list[i], eog_list[i]), axis=2) for i in range(nb_files)]
        gaze_multisubj_list = [np.concatenate((gaze_multisubj_list[i], gaze_list[i]), axis=2) for i in range(nb_files)]
    if SAVE:
        # save all data (eeg_multisubj_list, eog_multisubj_list, feat_att_list, feat_unatt_list, fs, nb_files) into a single file
        data = {'eeg_multisubj_list': eeg_multisubj_list, 'eog_multisubj_list': eog_multisubj_list, 'feat_att_list': feat_att_list, 'feat_unatt_list': feat_unatt_list, 'gaze_multisubj_list': gaze_multisubj_list, 'fs': fs, 'len_seg_list': len_seg_list}
        file_name = 'data_twoobj.pkl'
        with open(data_path + file_name, 'wb') as f:
            pickle.dump(data, f)
    return eeg_multisubj_list, eog_multisubj_list, feat_att_list, feat_unatt_list, gaze_multisubj_list, fs, len_seg_list


def add_new_data(subj_path, fsStim, bads, feats_path_folder):
    data_path = 'data/'
    file_name = 'data_twoobj.pkl'
    with open(data_path + file_name, 'rb') as f:
        data = pickle.load(f)
    nb_subj_old = data['eeg_multisubj_list'][0].shape[2]
    eeg_multisubj_add, eog_multisubj_add, _, _, gaze_multisubj_add, _, _ = data_multi_subj(subj_path[nb_subj_old:], fsStim, bads[nb_subj_old:], feats_path_folder, SAVE=False)
    eeg_multisubj_list = [np.concatenate((old, new), axis=2) for old, new in zip(data['eeg_multisubj_list'], eeg_multisubj_add)]
    eog_multisubj_list = [np.concatenate((old, new), axis=2) for old, new in zip(data['eog_multisubj_list'], eog_multisubj_add)]
    gaze_multisubj_list = [np.concatenate((old, new), axis=2) for old, new in zip(data['gaze_multisubj_list'], gaze_multisubj_add)]
    data['eeg_multisubj_list'] = eeg_multisubj_list
    data['eog_multisubj_list'] = eog_multisubj_list
    data['gaze_multisubj_list'] = gaze_multisubj_list
    with open(data_path + file_name, 'wb') as f:
        pickle.dump(data, f)
    return eeg_multisubj_list, eog_multisubj_list, data['feat_att_list'], data['feat_unatt_list'], gaze_multisubj_list, data['fs'], data['len_seg_list']


def remove_shot_cuts_and_center(data, fs, time_points=None, remove_time=1):
    T = data.shape[0]
    if time_points is None:
        time_points = [0, T]
    nearby_idx = []
    for p in time_points:
        len_points = int(remove_time*fs)
        nearby_idx = nearby_idx + list(range(max(0, p-len_points), min(p+len_points, T)))
    nearby_idx = list(set(nearby_idx))
    data_clean = np.delete(data, nearby_idx, axis=0)
    data_clean = data_clean - np.mean(data_clean, axis=0)
    return data_clean


def load_data(subj_path, fsStim, bads, feats_path_folder, LOAD_ONLY, ALL_NEW):
    file_name = 'data_twoobj.pkl'
    if LOAD_ONLY:
        data_path = 'data/'
        with open(data_path + file_name, 'rb') as f:
            data = pickle.load(f)
        eeg_multisubj_list = data['eeg_multisubj_list']
        eog_multisubj_list = data['eog_multisubj_list']
        feat_att_list = data['feat_att_list']
        feat_unatt_list = data['feat_unatt_list']
        gaze_multisubj_list = data['gaze_multisubj_list']
        fs = data['fs']
        len_seg_list = data['len_seg_list']
    else:
        if ALL_NEW:
            eeg_multisubj_list, eog_multisubj_list, feat_att_list, feat_unatt_list, gaze_multisubj_list, fs, len_seg_list = data_multi_subj(subj_path, fsStim, bads, feats_path_folder)
        else:
            eeg_multisubj_list, eog_multisubj_list, feat_att_list, feat_unatt_list, gaze_multisubj_list, fs, len_seg_list = add_new_data(subj_path, fsStim, bads, feats_path_folder)
    return eeg_multisubj_list, eog_multisubj_list, feat_att_list, feat_unatt_list, gaze_multisubj_list, fs, len_seg_list


def select_channels(eeg_list, region='parietal_occipital'):
    region_dict = {
        'frontal': ['Fp1', 'AF7', 'AF3', 'F1', 'F3', 'F5', 'F7', 'Fpz', 'Fp2', 'AF8', 'AF4', 'AFz', 'Fz', 'F2', 'F4', 'F6', 'F8'],
        'frontal_central': ['FC5', 'FC3', 'FC1', 'C1', 'C3', 'C5', 'FC6', 'FC4', 'FC2', 'FCz', 'Cz', 'C2', 'C4', 'C6'],
        'temporal': ['FT7', 'T7', 'TP7', 'FT8', 'T8', 'TP8'],
        'central_parietal': ['CP5', 'CP3', 'CP1', 'P1', 'P3', 'P5', 'P7', 'P9', 'Pz', 'CPz', 'CP6', 'CP4', 'CP2', 'P2', 'P4', 'P6', 'P8', 'P10'],
        'parietal_occipital': ['PO7', 'PO3', 'POz', 'PO4', 'PO8', 'O1', 'Oz', 'O2', 'Iz'],
        'selected': ['PO7', 'PO4', 'POz', 'O1']
    }
    montage = mne.channels.make_standard_montage('biosemi64')
    vis_collection = region_dict[region]
    vis_collection_indices = [montage.ch_names.index(ch) for ch in vis_collection]
    subset_list = [eeg[:,vis_collection_indices,:] for eeg in eeg_list]
    return subset_list


def prepare_data_multimod(selected_subj=None, SINGLEOBJ=False, eeg_region=None):
    subjects = ['Pilot_1', 'Pilot_2', 'Pilot_4', 'Pilot_5', 'Pilot_6', 'Pilot_7', 'Pilot_8', 'Pilot_9', 'Pilot_10', 'Pilot_11', 'Pilot_12', 'Pilot_13', 'Pilot_14', 'Pilot_15', 'Pilot_17', 'Pilot_18', 'Pilot_19', 'Pilot_20', 'Pilot_21']
    PATTERN = 'Overlay'
    nb_subj = len(subjects)
    data_path = '../Multi_Object/data/' + PATTERN + '/'
    file_name = 'data_singleobj.pkl' if SINGLEOBJ else 'data_twoobj.pkl'
    with open(data_path + file_name, 'rb') as f:
        data = pickle.load(f)
    eeg_multisubj_list = data['eeg_multisubj_list']
    if eeg_region is not None:
        eeg_multisubj_list = select_channels(eeg_multisubj_list, eeg_region)
    eog_multisubj_list = data['eog_multisubj_list']
    feat_all_att_list = data['feat_att_list']
    feat_all_unatt_list = data['feat_unatt_list'] if not SINGLEOBJ else None
    gaze_multisubj_list = data['gaze_multisubj_list']
    fs = data['fs']

    gaze_velocity_list = [calcu_gaze_velocity(gaze) for gaze in gaze_multisubj_list]
    gaze_coords_list = [gaze[:,0:2,:] for gaze in gaze_multisubj_list]
    saccade_multisubj_list = [np.expand_dims(gaze[:,2,:], axis=1) for gaze in gaze_multisubj_list]
    blink_multisubj_list = [np.expand_dims(gaze[:,3,:], axis=1) for gaze in gaze_multisubj_list]
    saccade_multisubj_list = refine_saccades(saccade_multisubj_list, blink_multisubj_list)
    eog_velocity_list = [calcu_gaze_vel_from_EOG(eog) for eog in eog_multisubj_list]
    gaze_velocity_list = [interpolate_blinks(gaze_velocity, blink) for gaze_velocity, blink in zip(gaze_velocity_list, blink_multisubj_list)]
    gaze_coords_list = [interpolate_blinks(gaze_coords, blink) for gaze_coords, blink in zip(gaze_coords_list, blink_multisubj_list)]
    eog_velocity_list = [interpolate_blinks(eog_velocity, blink) for eog_velocity, blink in zip(eog_velocity_list, blink_multisubj_list)] # blinks are not removed as cleanly as in the gaze data
    mod_list = [eeg_multisubj_list, eog_multisubj_list, gaze_coords_list, gaze_velocity_list, eog_velocity_list, saccade_multisubj_list, feat_all_att_list, feat_all_unatt_list]

    mod_list = [[remove_shot_cuts_and_center(d, fs) for d in sublist] if sublist is not None else None for sublist in mod_list]
    [eeg_multisubj_list, eog_multisubj_list, gaze_coords_list, gaze_velocity_list, eog_velocity_list, saccade_multisubj_list, feat_all_att_list, feat_all_unatt_list] = mod_list
    modal_dict = {'EEG': eeg_multisubj_list, 'EOG': eog_multisubj_list, 'GAZE': gaze_coords_list, 'GAZE_V': gaze_velocity_list
              , 'EOG_V': eog_velocity_list, 'SACC': saccade_multisubj_list}

    if selected_subj is not None:
        Subj_Set = [subjects.index(sub) for sub in selected_subj]
    else: 
        Subj_Set = range(nb_subj)
    
    return modal_dict, feat_all_att_list, feat_all_unatt_list, Subj_Set


def calcu_gaze_velocity(gaze):
    if np.ndim(gaze) == 2:
        gaze = np.expand_dims(gaze, axis=2)
    _, D, _ = gaze.shape
    if D > 2:
        gaze = gaze[:,:2,:]
    pos_diff = np.diff(gaze, axis=0, prepend=np.expand_dims(gaze[0,:], axis=0))
    gaze_velocity = np.sqrt(np.sum(pos_diff**2, axis=1, keepdims=True))
    return gaze_velocity


def calcu_gaze_vel_from_EOG(eog):
    if np.ndim(eog) == 2:
        eog = np.expand_dims(eog, axis=2)
    eog_y = eog[:,0,:] - eog[:,1,:]
    eog_x = eog[:,2,:] - eog[:,3,:]
    eog_xy = np.stack((eog_x, eog_y), axis=1)
    gaze_velocity = calcu_gaze_velocity(eog_xy)
    return gaze_velocity
    

def refine_saccades(saccade_multisubj_list, blink_multisubj_list):
    saccade_multisubj_list = [saccade_multisubj.astype(bool) for saccade_multisubj in saccade_multisubj_list]
    blink_multisubj_list = [blink_multisubj.astype(bool) for blink_multisubj in blink_multisubj_list]
    saccade_multisubj_list = [np.logical_xor(np.logical_and(saccade_multisubj, blink_multisubj), saccade_multisubj) for saccade_multisubj, blink_multisubj in zip(saccade_multisubj_list, blink_multisubj_list)]
    saccade_multisubj_list = [saccade.astype(float) for saccade in saccade_multisubj_list]
    return saccade_multisubj_list


def get_mask_list(Sacc_list, before=15, after=30, ThreeD=False):
    mask_list = []
    for Sacc in Sacc_list:
        T = Sacc.shape[0]
        Sacc = Sacc > 0.5
        idx_surround = np.where(Sacc)[0]
        idx_surround = np.concatenate([np.arange(i-before, i+after+1) for i in idx_surround])
        idx_surround = np.unique(idx_surround)
        idx_surround = idx_surround[(idx_surround>=0) & (idx_surround<T)]
        Sacc[idx_surround] = True
        mask = np.logical_not(Sacc)
        if ThreeD:
            mask = np.expand_dims(mask, axis=2)
        mask_list.append(mask)
    return mask_list


def expand_mask(mask, lag, offset):
    '''
    If the mask will be applied to hankelized data, the mask should be expanded to cover the lags
    '''
    mask_correct_offset = list(np.squeeze(mask))
    mask_correct_offset = (mask_correct_offset + offset*[True])
    mask_exp = copy.deepcopy(mask_correct_offset)
    for i in range(len(mask_correct_offset)):
        if mask_correct_offset[i] == False:
            end = min(i+lag, len(mask_correct_offset))
            mask_exp[i:end] = (end-i)*[False]
    return mask_exp[offset:]


def data_loss_due_to_mask(mask_list, lag, offset):
    '''
    Calculate the percentage of data loss due to the mask
    '''
    mask_exp_list = [expand_mask(mask, lag, offset) for mask in mask_list]
    data_loss = 1 - sum([sum(mask_exp) for mask_exp in mask_exp_list]) / sum([len(mask_exp) for mask_exp in mask_exp_list])
    return data_loss


def remove_saccade(datalist, Sacc, remove_before=15, remove_after=30):
    T = Sacc.shape[0]
    # transform the saccade into a binary mask
    Sacc = Sacc > 0.5
    # find the indices of the points around the saccade
    idx_remove = np.where(Sacc)[0]
    idx_remove = np.concatenate([np.arange(i-remove_before, i+remove_after+1) for i in idx_remove])
    # remove the repeated indices and the indices out of the range
    idx_remove = np.unique(idx_remove)
    idx_remove = idx_remove[(idx_remove>=0) & (idx_remove<T)]
    # remove the time points around the saccade
    datalist = [np.delete(data, idx_remove, axis=0) for data in datalist]
    return datalist


def interpolate_blinks(ts, blinks):
    '''
    Interpolate the time series around the blinks indicated by the binary mask
    '''
    time_series = copy.deepcopy(ts)
    # If the time series is 3D, replicate the blinks along the second dimension
    if time_series.shape[1] != blinks.shape[0]:
        blinks = np.tile(blinks, (1, time_series.shape[1], 1))
    # Transform the blinks into a binary mask
    blinks = blinks > 0.5
    # Set the indices of the blinks to NaN
    time_series[blinks] = np.nan
    if np.ndim(time_series) == 2:
        time_series = clean_features(time_series, smooth=False)
    else:
        for i in range(time_series.shape[2]):
            time_series[:,:,i] = clean_features(time_series[:,:,i], smooth=False)
    return time_series

