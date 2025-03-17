import numpy as np
import itertools
import utils
from sklearn.covariance import LedoitWolf
from scipy.linalg import eigh, sqrtm
from numpy import linalg as LA


class _BaseModel:
    def __init__(self, latent_dimensions: int = 5):
        self.latent_dimensions = latent_dimensions

    def fit(self, views):
        raise NotImplementedError("Subclasses should implement this method")

    def transform(self, views):
        return [views @ W for views, W in zip(views, self.weights_)]

    def pairwise_correlations(self, views):
        representations = self.transform(views)
        all_corrs = []
        for x, y in itertools.product(representations, repeat=2):
            all_corrs.append(np.diag(utils.cross_corrcoef(x.T, y.T)))
        all_corrs = np.array(all_corrs).reshape(
            (self.n_views_, self.n_views_, self.latent_dimensions)
        )
        return all_corrs

    def average_pairwise_correlations(self, views):
        pair_corrs = self.pairwise_correlations(views)
        # Sum all the pairwise correlations for each dimension, subtract self-correlations, and divide by the number of representations
        dim_corrs = np.sum(pair_corrs, axis=(0, 1)) - pair_corrs.shape[0]
        # Number of pairs is n_views choose 2
        num_pairs = (self.n_views_ * (self.n_views_ - 1)) / 2
        dim_corrs = dim_corrs / (2 * num_pairs)
        return dim_corrs


class LS_LW(_BaseModel):
    def __init__(self, alpha: float=0.0, beta: float=0.0):
        self.latent_dimensions = 1
        self.alpha = alpha
        self.beta = beta
    
    def fit(self, views, Rinit=None, rinit=None):
        data, feats = views
        Rdd = LedoitWolf().fit(data).covariance_
        if Rinit is not None:
            Rdd = self.alpha * Rinit + (1 - self.alpha) * Rdd
        rdf = np.cov(data, feats, rowvar=False)[:data.shape[1], data.shape[1]:]
        if rinit is not None:
            rdf = self.beta * rinit + (1 - self.beta) * rdf
        self.Rdd = Rdd
        self.rdf = rdf
        W = LA.inv(Rdd) @ rdf
        self.weights_ = [W, np.eye(feats.shape[1])]
        self.n_views_ = 2
        return self


class MCCA_LW(_BaseModel):
    def __init__(self, latent_dimensions: int = 5, alpha: float=0.0, beta: float=0.0):
        super().__init__(latent_dimensions)
        self.alpha = alpha
        self.beta = beta

    def fit(self, views, Rinit=None, Dinit=None):
        T, _ = views[0].shape
        dim_list = [data.shape[1] for data in views]
        X = np.concatenate(tuple(views), axis=1)
        Rxx, Dxx = utils.get_cov_mtx(X, dim_list, regularization='lwcov')
        if Rinit is not None:
            Rxx = self.beta * Rinit + (1 - self.beta) * Rxx
            Dxx = self.alpha * Dinit + (1 - self.alpha) * Dxx
        self.Rxx = Rxx
        self.Dxx = Dxx
        lam, W = eigh(Dxx, Rxx, subset_by_index=[0,self.latent_dimensions-1]) # automatically ascend
        Lam = np.diag(lam)
        # Right scaling
        W = W @ sqrtm(LA.inv(Lam.T @ W.T @ Rxx * T @ W @ Lam))
        # Reshape W as (DL*n_components*N)
        self.weights_ = utils.W_organize(np.real(W), views)
        self.n_views_ = len(views)
        return self
    
    def fit_R_2view(self, Rxx, dimv1):
        Dxx = np.zeros_like(Rxx)
        Dxx[:dimv1, :dimv1] = Rxx[:dimv1, :dimv1]
        Dxx[dimv1:, dimv1:] = Rxx[dimv1:, dimv1:]
        self.Rxx = Rxx
        self.Dxx = Dxx
        lam, W = eigh(Dxx, Rxx, subset_by_index=[0,self.latent_dimensions-1]) # automatically ascend
        Lam = np.diag(lam)
        # Right scaling
        W = W @ sqrtm(LA.inv(Lam.T @ W.T @ Rxx @ W @ Lam))
        self.weights_ = [W[:dimv1,:], W[dimv1:,:]]
        self.n_views_ = 2
        return self


class CorrCA_LW(_BaseModel):
    def __init__(self, latent_dimensions: int = 5):
        super().__init__(latent_dimensions)

    def fit(self, views):
        X = np.stack(views, axis=2)
        T, D, N = X.shape
        Rw = np.zeros([D, D])
        for n in range(N):
            Rw += LedoitWolf().fit(X[:,:,n]).covariance_
        Rt = N**2 * LedoitWolf().fit(np.average(X, axis=2)).covariance_
        Rb = (Rt - Rw) / (N - 1)

        ISC, W = eigh(Rb, Rw, subset_by_index=[D - self.latent_dimensions, D - 1])
        ISC = np.real(ISC)
        W = np.real(W)
        ISC = np.squeeze(np.fliplr(np.expand_dims(ISC, axis=0)))
        W = np.fliplr(W)
        # right scaling
        Lam = np.diag(1 / (ISC * (N - 1) + 1))
        W = W @ sqrtm(np.linalg.inv(Lam.T @ W.T @ Rt * T @ W @ Lam))
        self.weights_ = [W] * N
        self.n_views_ = N
        return self

