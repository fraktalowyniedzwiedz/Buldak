import os
import numpy as np
from .base import RegressionDataset


class PrescribedConditionDataset(RegressionDataset):
    """
    for augmented normal matrix M_bar = Z^T Z / A, where Z = [X | 1]

    X = U S V^T where
    U in R^{n x d} has orthonormal columns and U^T 1 = 0, so the
    columns of X are exactly centred and Z^T Z / n is exactly block
    diagonal
    V in R^{d x d} has last column 1/sqrt(d), so all column variances
    of X equal (d-1)/d + 1/(d kappa), so any per-column
    standardisation is a scalar multiple of the identity and so also
    kappa-preserving
    S = diag(sqrt(n), ..., sqrt(n), sqrt(n/kappa)) sets the spectrum
    of X^T X / n to {1, ..., 1, 1/kappa}

    So

    kappa_2(X^T X / n) = kappa
    kappa_2(Z^T Z / n) = kappa
    """

    def __init__(self, n_samples=100, n_features=5, kappa_target=10.0,
                 noise=0.1, random_state=42, **kwargs):
        env_kappa = os.environ.get('PRESCRIBED_KAPPA')
        if env_kappa is not None:
            kappa_target = float(env_kappa)
        self.n_samples = n_samples
        self.n_features = n_features
        self.kappa_target = float(kappa_target)
        self.noise = noise
        self.random_state = random_state
        super().__init__(**kwargs)

    def _build_V(self, d, rng):
        """Return V in R^{d x d} with last column 1/sqrt(d)."""
        e_last = np.ones(d) / np.sqrt(d)
        # orthonormal basis of e_last^perp: SVD of the orthogonal projector
        B = np.eye(d) - np.outer(e_last, e_last)
        Ub, _, _ = np.linalg.svd(B)
        V_perp = Ub[:, :d - 1] # d x (d-1), orthonormal
        # fix signs deterministically (SVD is sign-ambiguous)
        signs = np.sign(V_perp[0, :])
        signs[signs == 0] = 1.0
        V_perp = V_perp * signs
        V = np.column_stack([V_perp, e_last])  # d x d, orthogonal
        return V

    def _build_U(self, n, d, rng):
        """Return U in R^{n x d} orthonormal with U^T 1 = 0"""
        A = rng.standard_normal((n, d))
        A -= A.mean(axis=0, keepdims=True) # enforce 1^T A = 0
        U, _ = np.linalg.qr(A)
        # numerical cleanup, project out 1 and reortho
        U -= np.outer(np.ones(n), np.ones(n) @ U) / n
        U, _ = np.linalg.qr(U)
        return U

    def load(self):
        rng = np.random.default_rng(self.random_state)
        n, d = self.n_samples, self.n_features
        kappa = max(float(self.kappa_target), 1.0)

        V = self._build_V(d, rng)
        U = self._build_U(n, d, rng)

        s = np.full(d, np.sqrt(n))
        s[-1] = np.sqrt(n / kappa)

        X = (U * s) @ V.T

        theta_true = rng.standard_normal(d)
        y = X @ theta_true + rng.normal(0.0, self.noise, size=n)
        return X, y

    def dataset_name(self):
        return (f"PrescribedCond_N{self.n_samples}"
                f"_D{self.n_features}"
                f"_kappa{self.kappa_target:g}")