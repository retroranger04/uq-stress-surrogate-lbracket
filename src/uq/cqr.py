"""
Conformalized Quantile Regression (CQR) on top of the frozen Phase-1 Deep
Ensemble.

Following Romano, Patterson, Candes (NeurIPS 2019), CQR wraps any base
quantile predictor (q_lo, q_hi) in a split-conformal calibration layer:

    E_i   = max(q_lo(X_i) - Y_i, Y_i - q_hi(X_i))          # conformity score
    Q_hat = Quantile_{1 - alpha} of {E_i} (finite-sample corrected)
    C(X)  = [ q_lo(X) - Q_hat,  q_hi(X) + Q_hat ]          # calibrated interval

Theorem 1 of Romano 2019 gives P(Y in C(X)) >= 1 - alpha for any exchangeable
(calibration, test) pair, distribution-free and finite-sample.

Base quantile predictors here are derived analytically from the Phase-1
ensemble distribution: with per-node mean mu and std s aggregated across the
5 members,

    q_lo(X) = mu - z * s
    q_hi(X) = mu + z * s        with z = Phi^{-1}(1 - alpha/2)

motivated by Gopakumar 2024 (nonconformity scores: CQR, absolute error, std-
dev) and Angelopoulos 2023 Section 2.2 (CQR with arbitrary quantile
estimators). No auxiliary quantile network is trained -- Phase 1 already
established a strong (std vs error) correlation (sample-level Pearson 0.944);
CQR adds the distribution-free calibration layer Phase 1 was missing.

Calibration is "cell-wise" / per-node marginal (Gopakumar 2024): pool the
conformity scores across all (sample, node) pairs on the calibration set and
take one global Q_hat. Coverage is therefore marginal across nodes and
samples; conditional coverage is checked empirically by slicing the test set
on parameter quartiles.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm


def z_for_alpha(alpha: float) -> float:
    """Two-sided Gaussian quantile z s.t. mu +/- z*s covers 1 - alpha."""
    return float(norm.ppf(1.0 - alpha / 2.0))


def base_quantiles(mu: np.ndarray, sigma: np.ndarray, alpha: float
                   ) -> tuple[np.ndarray, np.ndarray]:
    """Gaussian-style base quantile bounds from ensemble mean+std."""
    z = z_for_alpha(alpha)
    return mu - z * sigma, mu + z * sigma


def conformity_scores(y: np.ndarray, q_lo: np.ndarray, q_hi: np.ndarray
                      ) -> np.ndarray:
    """CQR conformity score E = max(q_lo - y, y - q_hi). Positive => y lies
    outside the base interval; negative => interior (slack)."""
    return np.maximum(q_lo - y, y - q_hi)


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """Finite-sample-corrected empirical quantile of the calibration scores.

    Takes the ceil((n+1)(1-alpha))/n empirical quantile, which yields the
    Romano 2019 Theorem 1 guarantee P(Y in C(X)) >= 1 - alpha."""
    n = scores.size
    if n == 0:
        raise ValueError("empty calibration set")
    k = np.ceil((n + 1.0) * (1.0 - alpha)) / n
    k = min(k, 1.0)
    return float(np.quantile(scores, k, method="higher"))


def calibrated_interval(mu: np.ndarray, sigma: np.ndarray, alpha: float,
                        q_hat: float) -> tuple[np.ndarray, np.ndarray]:
    """Apply the conformal widening to the Gaussian base interval."""
    q_lo, q_hi = base_quantiles(mu, sigma, alpha)
    return q_lo - q_hat, q_hi + q_hat


def coverage(y: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> float:
    """Fraction of entries satisfying lo <= y <= hi."""
    return float(((y >= lo) & (y <= hi)).mean())


def width(lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    return hi - lo
