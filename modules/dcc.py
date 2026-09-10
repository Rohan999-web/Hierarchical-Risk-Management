import numpy as np
import pandas as pd
from scipy.optimize import minimize


def dcc_log_likelihood(
    params: np.ndarray,
    Z: np.ndarray,
    Q_bar: np.ndarray
) -> float:

    p0, p1 = params

    # Transform parameters so a and b stay positive
    # and a + b remains below 1.
    denom = 1.0 + np.exp(p0) + np.exp(p1)

    a = np.exp(p0) / denom
    b = np.exp(p1) / denom

    if a + b >= 0.9999:
        return 1e10

    T, N = Z.shape

    Q = Q_bar.copy()
    total_loss = 0.0

    for t in range(T):

        z = Z[t]

        # DCC update
        Q = (
            (1.0 - a - b) * Q_bar
            + a * np.outer(z, z)
            + b * Q
        )

        # Convert Q into correlation matrix R
        diagonal = np.maximum(
            np.sqrt(np.diag(Q)),
            1e-10
        )

        D_inv = np.diag(1.0 / diagonal)

        R = D_inv @ Q @ D_inv

        np.fill_diagonal(R, 1.0)

        try:
            sign, log_det = np.linalg.slogdet(R)

            if sign <= 0:
                return 1e10

            total_loss += (
                log_det
                + z @ np.linalg.inv(R) @ z
                - z @ z
            )

        except Exception:
            return 1e10

    return 0.5 * total_loss


def fit_dcc(
    standardized_residuals: pd.DataFrame
) -> dict:

    Z = standardized_residuals.values

    # Long-run covariance/correlation estimate
    Q_bar = np.cov(Z.T)

    # Numerical stability
    if np.linalg.eigvalsh(Q_bar).min() < 1e-8:
        Q_bar += 1e-6 * np.eye(Q_bar.shape[0])

    # Initial parameters
    x0 = [
        np.log(0.05 / 0.05),
        np.log(0.90 / 0.05)
    ]

    result = minimize(
        fun=dcc_log_likelihood,
        x0=x0,
        args=(Z, Q_bar),
        method="Nelder-Mead"
    )

    p0, p1 = result.x

    denom = (
        1.0
        + np.exp(p0)
        + np.exp(p1)
    )

    a = np.exp(p0) / denom
    b = np.exp(p1) / denom

    # Reconstruct Q up to the latest observation
    Q = Q_bar.copy()

    for t in range(len(Z)):

        z = Z[t]

        Q = (
            (1.0 - a - b) * Q_bar
            + a * np.outer(z, z)
            + b * Q
        )

    # Convert final Q into today's correlation matrix
    diagonal = np.maximum(
        np.sqrt(np.diag(Q)),
        1e-10
    )

    D_inv = np.diag(1.0 / diagonal)

    R_today = D_inv @ Q @ D_inv

    np.fill_diagonal(R_today, 1.0)

    R_today = np.clip(
        R_today,
        -0.9999,
        0.9999
    )

    return {
        "a": a,
        "b": b,
        "R_today": R_today
    }


def correlation_matrix_to_dataframe(
    correlation_matrix: np.ndarray,
    sectors: list
) -> pd.DataFrame:

    return pd.DataFrame(
        correlation_matrix,
        index=sectors,
        columns=sectors
    )