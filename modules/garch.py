import numpy as np
import pandas as pd
from scipy.optimize import minimize


# ============================================================
# GARCH(1,1) PARAMETER TRANSFORMATION
# ============================================================

def transform_parameters(params: np.ndarray):

    """
    Convert unconstrained optimizer parameters into
    valid and stable GARCH(1,1) parameters.

    We enforce:

        omega > 0
        alpha >= 0
        beta >= 0
        alpha + beta < 1

    We also keep alpha + beta below 0.995
    to avoid estimates sitting directly on
    the non-stationary boundary.
    """

    p0, p1, p2 = params

    # --------------------------------------------------------
    # Stable softmax
    # --------------------------------------------------------

    logits = np.array(
        [0.0, p1, p2],
        dtype=float
    )

    # Prevent overflow in exp()
    logits = (
        logits
        - np.max(logits)
    )

    exp_logits = np.exp(
        np.clip(
            logits,
            -50,
            50
        )
    )

    probabilities = (
        exp_logits
        / exp_logits.sum()
    )

    # --------------------------------------------------------
    # GARCH parameters
    # --------------------------------------------------------

    # Maximum allowed persistence
    persistence_limit = 0.995

    omega = np.exp(
        np.clip(
            p0,
            -50,
            50
        )
    )

    alpha = (
        persistence_limit
        * probabilities[1]
    )

    beta = (
        persistence_limit
        * probabilities[2]
    )

    return omega, alpha, beta


# ============================================================
# GARCH LOG-LIKELIHOOD
# ============================================================

def garch_log_likelihood(
    params: np.ndarray,
    returns: np.ndarray
) -> float:

    omega, alpha, beta = (
        transform_parameters(params)
    )

    # --------------------------------------------------------
    # Conditional variance
    # --------------------------------------------------------

    sigma2 = np.empty(
        len(returns),
        dtype=float
    )

    # Unconditional variance
    sigma2[0] = (
        omega
        / max(
            1.0 - alpha - beta,
            1e-8
        )
    )

    # GARCH(1,1)
    for t in range(1, len(returns)):

        sigma2[t] = (
            omega
            + alpha
            * returns[t - 1] ** 2
            + beta
            * sigma2[t - 1]
        )

        # Numerical safety
        sigma2[t] = max(
            sigma2[t],
            1e-12
        )

    # --------------------------------------------------------
    # Negative log-likelihood
    # --------------------------------------------------------

    log_likelihood = (
        0.5
        * np.sum(
            np.log(2.0 * np.pi)
            + np.log(sigma2)
            + (
                returns ** 2
            ) / sigma2
        )
    )

    if not np.isfinite(
        log_likelihood
    ):

        return 1e100

    return float(
        log_likelihood
    )


# ============================================================
# FIT ONE GARCH MODEL
# ============================================================

def fit_garch(
    returns: np.ndarray
) -> dict:

    # --------------------------------------------------------
    # Convert to numpy
    # --------------------------------------------------------

    returns = np.asarray(
        returns,
        dtype=float
    )

    # Remove invalid values
    returns = returns[
        np.isfinite(returns)
    ]

    if len(returns) < 50:

        raise ValueError(
            "Not enough observations "
            "for GARCH fitting."
        )

    # --------------------------------------------------------
    # Remove mean
    # --------------------------------------------------------

    r = (
        returns
        - returns.mean()
    )

    # Sample variance
    sample_var = max(
        np.var(r),
        1e-10
    )

    # --------------------------------------------------------
    # Starting points
    # --------------------------------------------------------

    starting_points = [

        [
            np.log(
                sample_var * 0.05
            ),
            0.0,
            2.0
        ],

        [
            np.log(
                sample_var * 0.10
            ),
            0.5,
            1.5
        ],

        [
            np.log(
                sample_var * 0.01
            ),
            1.0,
            2.0
        ],

        [
            np.log(
                sample_var * 0.10
            ),
            -1.0,
            2.0
        ]
    ]

    best_result = None
    best_loss = np.inf

    # --------------------------------------------------------
    # Multiple optimization attempts
    # --------------------------------------------------------

    for x0 in starting_points:

        try:

            result = minimize(
                fun=garch_log_likelihood,
                x0=x0,
                args=(r,),
                method="BFGS",
                options={
                    "maxiter": 1000,
                    "gtol": 1e-6
                }
            )

            # Accept finite results even when
            # scipy reports a convergence warning.
            if np.isfinite(
                result.fun
            ):

                if result.fun < best_loss:

                    best_loss = (
                        result.fun
                    )

                    best_result = result

        except Exception:

            continue

    # --------------------------------------------------------
    # Complete optimization failure
    # --------------------------------------------------------

    if best_result is None:

        print(
            "WARNING: GARCH optimization "
            "failed. Using constant variance."
        )

        return {

            "omega": sample_var,

            "alpha": 0.0,

            "beta": 0.0,

            "sigma2": np.full(
                len(r),
                sample_var
            )
        }

    # --------------------------------------------------------
    # Extract parameters
    # --------------------------------------------------------

    omega, alpha, beta = (
        transform_parameters(
            best_result.x
        )
    )

    # --------------------------------------------------------
    # Conditional variance
    # --------------------------------------------------------

    sigma2 = np.empty(
        len(r),
        dtype=float
    )

    sigma2[0] = (
        omega
        / max(
            1.0 - alpha - beta,
            1e-8
        )
    )

    for t in range(
        1,
        len(r)
    ):

        sigma2[t] = (
            omega
            + alpha
            * r[t - 1] ** 2
            + beta
            * sigma2[t - 1]
        )

        sigma2[t] = max(
            sigma2[t],
            1e-12
        )

    # --------------------------------------------------------
    # Return model
    # --------------------------------------------------------

    return {

        "omega": omega,

        "alpha": alpha,

        "beta": beta,

        "sigma2": sigma2
    }


# ============================================================
# FIT GARCH FOR ALL SECTORS
# ============================================================

def fit_all_garch(
    sector_returns: pd.DataFrame
) -> dict:

    results = {}

    print(
        "\nFitting GARCH(1,1) models..."
    )

    for sector in (
        sector_returns.columns
    ):

        results[sector] = (
            fit_garch(
                sector_returns[
                    sector
                ].values
            )
        )

        alpha = (
            results[sector][
                "alpha"
            ]
        )

        beta = (
            results[sector][
                "beta"
            ]
        )

        print(
            f"{sector}: "
            f"alpha={alpha:.4f}, "
            f"beta={beta:.4f}, "
            f"alpha+beta="
            f"{alpha + beta:.4f}"
        )

    return results


# ============================================================
# STANDARDIZED RESIDUALS
# ============================================================

def extract_standardized_residuals(
    sector_returns: pd.DataFrame,
    garch_results: dict
) -> pd.DataFrame:

    standardized = {}

    for sector in (
        sector_returns.columns
    ):

        returns = (
            sector_returns[
                sector
            ].values
        )

        volatility = np.sqrt(
            garch_results[
                sector
            ]["sigma2"]
        )

        standardized[
            sector
        ] = (

            (
                returns
                - returns.mean()
            )

            /

            np.maximum(
                volatility,
                1e-10
            )
        )

    return pd.DataFrame(
        standardized,
        index=sector_returns.index
    )