import numpy as np
import pandas as pd

from modules.garch import (
    fit_all_garch,
    extract_standardized_residuals
)

from modules.dcc import fit_dcc

from modules.hrp import (
    correlation_to_distance,
    run_hierarchical_clustering,
    get_leaf_ordering,
    build_covariance_matrix,
    calculate_hrp_weights
)


def calculate_hrp_weights_for_window(
    sector_returns: pd.DataFrame
) -> np.ndarray:

    """
    Calculate HRP sector weights for one historical window.
    """

    # 1. GARCH
    garch_results = fit_all_garch(
        sector_returns
    )

    # 2. Standardized residuals
    standardized_residuals = (
        extract_standardized_residuals(
            sector_returns,
            garch_results
        )
    )

    # 3. DCC
    dcc_results = fit_dcc(
        standardized_residuals
    )

    correlation = dcc_results["R_today"]

    # 4. Correlation -> distance
    distance = correlation_to_distance(
        correlation
    )

    # 5. Hierarchical clustering
    linkage_matrix = (
        run_hierarchical_clustering(
            distance
        )
    )

    ordering = get_leaf_ordering(
        linkage_matrix
    )

    # 6. Covariance matrix
    covariance = build_covariance_matrix(
        correlation,
        garch_results,
        list(sector_returns.columns)
    )

    # 7. HRP weights
    weights = calculate_hrp_weights(
        covariance,
        ordering
    )

    return np.asarray(
        weights,
        dtype=float
    )


def run_rolling_hrp_backtest(
    sector_returns: pd.DataFrame,
    window: int = 100,
    rebalance_every: int = 20
) -> tuple:

    """
    Rolling HRP backtest with periodic rebalancing.

    Parameters
    ----------
    sector_returns:
        Daily sector log returns.

    window:
        Number of historical trading days used
        to calculate HRP.

    rebalance_every:
        Number of trading days between HRP
        recalculations.

    Example:

        window = 100
        rebalance_every = 20

        means:

        Use previous 100 days
              ↓
        calculate HRP
              ↓
        hold weights for 20 days
              ↓
        recalculate
              ↓
        repeat
    """

    if len(sector_returns) <= window:

        raise ValueError(
            "Not enough data for rolling backtest."
        )

    if rebalance_every < 1:

        raise ValueError(
            "rebalance_every must be >= 1."
        )

    # Remove missing observations
    sector_returns = (
        sector_returns
        .dropna()
    )

    sectors = list(
        sector_returns.columns
    )

    n_sectors = len(sectors)
    n_days = len(sector_returns)

    # Equal-weight benchmark
    equal_weights = (
        np.ones(n_sectors)
        / n_sectors
    )

    # Results
    portfolio_returns = []
    benchmark_returns = []
    weight_history = []
    dates = []

    current_weights = equal_weights.copy()

    print(
        "\nStarting rolling HRP backtest..."
    )

    print(
        f"Historical window: {window} days"
    )

    print(
        f"Rebalance frequency: "
        f"every {rebalance_every} days"
    )

    total_oos_days = n_days - window

    estimated_rebalances = int(
        np.ceil(
            total_oos_days
            / rebalance_every
        )
    )

    print(
        f"Out-of-sample days: "
        f"{total_oos_days}"
    )

    print(
        f"Expected HRP calculations: "
        f"~{estimated_rebalances}"
    )

    # ---------------------------------------------------------
    # Rolling backtest
    # ---------------------------------------------------------

    for i in range(
        window,
        n_days
    ):

        # Recalculate HRP only at rebalance dates
        rebalance = (
            (i - window) % rebalance_every == 0
        )

        if rebalance:

            training_data = (
                sector_returns.iloc[
                    i - window:i
                ]
            )

            print(
                f"\nRebalancing on "
                f"{sector_returns.index[i]}"
            )

            try:

                current_weights = (
                    calculate_hrp_weights_for_window(
                        training_data
                    )
                )

                current_weights = np.asarray(
                    current_weights,
                    dtype=float
                )

                # Remove invalid values
                current_weights = (
                    np.nan_to_num(
                        current_weights,
                        nan=0.0,
                        posinf=0.0,
                        neginf=0.0
                    )
                )

                weight_sum = (
                    current_weights.sum()
                )

                # Normalize
                if weight_sum > 0:

                    current_weights = (
                        current_weights
                        / weight_sum
                    )

                else:

                    current_weights = (
                        equal_weights.copy()
                    )

                print(
                    "HRP Weights: "
                    + ", ".join(
                        f"{sector}="
                        f"{weight * 100:.2f}%"
                        for sector, weight
                        in zip(
                            sectors,
                            current_weights
                        )
                    )
                )

            except Exception as e:

                print(
                    f"WARNING: HRP failed: {e}"
                )

                print(
                    "Using equal weights."
                )

                current_weights = (
                    equal_weights.copy()
                )

        # -----------------------------------------------------
        # Apply current weights to today's return
        # -----------------------------------------------------

        today_returns = (
            sector_returns.iloc[i].values
        )

        portfolio_return = float(
            np.dot(
                current_weights,
                today_returns
            )
        )

        benchmark_return = float(
            np.dot(
                equal_weights,
                today_returns
            )
        )

        # -----------------------------------------------------
        # Store
        # -----------------------------------------------------

        portfolio_returns.append(
            portfolio_return
        )

        benchmark_returns.append(
            benchmark_return
        )

        weight_history.append(
            current_weights.copy()
        )

        dates.append(
            sector_returns.index[i]
        )

    # ---------------------------------------------------------
    # Convert to Series/DataFrames
    # ---------------------------------------------------------

    portfolio_returns = pd.Series(
        portfolio_returns,
        index=dates,
        name="HRP"
    )

    benchmark_returns = pd.Series(
        benchmark_returns,
        index=dates,
        name="EqualWeight"
    )

    weights_df = pd.DataFrame(
        weight_history,
        index=dates,
        columns=sectors
    )

    # ---------------------------------------------------------
    # Cumulative log-return performance
    # ---------------------------------------------------------

    portfolio_cumulative = (
        np.exp(
            portfolio_returns.cumsum()
        ) - 1
    )

    benchmark_cumulative = (
        np.exp(
            benchmark_returns.cumsum()
        ) - 1
    )

    performance_df = pd.DataFrame(
        {
            "HRP_Return":
                portfolio_returns,

            "Benchmark_Return":
                benchmark_returns,

            "HRP_Cumulative":
                portfolio_cumulative,

            "Benchmark_Cumulative":
                benchmark_cumulative
        }
    )

    return (
        performance_df,
        weights_df
    )


def calculate_performance_metrics(
    performance_df: pd.DataFrame
) -> dict:

    """
    Calculate performance statistics.
    """

    hrp_returns = performance_df[
        "HRP_Return"
    ]

    benchmark_returns = performance_df[
        "Benchmark_Return"
    ]

    # ---------------------------------------------------------
    # Cumulative return
    # ---------------------------------------------------------

    hrp_cumulative = (
        np.exp(
            hrp_returns.sum()
        ) - 1
    )

    benchmark_cumulative = (
        np.exp(
            benchmark_returns.sum()
        ) - 1
    )

    # ---------------------------------------------------------
    # Annualized return
    # ---------------------------------------------------------

    n_days = len(
        performance_df
    )

    years = n_days / 252.0

    hrp_annualized = (
        (1 + hrp_cumulative)
        ** (1 / years)
        - 1
    )

    benchmark_annualized = (
        (1 + benchmark_cumulative)
        ** (1 / years)
        - 1
    )

    # ---------------------------------------------------------
    # Annualized volatility
    # ---------------------------------------------------------

    hrp_volatility = (
        hrp_returns.std()
        * np.sqrt(252)
    )

    benchmark_volatility = (
        benchmark_returns.std()
        * np.sqrt(252)
    )

    # ---------------------------------------------------------
    # Sharpe ratio
    # ---------------------------------------------------------

    if hrp_returns.std() > 0:

        hrp_sharpe = (
            hrp_returns.mean()
            / hrp_returns.std()
            * np.sqrt(252)
        )

    else:

        hrp_sharpe = 0.0

    if benchmark_returns.std() > 0:

        benchmark_sharpe = (
            benchmark_returns.mean()
            / benchmark_returns.std()
            * np.sqrt(252)
        )

    else:

        benchmark_sharpe = 0.0

    # ---------------------------------------------------------
    # Equity curves
    # ---------------------------------------------------------

    hrp_equity = np.exp(
        hrp_returns.cumsum()
    )

    benchmark_equity = np.exp(
        benchmark_returns.cumsum()
    )

    # ---------------------------------------------------------
    # Drawdowns
    # ---------------------------------------------------------

    hrp_peak = (
        hrp_equity
        .cummax()
    )

    benchmark_peak = (
        benchmark_equity
        .cummax()
    )

    hrp_drawdown = (
        hrp_equity
        / hrp_peak
        - 1
    )

    benchmark_drawdown = (
        benchmark_equity
        / benchmark_peak
        - 1
    )

    hrp_max_drawdown = (
        hrp_drawdown.min()
    )

    benchmark_max_drawdown = (
        benchmark_drawdown.min()
    )

    return {

        "HRP Cumulative Return":
            hrp_cumulative,

        "Benchmark Cumulative Return":
            benchmark_cumulative,

        "HRP Annualized Return":
            hrp_annualized,

        "Benchmark Annualized Return":
            benchmark_annualized,

        "HRP Annualized Volatility":
            hrp_volatility,

        "Benchmark Annualized Volatility":
            benchmark_volatility,

        "HRP Sharpe Ratio":
            hrp_sharpe,

        "Benchmark Sharpe Ratio":
            benchmark_sharpe,

        "HRP Maximum Drawdown":
            hrp_max_drawdown,

        "Benchmark Maximum Drawdown":
            benchmark_max_drawdown
    }