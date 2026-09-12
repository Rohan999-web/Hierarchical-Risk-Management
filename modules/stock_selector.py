import pandas as pd


def select_top_stocks_by_sector(
    universe,
    stocks_per_sector=15
):
    """
    Select the top stocks within each sector based on
    Stage 1 Composite Score.

    Parameters
    ----------
    universe : pd.DataFrame
        Stage 1 stock universe containing:
        Ticker, Sector, Composite Score

    stocks_per_sector : int
        Number of stocks to retain per sector.

    Returns
    -------
    pd.DataFrame
        Selected stocks, with exactly `stocks_per_sector`
        stocks per sector.
    """

    required_columns = {
        "Ticker",
        "Sector",
        "Composite Score"
    }

    missing = required_columns - set(universe.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    if stocks_per_sector < 1:
        raise ValueError(
            "stocks_per_sector must be at least 1."
        )

    selected = (
        universe
        .sort_values(
            ["Sector", "Composite Score"],
            ascending=[True, False]
        )
        .groupby("Sector", group_keys=False)
        .head(stocks_per_sector)
        .copy()
    )

    counts = selected["Sector"].value_counts()

    invalid = counts[counts != stocks_per_sector]

    if len(invalid) > 0:
        raise ValueError(
            "Some sectors do not contain enough stocks "
            f"for selection: {invalid.to_dict()}"
        )

    if selected["Ticker"].nunique() != len(selected):
        raise ValueError(
            "Duplicate tickers found after stock selection."
        )

    return selected.reset_index(drop=True)


def select_top_eligible_stocks_by_sector(
    universe,
    eligible_tickers,
    stocks_per_sector=15
):
    """Select the highest-ranked eligible stocks in every sector.

    ``eligible_tickers`` must represent stocks that passed the historical-data
    check for the requested modelling period.  The ranking is still determined
    solely by ``Composite Score``; eligibility only prevents a stock with an
    insufficient price history from silently changing the NN input dimension.
    """

    if len(eligible_tickers) == 0:
        raise ValueError("eligible_tickers must contain at least one ticker.")

    eligible = set(eligible_tickers)

    ranked = universe[
        universe["Ticker"].isin(eligible)
    ].copy()

    selected = select_top_stocks_by_sector(
        ranked,
        stocks_per_sector=stocks_per_sector
    )

    selected_sectors = set(selected["Sector"])
    missing_sectors = set(universe["Sector"]) - selected_sectors

    if missing_sectors:
        raise ValueError(
            "No eligible stocks found for sectors: "
            f"{sorted(missing_sectors)}"
        )

    return selected
