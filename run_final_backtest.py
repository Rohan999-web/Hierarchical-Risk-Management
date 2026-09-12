import pandas as pd

from modules.backtest import calculate_performance_metrics, run_rolling_hrp_backtest
from modules.data_loader import download_yahoo_data
from modules.nn_backtest import run_rolling_nn_backtest
from modules.returns import compute_stock_returns
from modules.stock_selector import select_top_eligible_stocks_by_sector


def main():
    universe = pd.read_csv("final_selection_30_per_sector.csv")
    risk_returns = {}
    nn_returns = {}
    selections = []

    for sector, candidates in universe.groupby("Sector"):
        prices = download_yahoo_data(
            candidates["Ticker"].tolist(),
            years=14.5,
            refresh_short_cache=False,
        )
        selected = select_top_eligible_stocks_by_sector(
            candidates, prices.columns, stocks_per_sector=15
        )
        selections.append(selected)
        log_returns, simple_returns = compute_stock_returns(
            prices[selected["Ticker"].tolist()]
        )
        nn_performance, _ = run_rolling_nn_backtest(simple_returns)
        risk_returns[sector] = log_returns.mean(axis=1)
        nn_returns[sector] = nn_performance["NN_Return"]

    selected_universe = pd.concat(selections, ignore_index=True)
    selected_universe.to_csv("stage3_selected_universe.csv", index=False)

    performance, weights = run_rolling_hrp_backtest(
        pd.DataFrame(risk_returns).dropna(),
        portfolio_returns_data=pd.DataFrame(nn_returns).dropna(),
        window=100,
        rebalance_every=20,
    )
    performance.to_csv("final_hrp_nn_performance.csv")
    weights.to_csv("final_hrp_nn_weights.csv")
    pd.Series(calculate_performance_metrics(performance), name="Value").to_csv(
        "final_hrp_nn_metrics.csv"
    )
    print("Completed final HRP+NN backtest.")


if __name__ == "__main__":
    main()
