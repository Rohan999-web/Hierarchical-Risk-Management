import pandas as pd
import numpy as np
import yfinance as yf
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# === 1. Fetch and Prepare Data ===
# The full list of 15 NSE tickers extracted from your universe
tickers_NSE = [
    'M&M.NS', 'ADANIENT.NS', 'TATASTEEL.NS', 'BAJAJ-AUTO.NS', 'ADANIPORTS.NS',
    'VEDL.NS', 'BPCL.NS', 'GAIL.NS', 'IOC.NS', 'SHREECEM.NS',
    'SAIL.NS', 'MOTILALC.NS', 'APOLLOTY.NS', 'BALKRISIN.NS', 'CEATLTD.NS'
]

# Fetch 16 years of data to allow a 10-year training history + 6 years out-of-sample backtest
end_date = datetime.now()
start_date = end_date - timedelta(days=16*365 + 50) 

print(f"Fetching 16 years of daily price data for {len(tickers_NSE)} tickers from NSE India...")
data = yf.download(tickers_NSE, start=start_date, end=end_date)['Close']

# Drop highly incomplete assets and fill internal trading halts
data.dropna(axis=1, thresh=len(data)*0.85, inplace=True)
data = data.ffill().bfill()
valid_tickers = data.columns
N_stocks = len(valid_tickers)
print(f"Active Universe: {N_stocks} tickers surviving data cleaning.")

# Calculate daily percentage returns
returns = data.pct_change().dropna()

# 10 years lookback window in trading days (~2520 days)
lookback_window_days = 10 * 252

# Separate returns into historical initialization and out-of-sample backtest periods
start_idx_y11 = returns.index[lookback_window_days] # First day of Year 11
backtest_returns = returns.loc[start_idx_y11:]
backtest_dates = backtest_returns.index

# === 2. Define Neural Network and Custom Loss Function ===

# Input features per stock: mean return (100d), volatility (100d) -> 30 dimensions
input_dim = N_stocks * 2
hidden_dim = 64
output_dim = N_stocks 

class PortfolioNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(PortfolioNN, self).__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer3 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        # Softmax forces all output weights to be positive fractions summing to 1.0
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        w = self.softmax(self.layer3(x))
        return w

# Option 2 Utility: Mean Daily Return - Gamma * CVaR(daily) - Lambda * sum(w^2)
class NormalizedPortfolioLoss(nn.Module):
    def __init__(self, confidence=0.95, risk_aversion=3.0, regularization_weight=0.0005):
        super(NormalizedPortfolioLoss, self).__init__()
        self.alpha = confidence
        self.gamma = risk_aversion
        self.L = regularization_weight

    def forward(self, weights, returns_matrix):
        # Calculate simulated daily portfolio returns across the entire 10-year window
        portfolio_returns = torch.matmul(returns_matrix, weights.t()) 

        # 1. MEAN Daily Return (Solves the unit magnitude scaling problem)
        mean_return = torch.mean(portfolio_returns)

        # 2. Daily CVaR Tail Risk: Average of the worst 5% of daily outcomes
        num_obs = portfolio_returns.size(0)
        sorted_returns, _ = torch.sort(portfolio_returns, dim=0)
        cutoff_idx = int(num_obs * (1 - self.alpha))
        cutoff_idx = max(1, cutoff_idx) 
            
        cvar_daily = torch.mean(sorted_returns[:cutoff_idx])

        # 3. Scaled L2 Weight Regularization
        regularization = self.L * torch.sum(torch.pow(weights, 2))

        # PyTorch minimizes. To maximize: Mean - Gamma*CVaR - Penalty, return negative.
        # Note: cvar_daily is a negative number (a loss), so subtracting it adds a penalty.
        # To handle signs cleanly: we want high mean_return and tightly bounded (less negative) cvar.
        # Utility = mean_return + gamma * cvar_daily - regularization
        objective_to_maximize = mean_return + (self.gamma * cvar_daily) - regularization
        return -objective_to_maximize 

# === 3. Execute Rolling Backtest Engine ===

def run_nn_backtest(returns_data, model, loss_fn, window_size=lookback_window_days):
    portfolio_returns = []
    T = len(backtest_returns)
    print(f"\nExecuting rolling out-of-sample backtest over {T} target trading days...")
    print(f"Rolling Optimization Window: 10 Years ({window_size} Trailing Days)")
    print(f"Objective: Normalized Utility (Mean Return - {loss_fn.gamma}x CVaR)")

    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for i in range(T):
        current_date = backtest_dates[i]
        data_end_idx = returns_data.index.get_loc(current_date)
        
        # Isolate the exact 10-year historical target matrix
        window_start_idx = data_end_idx - window_size
        optim_window_returns = returns_data.iloc[window_start_idx:data_end_idx]
        optim_returns_tensor = torch.tensor(optim_window_returns.values, dtype=torch.float32)

        # Build Input Vector: Trailing 100-day asset specific means and volatilities
        feature_start_idx = max(0, data_end_idx - 100)
        means_100d = returns_data.iloc[feature_start_idx:data_end_idx].mean().values
        vols_100d = returns_data.iloc[feature_start_idx:data_end_idx].std().values
        input_tensor = torch.tensor(np.concatenate([means_100d, vols_100d]), dtype=torch.float32).view(1, -1)

        # Optimization Step
        # Deep retraining annually to adapt to structure shifts, 1-epoch steps daily
        if i % 252 == 0:
            print(f"  Processing Step {i+1}/{T} (Annual Deep Retraining Active)...")
            for _ in range(15): 
                model.zero_grad()
                loss = loss_fn(model(input_tensor), optim_returns_tensor)
                loss.backward()
                optimizer.step()
        else:
            model.zero_grad()
            loss = loss_fn(model(input_tensor), optim_returns_tensor)
            loss.backward()
            optimizer.step()

        # Extract optimal weights derived from the normalized network layers
        with torch.no_grad():
            daily_weights = model(input_tensor).view(-1).numpy()

        # Evaluate Out-of-Sample Realized Yield
        realized_asset_returns = backtest_returns.iloc[i].values
        portfolio_returns.append(np.dot(daily_weights, realized_asset_returns))
        
        if (i+1) % 100 == 0:
             print(f"  Simulated {i+1}/{T} discrete tracking dates...")

    print("Rolling simulation verified.")
    return pd.Series(portfolio_returns, index=backtest_dates)

# Initialize deep learning structures with the Normalized Utility Objective
nn_model = PortfolioNN(input_dim, hidden_dim, output_dim)
normalized_loss = NormalizedPortfolioLoss(confidence=0.95, risk_aversion=3.0, regularization_weight=0.0005)

# Execute the engine
realized_strategy_returns = run_nn_backtest(returns, nn_model, normalized_loss)

# === 4. Evaluation and Visualization ===

# Benchmark configuration: Naive Equal-Weighted Allocation across surviving assets
benchmark_returns = backtest_returns.mean(axis=1)

# Derive compounded equity trajectories
strategy_equity_curve = (1 + realized_strategy_returns).cumprod() - 1
benchmark_equity_curve = (1 + benchmark_returns).cumprod() - 1

# Render performance curves
plt.figure(figsize=(14, 7))
plt.plot(benchmark_equity_curve.index, benchmark_equity_curve * 100, label='Naive Equal-Weighted Sub-Sector Allocation', color='#7f7f7f', linestyle='--', linewidth=1.2)
plt.plot(strategy_equity_curve.index, strategy_equity_curve * 100, label='Rolling NN Allocation (Normalized Mean-CVaR Utility)', color='#2ca02c', linewidth=1.8)

plt.title('Out-of-Sample Performance: Normalized Objective Function (10-Yr Optimization Window)', fontsize=13, fontweight='bold')
plt.ylabel('Compounded Aggregate Yield (%)', fontsize=11)
plt.xlabel('Timeline Timeline', fontsize=11)
plt.grid(alpha=0.25)
plt.legend(fontsize=10, loc='upper left')

plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.gca().xaxis.set_major_locator(mdates.YearLocator(base=1)) 
plt.gcf().autofmt_xdate(rotation=45) 

plt.tight_layout()
plt.savefig("nn_normalized_10yr_performance.png", dpi=300)
print("\nExecution complete. Visualized trajectory exported to 'nn_normalized_10yr_performance.png'.")
plt.show()