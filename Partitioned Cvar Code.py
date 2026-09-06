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
tickers_NSE = [
    'M&M.NS', 'ADANIENT.NS', 'TATASTEEL.NS', 'BAJAJ-AUTO.NS', 'ADANIPORTS.NS',
    'VEDL.NS', 'BPCL.NS', 'GAIL.NS', 'IOC.NS', 'SHREECEM.NS',
    'SAIL.NS', 'MOTILALC.NS', 'APOLLOTY.NS', 'BALKRISIN.NS', 'CEATLTD.NS'
]

# Fetch 16 years of data to support a 10-year training lookback + 6 years out-of-sample backtest
end_date = datetime.now()
start_date = end_date - timedelta(days=16*365 + 50) 

print(f"Fetching 16 years of daily price data for {len(tickers_NSE)} targeted equities...")
data = yf.download(tickers_NSE, start=start_date, end=end_date)['Close']

# Drop highly incomplete assets and fill internal trading halts
data.dropna(axis=1, thresh=len(data)*0.85, inplace=True)
data = data.ffill().bfill()
valid_tickers = data.columns
N_stocks = len(valid_tickers)
print(f"Active Universe locked: {N_stocks} targets surviving data cleaning.")

# Calculate daily simple percentage returns
returns = data.pct_change().dropna()
lookback_window_days = 10 * 252

# Separate returns into initialization history and out-of-sample simulation boundaries
start_idx_y11 = returns.index[lookback_window_days] 
backtest_returns = returns.loc[start_idx_y11:]
backtest_dates = backtest_returns.index

# === 2. Define Neural Network Architecture ===
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
        # Softmax strictly bounds output predictions to positive fractions summing to 1.0
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        w = self.softmax(self.layer3(x))
        return w

# === 3. Custom Loss Module: CVaR + L2 + Extreme Transaction Friction ===
class HighFrictionLoss(nn.Module):
    # CRITICAL: cost_rate=0.0050 enforces your explicit 0.5% transaction cost penalty.
    def __init__(self, confidence=0.95, risk_aversion=3.0, regularization_weight=0.0005, cost_rate=0.0050, partitions=5, scale_factor=1000.0):
        super(HighFrictionLoss, self).__init__()
        self.alpha = confidence
        self.gamma = risk_aversion
        self.L = regularization_weight
        self.c = cost_rate # Execution friction rate (0.0050 = 50 bps)
        self.num_partitions = partitions
        self.scale = scale_factor 

    def forward(self, weights, returns_matrix, prev_weights=None):
        # 1. Portfolio Mean Return
        portfolio_returns = torch.matmul(returns_matrix, weights.t()) 
        mean_return = torch.mean(portfolio_returns)

        # 2. Partitioned Average CVaR
        total_obs = portfolio_returns.size(0)
        partition_size = total_obs // self.num_partitions
        clean_returns = portfolio_returns[: self.num_partitions * partition_size]
        
        reshaped_returns = clean_returns.view(self.num_partitions, partition_size)
        sorted_returns, _ = torch.sort(reshaped_returns, dim=1)

        cutoff_idx = int(partition_size * (1 - self.alpha))
        cutoff_idx = max(1, cutoff_idx) 
        
        partition_cvars = torch.mean(sorted_returns[:, :cutoff_idx], dim=1)
        cvar_avg = torch.mean(partition_cvars)

        # 3. L2 Weight Regularization
        regularization = self.L * torch.sum(torch.pow(weights, 2))

        # 4. TRANSACTION COST PENALTY (L1 norm of absolute turnover)
        if prev_weights is not None:
            turnover = torch.sum(torch.abs(weights - prev_weights))
            transaction_friction = self.c * turnover
        else:
            transaction_friction = torch.tensor(0.0, device=weights.device)

        # Maximize: Mean + Gamma*|CVaR| - L2 Penalty - Transaction Friction
        objective_to_maximize = mean_return + (self.gamma * cvar_avg) - regularization - transaction_friction
        
        return -objective_to_maximize * self.scale 

# === 4. Execute High-Friction Rolling Engine ===
def run_nn_backtest(returns_data, model, loss_fn, window_size=lookback_window_days):
    portfolio_returns = []
    T = len(backtest_returns)
    print(f"\nExecuting high-friction rolling backtest over {T} trading days...")
    print(f"Friction Rate Configured: {loss_fn.c * 100:.2f}% (50 bps) per absolute weight shift")

    optimizer = optim.Adam(model.parameters(), lr=0.01)
    locked_prev_weights = None

    for i in range(T):
        current_date = backtest_dates[i]
        data_end_idx = returns_data.index.get_loc(current_date)
        
        window_start_idx = data_end_idx - window_size
        optim_window_returns = returns_data.iloc[window_start_idx:data_end_idx]
        optim_returns_tensor = torch.tensor(optim_window_returns.values, dtype=torch.float32)

        feature_start_idx = max(0, data_end_idx - 100)
        means_100d = returns_data.iloc[feature_start_idx:data_end_idx].mean().values
        vols_100d = returns_data.iloc[feature_start_idx:data_end_idx].std().values
        
        scaled_features = np.concatenate([means_100d, vols_100d]) * 100.0
        input_tensor = torch.tensor(scaled_features, dtype=torch.float32).view(1, -1)

        # Determine target execution convergence schedule
        if i == 0:
            epochs = 300
        elif i % 252 == 0:
            epochs = 50
        else:
            epochs = 5

        # Execute optimization loop passing locked prior state
        for _ in range(epochs):
            model.zero_grad()
            weights_pred = model(input_tensor)
            loss = loss_fn(weights_pred, optim_returns_tensor, prev_weights=locked_prev_weights)
            loss.backward()
            optimizer.step()

        # Extract optimized target weights for the current step
        with torch.no_grad():
            current_weights_tensor = model(input_tensor)
            daily_weights = current_weights_tensor.view(-1).numpy()
            locked_prev_weights = current_weights_tensor.clone().detach()

        # Calculate live OUT-OF-SAMPLE Net Returns
        realized_asset_returns = backtest_returns.iloc[i].values
        gross_portfolio_return = np.dot(daily_weights, realized_asset_returns)
        
        # Deduct massive 0.5% friction drag from the backtest ledger
        if i > 0:
            actual_turnover = np.sum(np.abs(daily_weights - locked_prev_weights.view(-1).numpy()))
            actual_friction = loss_fn.c * actual_turnover
        else:
            actual_friction = 0.0
            
        net_portfolio_return = gross_portfolio_return - actual_friction
        portfolio_returns.append(net_portfolio_return)
        
        if (i+1) % 100 == 0:
             print(f"  Processed {i+1}/{T} high-friction tracking steps...")

    return pd.Series(portfolio_returns, index=backtest_dates)

# Initialize structures targeting the 0.5% transaction cost target
nn_model = PortfolioNN(input_dim, hidden_dim, output_dim)
high_friction_loss = HighFrictionLoss(confidence=0.95, risk_aversion=3.0, regularization_weight=0.0005, cost_rate=0.0050, partitions=5, scale_factor=1000.0)

realized_strategy_returns = run_nn_backtest(returns, nn_model, high_friction_loss)

# === 5. Output Visualization Compilation ===
benchmark_returns = backtest_returns.mean(axis=1)

strategy_equity = (1 + realized_strategy_returns).cumprod() - 1
benchmark_equity = (1 + benchmark_returns).cumprod() - 1

plt.figure(figsize=(14, 7))
plt.plot(benchmark_equity.index, benchmark_equity * 100, label='Naive Equal-Weighted Sub-Sector Baseline', color='#7f7f7f', linestyle='--', linewidth=1.2)
plt.plot(strategy_equity.index, strategy_equity * 100, label='High-Friction NN Allocation (Net Realized Yield at c=0.5%)', color='#d62728', linewidth=1.8)

plt.title('Out-of-Sample Performance: High Transaction Friction Objective (c = 0.5%)', fontsize=13, fontweight='bold')
plt.ylabel('Compounded Net Yield (%)', fontsize=11)
plt.xlabel('Timeline Timeline', fontsize=11)
plt.grid(alpha=0.25)
plt.legend(fontsize=10, loc='upper left')

plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.gca().xaxis.set_major_locator(mdates.YearLocator(base=1)) 
plt.gcf().autofmt_xdate(rotation=45) 

plt.tight_layout()
output_filename = "nn_high_friction_0.5pct_cvar.png"
plt.savefig(output_filename, dpi=300)
print(f"\nExecution complete. High-friction equity curve exported cleanly to '{output_filename}'.")
plt.show()