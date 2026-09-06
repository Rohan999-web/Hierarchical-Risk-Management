import warnings
import os
import sys
import time
from datetime import datetime, timedelta

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import torch
import torch.nn as nn
import torch.optim as optim

# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 0 ── UNIVERSE SETUP & CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

UNIVERSE_FILE = "sectorwise_ranked_stocks.csv"
HRP_WINDOW_SIZE = 100
NN_LOOKBACK_YEARS = 10
HRP_TEST_YEARS = 4

def load_universe_mapping(filepath: str) -> dict:
    print(f"Reading target universe matrix from: '{filepath}'...")
    if filepath.endswith('.csv'):
        df = pd.read_csv(filepath)
    elif filepath.endswith(('.xls', '.xlsx')):
        df = pd.read_excel(filepath)
    else:
        raise ValueError("Target mapping must be formatted as a standard .csv or .xlsx matrix.")

    df.columns = df.columns.str.strip().str.title()
    if 'Ticker' not in df.columns or 'Sector' not in df.columns:
        raise KeyError("Source configuration file requires standard 'Ticker' and 'Sector' columns.")

    universe = {}
    for sector, group in df.groupby('Sector'):
        tickers = group['Ticker'].str.strip().tolist()
        universe[sector] = tickers
        
    total_stocks = sum(len(t) for t in universe.values())
    print(f"Successfully loaded {total_stocks} assets across {len(universe)} active sectors.")
    return universe

try:
    SECTOR_MAPPING = load_universe_mapping(UNIVERSE_FILE)
    SECTORS = list(SECTOR_MAPPING.keys())
    ALL_TICKERS = [t for sublist in SECTOR_MAPPING.values() for t in sublist]
except Exception as e:
    print(f"\nSystem Initialization Halted: {e}")
    raise e

# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 1 ── PRICING EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────

def download_yahoo_data(tickers: list, required_years: float = 14.5) -> pd.DataFrame:
    try:
        import yfinance as yf
    except ImportError:
        raise ImportError("Execution prerequisite missing. Execute: pip install yfinance")

    end_date = datetime.today()
    start_date = end_date - timedelta(days=int(required_years * 365.25) + 60)

    print(f"\nExtracting historical arrays for {len(tickers)} targets...")
    df_list = []
    failed_tickers = []
    
    for i, ticker in enumerate(tickers, 1):
        try:
            data = yf.download(ticker, start=start_date, end=end_date, progress=False)
            if data.empty or 'Close' not in data.columns:
                failed_tickers.append(ticker)
                continue
                
            close_series = data['Close'].copy()
            if isinstance(close_series, pd.DataFrame):
                close_series = close_series.iloc[:, 0]
                
            close_series.name = ticker
            df_list.append(close_series)
            
        except Exception:
            failed_tickers.append(ticker)

    wide_df = pd.concat(df_list, axis=1).sort_index()
    threshold = int(len(wide_df) * 0.85)
    valid_cols = wide_df.dropna(thresh=threshold, axis=1).columns
    wide_df = wide_df[valid_cols].ffill().bfill()
    return wide_df

# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 2 ── NEURAL NETWORK SUB-LAYER (STAGE 1)
# ─────────────────────────────────────────────────────────────────────────────

class IntraSectorNN(nn.Module):
    # Added LeakyReLU and Temperature scaling to prevent sub-layer weight freezing
    def __init__(self, input_dim, hidden_dim, output_dim, temperature=0.5):
        super(IntraSectorNN, self).__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer3 = nn.Linear(hidden_dim, output_dim)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.01)
        self.temperature = temperature
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.leaky_relu(self.layer1(x))
        x = self.leaky_relu(self.layer2(x))
        logits = self.layer3(x)
        w = self.softmax(logits / self.temperature)
        return w

class FrictionAdjustedPartitionedLoss(nn.Module):
    def __init__(self, confidence=0.95, risk_aversion=3.0, regularization_weight=0.0005, cost_rate=0.0050, partitions=5, scale_factor=1000.0):
        super(FrictionAdjustedPartitionedLoss, self).__init__()
        self.alpha = confidence
        self.gamma = risk_aversion
        self.L = regularization_weight
        self.c = cost_rate
        self.num_partitions = partitions
        self.scale = scale_factor 

    def forward(self, weights, returns_matrix, prev_weights=None):
        portfolio_returns = torch.matmul(returns_matrix, weights.t()) 
        mean_return = torch.mean(portfolio_returns)

        total_obs = portfolio_returns.size(0)
        partition_size = total_obs // self.num_partitions
        clean_returns = portfolio_returns[: self.num_partitions * partition_size]
        
        reshaped_returns = clean_returns.view(self.num_partitions, partition_size)
        sorted_returns, _ = torch.sort(reshaped_returns, dim=1)

        cutoff_idx = max(1, int(partition_size * (1 - self.alpha)))
        partition_cvars = torch.mean(sorted_returns[:, :cutoff_idx], dim=1)
        cvar_avg = torch.mean(partition_cvars)

        regularization = self.L * torch.sum(torch.pow(weights, 2))

        if prev_weights is not None:
            turnover = torch.sum(torch.abs(weights - prev_weights))
            transaction_friction = self.c * turnover
        else:
            transaction_friction = torch.tensor(0.0, device=weights.device)

        objective = mean_return + (self.gamma * cvar_avg) - regularization - transaction_friction
        return -objective * self.scale 

def optimize_sector_via_nn(stock_returns: pd.DataFrame, nn_lookback_days: int) -> pd.Series:
    N_stocks = stock_returns.shape[1]
    model = IntraSectorNN(N_stocks * 2, 64, N_stocks, temperature=0.5)
    loss_fn = FrictionAdjustedPartitionedLoss(cost_rate=0.0050)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    simulation_dates = stock_returns.index[nn_lookback_days:]
    optimized_returns = []
    locked_prev_weights = None
    
    for step, current_date in enumerate(simulation_dates):
        end_idx = stock_returns.index.get_loc(current_date)
        start_idx = end_idx - nn_lookback_days
        
        window_returns = stock_returns.iloc[start_idx:end_idx]
        returns_tensor = torch.tensor(window_returns.values, dtype=torch.float32)
        
        feature_start = max(0, end_idx - 100)
        means = stock_returns.iloc[feature_start:end_idx].mean().values
        vols = stock_returns.iloc[feature_start:end_idx].std().values
        input_tensor = torch.tensor(np.concatenate([means, vols]) * 100.0, dtype=torch.float32).view(1, -1)
        
        epochs = 300 if step == 0 else (50 if step % 252 == 0 else 5)
        for _ in range(epochs):
            model.zero_grad()
            w_pred = model(input_tensor)
            loss = loss_fn(w_pred, returns_tensor, prev_weights=locked_prev_weights)
            loss.backward()
            optimizer.step()
            
        with torch.no_grad():
            w_tensor = model(input_tensor)
            weights_today = w_tensor.view(-1).numpy()
            locked_prev_weights = w_tensor.clone().detach()
            
        actual_asset_returns = stock_returns.iloc[end_idx].values
        gross_return = np.dot(weights_today, actual_asset_returns)
        friction_drag = loss_fn.c * np.sum(np.abs(weights_today - locked_prev_weights.view(-1).numpy())) if step > 0 else 0.0
        optimized_returns.append(gross_return - friction_drag)
        
    return pd.Series(optimized_returns, index=simulation_dates)

# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 3 ── DUAL-STREAM RETURN MATRIX BUILDER
# ─────────────────────────────────────────────────────────────────────────────

def compile_dual_streams(master_prices: pd.DataFrame, mapping: dict) -> tuple:
    """
    CRITICAL ARCHITECTURAL UPDATE: 
    Compiles TWO completely independent sector return matrices.
    Stream A: Pure Equal-Weighted returns (Used strictly to train HRP rolling covariances).
    Stream B: Neural Network optimized net returns (Used strictly for out-of-sample PnL).
    """
    master_returns = master_prices.pct_change().dropna()
    nn_lookback_days = NN_LOOKBACK_YEARS * 252 
    
    nn_sector_ledger = {}
    eq_sector_ledger = {}
    
    print(f"\nCompiling dual-stream matrices across {len(SECTORS)} sub-sectors...")
    for sec in SECTORS:
        active_stocks = [t for t in mapping[sec] if t in master_returns.columns]
        if not active_stocks:
            continue
            
        # Stream A Construction: Pure Equal-Weighting
        eq_sector_ledger[sec] = master_returns[active_stocks].mean(axis=1)
        
        # Stream B Construction: Neural Network Sub-Layer Optimization
        stock_subset = master_returns[active_stocks]
        nn_sector_ledger[sec] = optimize_sector_via_nn(stock_subset, nn_lookback_days)
        
    # Align temporal bounds cleanly across both stream dataframes
    eq_df = pd.DataFrame(eq_sector_ledger).dropna()
    nn_df = pd.DataFrame(nn_sector_ledger).dropna()
    
    # Ensure exact index matching
    common_dates = nn_df.index.intersection(eq_df.index)
    return eq_df.loc[common_dates], nn_df.loc[common_dates]

# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 4 ── INTER-SECTOR DCC-GARCH + HRP ENGINE (STAGE 2)
# ─────────────────────────────────────────────────────────────────────────────

def garch_log_likelihood(params: np.ndarray, returns: np.ndarray) -> float:
    p0, p1, p2  = params
    denom = 1.0 + np.exp(p1) + np.exp(p2)
    omega, alpha, beta = np.exp(p0), np.exp(p1)/denom, np.exp(p2)/denom
    T = len(returns)
    sigma2 = np.empty(T)
    sigma2[0] = omega / max(1.0 - alpha - beta, 1e-8)
    for t in range(1, T):
        sigma2[t] = max(omega + alpha * returns[t - 1] ** 2 + beta * sigma2[t - 1], 1e-10)
    return float(-0.5 * np.sum(np.log(sigma2) + returns ** 2 / sigma2))

def fit_garch(returns: np.ndarray) -> dict:
    r = returns - returns.mean()
    sample_var = np.var(r)
    best_result, best_ll = None, np.inf
    for x0 in [[np.log(sample_var*0.05), np.log(0.05/0.05), np.log(0.90/0.05)]]:
        try:
            res = minimize(garch_log_likelihood, x0=x0, args=(r,), method="Nelder-Mead")
            if res.success and res.fun < best_ll:
                best_ll, best_result = res.fun, res
        except: pass
    if best_result is None: return {"sigma2": np.full(len(r), sample_var)}
    p0, p1, p2 = best_result.x
    denom = 1.0 + np.exp(p1) + np.exp(p2)
    omega, alpha, beta = np.exp(p0), np.exp(p1)/denom, np.exp(p2)/denom
    sigma2 = np.empty(len(r))
    sigma2[0] = omega / max(1.0 - alpha - beta, 1e-8)
    for t in range(1, len(r)):
        sigma2[t] = max(omega + alpha * r[t-1]**2 + beta * sigma2[t-1], 1e-10)
    return {"omega": omega, "alpha": alpha, "beta": beta, "sigma2": sigma2}

def fit_dcc(Z_df: pd.DataFrame) -> dict:
    Z = Z_df.values
    Q_bar = np.cov(Z.T) + 1e-6 * np.eye(Z.shape[1])
    def dcc_ll(params):
        p0, p1 = params
        denom = 1.0 + np.exp(p0) + np.exp(p1)
        a, b = np.exp(p0)/denom, np.exp(p1)/denom
        if a + b >= 0.9999: return 1e10
        Q = Q_bar.copy()
        ll = 0.0
        for t in range(len(Z)):
            z = Z[t]
            Q = (1.0 - a - b) * Q_bar + a * np.outer(z, z) + b * Q
            inv_sqrt = np.diag(1.0 / np.maximum(np.sqrt(np.diag(Q)), 1e-10))
            R = inv_sqrt @ Q @ inv_sqrt
            np.fill_diagonal(R, 1.0)
            try:
                sign, log_det = np.linalg.slogdet(R)
                if sign <= 0: return 1e10
                ll += log_det + (z @ np.linalg.inv(R) @ z) - (z @ z)
            except: return 1e10
        return float(0.5 * ll)
    res = minimize(dcc_ll, x0=[np.log(0.05/0.05), np.log(0.90/0.05)], method="Nelder-Mead")
    p0, p1 = res.x
    denom = 1.0 + np.exp(p0) + np.exp(p1)
    a, b = np.exp(p0)/denom, np.exp(p1)/denom
    Q = Q_bar.copy()
    for z in Z: Q = (1.0 - a - b) * Q_bar + a * np.outer(z, z) + b * Q
    invs = np.diag(1.0 / np.maximum(np.sqrt(np.diag(Q)), 1e-10))
    R_today = invs @ Q @ invs
    np.fill_diagonal(R_today, 1.0)
    return {"R_today": np.clip(R_today, -0.9999, 0.9999)}

def hrp_linkage(R: np.ndarray) -> list:
    D = np.sqrt(np.maximum(0.0, 2.0 * (1.0 - np.clip(R, -1.0, 1.0))))
    np.fill_diagonal(D, 0.0)
    return dendrogram(linkage(squareform(D, checks=False), method="ward"), no_plot=True)["leaves"]

def hrp_weights(cov: np.ndarray, ordering: list) -> np.ndarray:
    w = np.ones(cov.shape[0])
    clusters = [list(ordering)]
    def c_var(c): return float(1.0 / np.sum(1.0 / np.maximum(np.diag(cov)[c], 1e-10)))
    while clusters:
        next_c = []
        for c in clusters:
            if len(c) <= 1: continue
            left, right = c[:len(c)//2], c[len(c)//2:]
            alloc = c_var(right) / max(c_var(left) + c_var(right), 1e-10)
            for i in left: w[i] *= alloc
            for i in right: w[i] *= (1.0 - alloc)
            if len(left) > 1: next_c.append(left)
            if len(right) > 1: next_c.append(right)
        clusters = next_c
    return w / w.sum()

# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 5 ── MASTER EXECUTION CONTROLLER
# ─────────────────────────────────────────────────────────────────────────────

def execute_master_pipeline():
    prices_df = download_yahoo_data(ALL_TICKERS, required_years=14.5)
    
    # Extract dual operational streams
    eq_target_stream, nn_target_stream = compile_dual_streams(prices_df, SECTOR_MAPPING)
    
    hrp_buffer_days = int(HRP_TEST_YEARS * 252) + HRP_WINDOW_SIZE 
    eq_lookback_frame = eq_target_stream.iloc[-hrp_buffer_days:]
    nn_execution_frame = nn_target_stream.iloc[-hrp_buffer_days:]
    
    surviving_sectors = list(eq_lookback_frame.columns)
    
    # Stream A maps to continuous log space exclusively for GARCH estimation accuracy
    log_eq_returns = np.log1p(eq_lookback_frame)
    
    total_steps = len(eq_lookback_frame)
    out_of_sample_steps = total_steps - HRP_WINDOW_SIZE
    
    print(f"\nExecuting Dual-Stream Integration Engine over {out_of_sample_steps} tracking days...")
    portfolio_net_yields = []
    tracking_timestamps = []

    for idx in range(HRP_WINDOW_SIZE, total_steps):
        # 1. HRP Lookback Training utilizes Stream A (Equal-Weighted Returns) exclusively
        train_slice = log_eq_returns.iloc[idx - HRP_WINDOW_SIZE : idx]
        garch_payload = {sec: fit_garch(train_slice[sec].values) for sec in surviving_sectors}
        
        Z_df = pd.DataFrame({sec: (train_slice[sec].values - train_slice[sec].values.mean()) / 
                             np.maximum(np.sqrt(garch_payload[sec]["sigma2"]), 1e-10) 
                             for sec in surviving_sectors}, index=train_slice.index)
        
        R_today = fit_dcc(Z_df)["R_today"]
        ordering = hrp_linkage(R_today)
        
        cov_today = np.diag([np.sqrt(max(garch_payload[sec]["sigma2"][-1], 1e-10)) for sec in surviving_sectors])
        H_today = cov_today @ R_today @ cov_today
        H_today = (H_today + H_today.T) / 2.0 + 1e-8 * np.eye(len(surviving_sectors))
        
        optimal_hrp_weights = hrp_weights(H_today, ordering)
        
        # 2. Out-of-Sample Ledger Realization utilizes Stream B (NN Returns) exclusively
        realized_nn_yields = nn_execution_frame.iloc[idx].values
        portfolio_step_yield = np.dot(optimal_hrp_weights, realized_nn_yields)
        
        portfolio_net_yields.append(portfolio_step_yield)
        tracking_timestamps.append(nn_execution_frame.index[idx])

    # Compile Visual Output
    ledger = pd.DataFrame({"Date": tracking_timestamps, "Integrated_Return": portfolio_net_yields}).set_index("Date")
    ledger["Integrated_Cumulative"] = (1 + ledger["Integrated_Return"]).cumprod() - 1
    
    # Pure Equal-Weighted Baseline for contrast
    ledger["Benchmark_Return"] = eq_lookback_frame.iloc[HRP_WINDOW_SIZE:].mean(axis=1).values
    ledger["Benchmark_Cumulative"] = (1 + ledger["Benchmark_Return"]).cumprod() - 1

    plt.figure(figsize=(14, 7))
    plt.plot(ledger.index, ledger["Benchmark_Cumulative"] * 100, label="Naive Equal-Weighted Baseline", color='#7f7f7f', linestyle='--', linewidth=1.2)
    plt.plot(ledger.index, ledger["Integrated_Cumulative"] * 100, label="Decoupled Engine (HRP Trained on Stream A, Executed on Stream B)", color='#ff7f0e', linewidth=2.0)
    
    plt.title("Master Out-of-Sample Performance: Decoupled Dual-Stream Optimization", fontsize=14, fontweight="bold")
    plt.ylabel("Compounded Net Portfolio Yield (%)", fontsize=12)
    plt.xlabel("Temporal Execution Axis", fontsize=12)
    plt.grid(alpha=0.25)
    plt.legend(fontsize=11, loc="upper left")
    plt.gcf().autofmt_xdate()
    plt.tight_layout()
    plt.savefig("decoupled_master_performance.png", dpi=300)
    print("\nExecution finalized successfully. Output exported to 'decoupled_master_performance.png'.")
    plt.show()

if __name__ == "__main__":
    execute_master_pipeline()