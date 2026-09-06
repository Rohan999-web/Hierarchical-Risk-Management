"""
=============================================================================
DCC-GARCH + HRP Portfolio Allocation (Rolling Backtest)
=============================================================================
Expanded Universe: 180 Stocks loaded dynamically from file
Source: Yahoo Finance (5 Years Data)
Rolling window: 100 days
Rebalancing: Daily over the out-of-sample period.
=============================================================================
"""

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

# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 0 ── UNIVERSE SETUP (Dynamic Loading)
# ─────────────────────────────────────────────────────────────────────────────

# Define the input file containing your 180 stocks
UNIVERSE_FILE = "final_selection_30_per_sector.csv" # Change to .xlsx if using Excel directly
WINDOW_SIZE   = 100                   # 100-day rolling lookback
YEARS_DATA    = 5

def load_universe_mapping(filepath: str) -> dict:
    """
    Reads the input file and maps stocks to sectors dynamically.
    Expects columns named 'Ticker' and 'Sector'.
    """
    print(f"Reading stock universe from '{filepath}'...")
    if filepath.endswith('.csv'):
        df = pd.read_csv(filepath)
    elif filepath.endswith(('.xls', '.xlsx')):
        df = pd.read_excel(filepath)
    else:
        raise ValueError("Unsupported file format. Please provide a .csv or .xlsx file.")

    # Clean standard column names
    df.columns = df.columns.str.strip().str.title()
    
    if 'Ticker' not in df.columns or 'Sector' not in df.columns:
        raise ValueError("File must contain 'Ticker' and 'Sector' columns.")

    universe = {}
    for sector, group in df.groupby('Sector'):
        # Ensure standard Yahoo Finance ticker formats (e.g., adding .NS for NSE if needed)
        tickers = group['Ticker'].str.strip().tolist()
        universe[sector] = tickers
        
    total_stocks = sum(len(t) for t in universe.values())
    print(f"Successfully loaded {total_stocks} stocks across {len(universe)} sectors.")
    return universe

# Initialize Universe dynamically
try:
    SECTOR_MAPPING = load_universe_mapping(UNIVERSE_FILE)
    SECTORS = list(SECTOR_MAPPING.keys())
    N_SECTORS = len(SECTORS)
    ALL_TICKERS = [t for sublist in SECTOR_MAPPING.values() for t in sublist]
except Exception as e:
    print(f"\nError initializing universe: {e}")
    print(f"Please ensure '{UNIVERSE_FILE}' exists in the current folder.")
    sys.exit(1)

# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 1 ── ROBUST YAHOO FINANCE LOADER
# ─────────────────────────────────────────────────────────────────────────────

def download_yahoo_data(tickers: list, years: int = 5) -> pd.DataFrame:
    """
    Downloads daily data using a disguised browser session and local caching.
    Prevents API blocks and avoids re-downloading successful files on failure.
    """
    import yfinance as yf
    import requests
    
    # 1. Setup a "Stealth" Session to bypass Yahoo's bot detection
    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    })

    end_date = datetime.today()
    start_date = end_date - timedelta(days=int(years * 365.25) + 30)
    
    # Create a temporary folder to cache individual stock downloads
    cache_dir = "yf_cache"
    os.makedirs(cache_dir, exist_ok=True)

    print(f"\nInitiating robust fetch for {len(tickers)} stocks...")
    print(f"Caching intermediate files to './{cache_dir}' to prevent total data loss on IP blocks.")
    
    df_list = []
    failed_tickers = []
    
    for i, ticker in enumerate(tickers, 1):
        # Sanitize filename for Windows/Mac (replace & with _amp_)
        safe_filename = ticker.replace("&", "_amp_") + ".csv"
        cache_path = os.path.join(cache_dir, safe_filename)
        
        # Check if we already successfully downloaded this stock in a previous run
        if os.path.exists(cache_path):
            try:
                series = pd.read_csv(cache_path, index_col=0, parse_dates=True).iloc[:, 0]
                series.name = ticker
                df_list.append(series)
                continue
            except Exception:
                pass # If cache file is corrupted, redownload
                
        try:
            # Download using the custom disguised session
            data = yf.download(ticker, start=start_date, end=end_date, session=session, progress=False)
            
            if data.empty or 'Close' not in data.columns:
                failed_tickers.append(ticker)
                continue
                
            close_series = data['Close'].copy()
            if isinstance(close_series, pd.DataFrame):
                close_series = close_series.iloc[:, 0]
                
            close_series.name = ticker
            
            # Drop empty rows and save to local cache instantly
            close_series.dropna().to_csv(cache_path, header=[ticker])
            df_list.append(close_series)
            
            # Polite, slightly randomized delay to mimic human browsing
            time.sleep(np.random.uniform(0.3, 0.7))
            
        except Exception as e:
            failed_tickers.append(ticker)
            
        # Real-time console update
        sys.stdout.write(f"  Processed {i}/{len(tickers)} assets (Cached locally)...\r")
        sys.stdout.flush()

    print("\n\nConsolidating cached files into alignment matrix...")
    if failed_tickers:
        print(f"  Note: Skipped {len(failed_tickers)} problematic assets: {failed_tickers[:8]}...")
        print("  Algorithm will dynamically adjust weights to active surviving assets.")

    # Combine all loaded dataframes
    wide_df = pd.concat(df_list, axis=1).sort_index()

    # Drop highly illiquid or newly listed stocks missing >10% of the timeline
    threshold = int(len(wide_df) * 0.90)
    valid_cols = wide_df.dropna(thresh=threshold, axis=1).columns
    dropped = set(wide_df.columns) - set(valid_cols)
    
    if dropped:
        print(f"  Cleaned out {len(dropped)} assets due to incomplete 5-year tracking history.")
        
    wide_df = wide_df[valid_cols].ffill().bfill()
    
    print(f"Matrix locked. Final Active Universe: {wide_df.shape[1]} Stocks across {wide_df.shape[0]} trading days.")
    return wide_df

    print("\nProcessing downloaded time-series matrix...")
    if failed_tickers:
        print(f"  WARNING: Failed to fetch data for {len(failed_tickers)} stocks: {failed_tickers[:5]}...")

    # Merge all valid stock series horizontally
    wide_df = pd.concat(df_list, axis=1)
    wide_df = wide_df.sort_index()

    # Data Quality Check: Drop stocks missing more than 10% of their historical window
    threshold = int(len(wide_df) * 0.90)
    valid_cols = wide_df.dropna(thresh=threshold, axis=1).columns
    dropped_cols = set(wide_df.columns) - set(valid_cols)
    
    if dropped_cols:
        print(f"  Dropped {len(dropped_cols)} stocks due to insufficient historical history (>10% missing).")
        
    wide_df = wide_df[valid_cols]
    
    # Forward-fill internal trading halts, back-fill initial alignment gaps
    wide_df = wide_df.ffill().bfill()
    
    print(f"Final usable pricing matrix shape: {wide_df.shape} (Trading Days x Stocks)")
    return wide_df

# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 2 ── RETURN COMPUTATION (Dynamic Sector Equal-Weighting)
# ─────────────────────────────────────────────────────────────────────────────

def compute_sector_returns(prices: pd.DataFrame, mapping: dict):
    """
    Computes equal-weighted sector returns using dynamically available stocks.
    Returns BOTH Log Returns (for DCC-GARCH modeling) and Simple Returns (for PnL).
    """
    log_stock_rets = np.log(prices / prices.shift(1)).dropna()
    simple_stock_rets = (prices / prices.shift(1) - 1).dropna()

    sector_log = {}
    sector_simple = {}
    
    for sec in SECTORS:
        # Dynamically identify valid constituents present in the cleaned pricing matrix
        tickers_in_sec = [t for t in mapping[sec] if t in prices.columns]
        
        if not tickers_in_sec:
            print(f"  CRITICAL WARNING: No valid pricing data remaining for sector '{sec}'.")
            continue
            
        w = 1.0 / len(tickers_in_sec)
        
        sector_log[sec] = log_stock_rets[tickers_in_sec].sum(axis=1) * w
        sector_simple[sec] = simple_stock_rets[tickers_in_sec].sum(axis=1) * w

    return pd.DataFrame(sector_log).dropna(), pd.DataFrame(sector_simple).dropna()

# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 3 ── GARCH(1,1) ESTIMATION
# ─────────────────────────────────────────────────────────────────────────────

def garch_log_likelihood(params: np.ndarray, returns: np.ndarray) -> float:
    p0, p1, p2  = params
    denom = 1.0 + np.exp(p1) + np.exp(p2)
    omega = np.exp(p0)
    alpha = np.exp(p1) / denom
    beta  = np.exp(p2) / denom

    T = len(returns)
    sigma2 = np.empty(T)
    sigma2[0] = omega / max(1.0 - alpha - beta, 1e-8)

    for t in range(1, T):
        sigma2[t] = omega + alpha * returns[t - 1] ** 2 + beta * sigma2[t - 1]
        sigma2[t] = max(sigma2[t], 1e-10)

    ll = -0.5 * np.sum(np.log(sigma2) + returns ** 2 / sigma2)
    return -ll

def garch_score(params: np.ndarray, returns: np.ndarray) -> np.ndarray:
    p0, p1, p2  = params
    ep1, ep2    = np.exp(p1), np.exp(p2)
    denom       = 1.0 + ep1 + ep2
    omega       = np.exp(p0)
    alpha       = ep1 / denom
    beta        = ep2 / denom

    T      = len(returns)
    sigma2 = np.empty(T)
    sigma2[0] = omega / max(1.0 - alpha - beta, 1e-8)

    v_w = np.empty(T); v_a = np.empty(T); v_b = np.empty(T)
    persistence = 1.0 - alpha - beta
    v_w[0] = 1.0 / max(persistence, 1e-8)
    v_a[0] = omega / max(persistence ** 2, 1e-12)
    v_b[0] = omega / max(persistence ** 2, 1e-12)

    for t in range(1, T):
        sigma2[t] = max(omega + alpha * returns[t-1]**2 + beta * sigma2[t-1], 1e-10)
        v_w[t] = 1.0 + beta * v_w[t-1]
        v_a[t] = returns[t-1]**2 + beta * v_a[t-1]
        v_b[t] = sigma2[t-1] + beta * v_b[t-1]

    z2 = returns**2 / sigma2
    cf = (z2 - 1.0) / (2.0 * sigma2)

    dL_domega = np.sum(cf * v_w)
    dL_dalpha = np.sum(cf * v_a)
    dL_dbeta  = np.sum(cf * v_b)

    g_p0 = dL_domega * omega
    g_p1 = dL_dalpha * alpha * (1.0 - alpha) + dL_dbeta * (-alpha * beta)
    g_p2 = dL_dalpha * (-alpha * beta) + dL_dbeta * beta * (1.0 - beta)

    return np.array([-g_p0, -g_p1, -g_p2])

def fit_garch(returns: np.ndarray) -> dict:
    r = returns - returns.mean()
    sample_var = np.var(r)
    best_result, best_ll = None, np.inf

    starting_points = [
        [np.log(sample_var * 0.05), np.log(0.05 / 0.05), np.log(0.90 / 0.05)],
        [np.log(sample_var * 0.10), np.log(0.08 / 0.04), np.log(0.88 / 0.04)]
    ]

    for x0 in starting_points:
        try:
            res = minimize(fun=garch_log_likelihood, x0=x0, args=(r,), jac=garch_score,
                           method="BFGS", options={"maxiter": 200, "gtol": 1e-6})
            if res.success and res.fun < best_ll:
                best_ll = res.fun
                best_result = res
        except:
            continue

    if best_result is None:
        return {"sigma2": np.full(len(r), sample_var)}

    p0, p1, p2 = best_result.x
    denom = 1.0 + np.exp(p1) + np.exp(p2)
    omega, alpha, beta = np.exp(p0), np.exp(p1) / denom, np.exp(p2) / denom

    sigma2 = np.empty(len(r))
    sigma2[0] = omega / max(1.0 - alpha - beta, 1e-8)
    for t in range(1, len(r)):
        sigma2[t] = max(omega + alpha * r[t-1]**2 + beta * sigma2[t-1], 1e-10)

    return {"omega": omega, "alpha": alpha, "beta": beta, "sigma2": sigma2}

def fit_all_garch(sector_returns: pd.DataFrame, verbose: bool = False) -> dict:
    results = {}
    for sec in sector_returns.columns:
        res = fit_garch(sector_returns[sec].values)
        results[sec] = res
        if verbose:
            print(f"  GARCH [{sec:15s}] ω={res.get('omega', 0):.2e} α={res.get('alpha', 0):.4f} β={res.get('beta', 0):.4f}")
    return results

# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 4 ── DCC ESTIMATION
# ─────────────────────────────────────────────────────────────────────────────

def extract_standardized_residuals(sector_returns: pd.DataFrame, garch_results: dict) -> pd.DataFrame:
    Z = {}
    for sec in sector_returns.columns:
        r = sector_returns[sec].values
        sig = np.sqrt(garch_results[sec]["sigma2"])
        Z[sec] = (r - r.mean()) / np.maximum(sig, 1e-10)
    return pd.DataFrame(Z, index=sector_returns.index)

def dcc_log_likelihood(params: np.ndarray, Z: np.ndarray, Q_bar: np.ndarray) -> float:
    p0, p1 = params
    denom = 1.0 + np.exp(p0) + np.exp(p1)
    a, b = np.exp(p0) / denom, np.exp(p1) / denom

    if a + b >= 0.9999: return 1e10

    T, N = Z.shape
    Q = Q_bar.copy()
    ll_total = 0.0

    for t in range(T):
        z = Z[t]
        Q = (1.0 - a - b) * Q_bar + a * np.outer(z, z) + b * Q
        diag_sqrt = np.maximum(np.sqrt(np.diag(Q)), 1e-10)
        D_inv = np.diag(1.0 / diag_sqrt)
        R = D_inv @ Q @ D_inv
        np.fill_diagonal(R, 1.0)

        try:
            sign, log_det = np.linalg.slogdet(R)
            if sign <= 0: return 1e10
            ll_total += log_det + (z @ np.linalg.inv(R) @ z) - (z @ z)
        except:
            return 1e10
    return 0.5 * ll_total

def fit_dcc(Z_df: pd.DataFrame, verbose: bool = False) -> dict:
    Z = Z_df.values
    Q_bar = np.cov(Z.T)
    if np.linalg.eigvalsh(Q_bar).min() < 1e-8:
        Q_bar += 1e-6 * np.eye(Q_bar.shape[0])

    x0 = [np.log(0.05 / 0.05), np.log(0.90 / 0.05)]
    res = minimize(fun=dcc_log_likelihood, x0=x0, args=(Z, Q_bar), method="Nelder-Mead")
    
    p0_hat, p1_hat = res.x
    denom = 1.0 + np.exp(p0_hat) + np.exp(p1_hat)
    a, b = np.exp(p0_hat) / denom, np.exp(p1_hat) / denom

    if verbose:
        print(f"  DCC a={a:.4f}, b={b:.4f}")

    Q = Q_bar.copy()
    for t in range(len(Z)):
        z = Z[t]
        Q = (1.0 - a - b) * Q_bar + a * np.outer(z, z) + b * Q
    
    D_inv = np.diag(1.0 / np.maximum(np.sqrt(np.diag(Q)), 1e-10))
    R_today = D_inv @ Q @ D_inv
    np.fill_diagonal(R_today, 1.0)
    R_today = np.clip(R_today, -0.9999, 0.9999)

    return {"R_today": R_today}

# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 5 ── DISTANCE MATRIX (Mantegna 1999)
# ─────────────────────────────────────────────────────────────────────────────

def correlation_to_distance(R: np.ndarray) -> np.ndarray:
    R_clipped = np.clip(R, -1.0, 1.0)
    np.fill_diagonal(R_clipped, 1.0)
    D = np.sqrt(2.0 * (1.0 - R_clipped))
    np.fill_diagonal(D, 0.0)
    return D

# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 6 ── HIERARCHICAL CLUSTERING (Ward's linkage)
# ─────────────────────────────────────────────────────────────────────────────

def run_hierarchical_clustering(D: np.ndarray, labels: list) -> tuple:
    D_condensed = squareform(D, checks=False)
    Z = linkage(D_condensed, method="ward")
    return Z

def get_leaf_ordering(Z: np.ndarray, n: int) -> list:
    dn = dendrogram(Z, no_plot=True)
    return dn["leaves"]

# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 7 ── HIERARCHICAL RISK PARITY
# ─────────────────────────────────────────────────────────────────────────────

def cluster_variance(cov: np.ndarray, indices: list) -> float:
    diag_vars = np.diag(cov)[indices]
    inv_var_sum = np.sum(1.0 / np.maximum(diag_vars, 1e-10))
    return 1.0 / max(inv_var_sum, 1e-10)

def hrp_weights(cov: np.ndarray, ordering: list) -> np.ndarray:
    N = cov.shape[0]
    w = np.ones(N)
    clusters = [list(ordering)]

    while clusters:
        new_clusters = []
        for cluster in clusters:
            if len(cluster) <= 1: continue

            mid = len(cluster) // 2
            left  = cluster[:mid]
            right = cluster[mid:]

            var_L = cluster_variance(cov, left)
            var_R = cluster_variance(cov, right)

            alpha = var_R / max(var_L + var_R, 1e-10)

            for idx in left: w[idx] *= alpha
            for idx in right: w[idx] *= (1.0 - alpha)

            if len(left)  > 1: new_clusters.append(left)
            if len(right) > 1: new_clusters.append(right)

        clusters = new_clusters

    w /= w.sum()
    return w

def build_covariance_today(R_today: np.ndarray, garch_results: dict, sectors: list) -> np.ndarray:
    N = len(sectors)
    D = np.zeros(N)
    for i, sec in enumerate(sectors):
        D[i] = np.sqrt(max(garch_results[sec]["sigma2"][-1], 1e-10))

    D_mat = np.diag(D)
    H = D_mat @ R_today @ D_mat
    H = (H + H.T) / 2.0
    H += 1e-8 * np.eye(N)
    return H

# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 8 ── ROLLING BACKTEST ENGINE
# ─────────────────────────────────────────────────────────────────────────────

def run_backtest():
    print("\n" + "=" * 65)
    print("  DCC-GARCH + HRP ROLLING BACKTEST ENGINE (5 YEARS DATA)")
    print("=" * 65)

    # 1. Download Pricing Matrix via Yahoo Finance
    prices = download_yahoo_data(ALL_TICKERS, years=YEARS_DATA)
    
    # 2. Get Dynamic Sector Returns
    log_rets, simple_rets = compute_sector_returns(prices, SECTOR_MAPPING)
    
    # Ensure sectors match exactly across dataframes dynamically
    valid_sectors = list(log_rets.columns)
    N_VALID_SECTORS = len(valid_sectors)
    
    T = len(log_rets)
    if T <= WINDOW_SIZE:
        raise ValueError(f"Not enough data. Total days: {T}, Window: {WINDOW_SIZE}")

    print(f"\nTime-Series successfully structured. Total trading days: {T}.")
    print(f"Active Portfolios Sub-Sectors: {N_VALID_SECTORS}")
    print(f"Rolling Training Window: {WINDOW_SIZE} days.")
    print(f"Out-of-Sample Backtest: {T - WINDOW_SIZE} days (Approx {(T - WINDOW_SIZE)/252:.2f} years).")
    print("Starting optimization loop... (Large matrices active, execution running)\n")

    port_returns = []
    dates = []
    weights_record = []

    # 3. Core Out-of-Sample Rolling Engine
    for t in range(WINDOW_SIZE, T):
        
        # --- A. Data Slicing ---
        train_log = log_rets.iloc[t - WINDOW_SIZE : t]
        
        # --- B. DCC-GARCH Pipeline ---
        garch_res = fit_all_garch(train_log, verbose=False)
        Z_df = extract_standardized_residuals(train_log, garch_res)
        dcc_res = fit_dcc(Z_df, verbose=False)
        R_today = dcc_res["R_today"]

        # --- C. Distance & Ward's Linkage ---
        D_dist = correlation_to_distance(R_today)
        Z_link = run_hierarchical_clustering(D_dist, valid_sectors)
        ordering = get_leaf_ordering(Z_link, N_VALID_SECTORS)

        # --- D. HRP Weights ---
        H_today = build_covariance_today(R_today, garch_res, valid_sectors)
        w = hrp_weights(H_today, ordering)
        
        # --- E. Out-of-Sample PnL ---
        actual_ret_today = simple_rets.iloc[t].values
        port_profit = np.dot(w, actual_ret_today)
        
        port_returns.append(port_profit)
        dates.append(simple_rets.index[t])
        weights_record.append(w)

        # Dynamic console tracking
        if (t - WINDOW_SIZE + 1) % 50 == 0 or t == T - 1:
            sys.stdout.write(f"  Simulated {t - WINDOW_SIZE + 1} / {T - WINDOW_SIZE} target dates...\r")
            sys.stdout.flush()

    # 4. Aggregation and Export
    results_df = pd.DataFrame({"Date": dates, "Portfolio_Return": port_returns}).set_index("Date")
    weights_df = pd.DataFrame(weights_record, index=dates, columns=valid_sectors)

    results_df["Cumulative_Return"] = (1 + results_df["Portfolio_Return"]).cumprod() - 1
    
    # Static Benchmark: Equal allocation across all valid sectors
    eq_w = np.ones(N_VALID_SECTORS) / N_VALID_SECTORS
    results_df["Benchmark_Return"] = simple_rets.iloc[WINDOW_SIZE:].dot(eq_w).values
    results_df["Benchmark_Cumulative"] = (1 + results_df["Benchmark_Return"]).cumprod() - 1

    results_df.to_csv("backtest_performance.csv")
    weights_df.to_csv("backtest_weights.csv")
    print("\n\nExecution Verified! Outputs written to 'backtest_performance.csv' and 'backtest_weights.csv'.")

    # 5. Visual Rendering
    plt.figure(figsize=(12, 6))
    plt.plot(results_df.index, results_df["Cumulative_Return"] * 100, label="Dynamic DCC-GARCH + HRP Allocation", color='#1f77b4', linewidth=1.5)
    plt.plot(results_df.index, results_df["Benchmark_Cumulative"] * 100, label="Naive Equal-Weighted Sub-Sectors", color='#7f7f7f', linestyle='--', linewidth=1.2)
    
    plt.title(f"Rolling Asset Allocation Engine out-of-sample (Universe: {len(ALL_TICKERS)} Initial Assets)", fontsize=13, fontweight="bold")
    plt.ylabel("Cumulative Aggregate Yield (%)", fontsize=11)
    plt.xlabel("Timeline Timeline", fontsize=11)
    plt.grid(alpha=0.3)
    plt.legend(fontsize=10)
    
    plt.tight_layout()
    plt.savefig("backtest_equity_curve.png", dpi=300)
    print("Visual Curve exported to 'backtest_equity_curve.png'.")
    plt.close()

if __name__ == "__main__":
    run_backtest()