Perfect. Working tree is clean, and we already have an empty README.md:

README.md    0 bytes

So we don't need to create another file. We'll fill this one.

One important thing before we write it: the README should describe the implementation we actually have, not the idealized research paper. That means we'll clearly distinguish the current backtest, architecture, methodology, results, and limitations.

I'll prepare the README in a professional GitHub style with:

project overview
problem statement
objectives
architecture
methodology
Python + Java + React stack
project structure
current backtest results
API endpoints
dashboard
setup/run instructions
limitations/future improvements

And we'll use the actual result from our final run rather than inventing numbers:

HRP + NN cumulative return: 170.01%
Benchmark: 171.73%
Annualized return: 29.13%
Annualized volatility: 12.34%
Sharpe: 2.07
Maximum drawdown: −17.05%
Backtest period: 30 Sep 2022 – 11 Sep 2026

One wording point I'll be careful about: these are historical backtest results, not live investment returns.

I'll generate the README content next; then we can put it into the actual README.md, review it, and commit it separately.

just do it bro why I have to tell you 
Analyzed

Done bro. I created the full professional README based on the project we actually built — including the methodology, architecture, real backtest results, API endpoints, dashboard, setup instructions, limitations, and future enhancements.

Download the completed README.md

Next step

Copy/replace your current empty:

Code_Group_Project/README.md

with this file.

Then run:

git status

Send me the output. We'll review exactly what Git sees before committing it. 

README.md
Document
how can I replace it with the empty one 

Yep, easiest way on Windows: replace the contents of your existing README.md with the downloaded one.

Option 1 — easiest
Download the README I created:

Download README.md

Open your project folder:

C:\Users\rohan\Downloads\Trading Project\Code_Group_Project

You'll see:

README.md
Open the downloaded README.md in VS Code.

Press:

Ctrl + A

then:

Ctrl + C
Open the empty README.md inside your project.

Press:

Ctrl + V

Save:

Ctrl + S

That's it. You don't need to delete or rename the existing file.

Then verify Git

In your project terminal:

git status

You should see something like:

Changes not staged for commit:
    modified: README.md

Send me that output and we'll commit it.

README.md
Document

Library
/
Hierarchical Risk Management
/
README.md
Hierarchical Risk-Aware Portfolio Management

An end-to-end quantitative portfolio construction and risk-management system that combines fundamental stock selection, volatility and correlation modeling, Hierarchical Risk Parity (HRP), neural-network-based intra-sector allocation, CVaR-based risk objectives, and walk-forward backtesting.

The project also includes a Spring Boot REST API and React dashboard for presenting portfolio performance, risk metrics, benchmark comparison, and sector allocation.

Note: This project is a historical research/backtesting system. It does not execute live trades or represent actual invested capital.

1. Problem Statement

Selecting a portfolio from a large stock universe is difficult because a portfolio should not be evaluated only by individual stock returns.

A practical portfolio construction process also needs to consider:

Fundamental and liquidity characteristics of stocks
Individual asset volatility
Relationships and correlations between assets
Portfolio concentration
Extreme downside risk
Portfolio turnover and transaction costs
Out-of-sample performance

This project addresses the problem by building a multi-stage, risk-aware portfolio construction pipeline and evaluating it using rolling walk-forward backtesting.

2. Project Objectives
Reduce a large stock universe to a smaller eligible investment universe.
Use fundamental and liquidity factors for stock selection.
Model changing volatility using GARCH.
Model time-varying correlations using DCC.
Apply Hierarchical Risk Parity for risk-aware portfolio allocation.
Use a neural network for intra-sector stock allocation.
Incorporate Partitioned CVaR to account for tail risk.
Evaluate the strategy using walk-forward out-of-sample backtesting.
Compare portfolio performance with a benchmark.
Expose results through a Spring Boot REST API.
Visualize results through a React dashboard.
3. System Architecture
                    Stock Universe
                         |
                         v
              +----------------------+
              |  Stock Selection     |
              |  Fundamental +       |
              |  Liquidity Filters   |
              +----------+-----------+
                         |
                         v
                  Selected Universe
                         |
                         v
              +----------------------+
              |   Risk Modeling     |
              | GARCH + DCC          |
              +----------+-----------+
                         |
                         v
              +----------------------+
              | Hierarchical Risk    |
              | Parity (HRP)         |
              +----------+-----------+
                         |
                         v
              +----------------------+
              | Neural Network +     |
              | Partitioned CVaR      |
              +----------+-----------+
                         |
                         v
              Walk-Forward Backtest
                         |
             +-----------+-----------+
             |                       |
             v                       v
       Performance Data        Portfolio Weights
             |                       |
             +-----------+-----------+
                         |
                         v
                Spring Boot API
                         |
                         v
                  React Dashboard
4. Methodology
Stage 1 — Stock Selection

The initial universe is organized into:

6 sectors
3 market-cap categories
18 sector-cap buckets
10 stocks selected per bucket

This produces an initial universe of approximately 180 stocks.

The selection process uses:

Return on Equity (ROE)
Debt-to-Equity (D/E)
Price-to-Earnings (P/E)
Revenue Growth
Free Cash Flow Yield
Average Daily Traded Value (ADTV)

Higher-is-better metrics are normalized positively, while lower-is-better metrics such as D/E and P/E are inverted during normalization.

A composite score is calculated for ranking candidates within each sector-cap bucket.

Stage 2 — Risk Modeling and HRP

The selected stocks are analyzed using historical return data.

GARCH

GARCH is used to estimate changing volatility rather than assuming that asset volatility remains constant.

DCC

Dynamic Conditional Correlation (DCC) is used to estimate changing correlations between assets.

This helps identify assets that tend to move together.

Hierarchical Risk Parity

HRP uses the risk and correlation structure to group related assets and construct a diversified allocation.

The goal is to avoid treating every stock as an independent investment and to account for the way assets interact within a portfolio.

Stage 3 — Neural Network and CVaR

For the intra-sector allocation layer, 15 eligible stocks are used per sector.

For each stock, the model uses:

100-day average return
100-day volatility

This produces:

15 stocks × 2 features = 30 input features

The neural network architecture is:

30 inputs
   |
64 neurons
   |
64 neurons
   |
15 outputs

A Softmax output converts the model output into non-negative stock weights that sum to 1 within the sector.

The optimization objective considers:

Portfolio return
Partitioned CVaR
L2 concentration penalty
Turnover
Partitioned CVaR

Historical returns are divided into five contiguous blocks. CVaR is calculated for each block and aggregated so that downside risk is not evaluated only through one combined historical distribution.

Walk-Forward Backtesting

The strategy is evaluated using a rolling out-of-sample process.

A historical training window is used to construct the portfolio. The resulting allocation is then evaluated on subsequent unseen observations before the process moves forward.

Current configuration includes:

2520 trading-day training window
100-day feature lookback
20-day portfolio rebalancing
5. Backtest Results

The current final backtest covers approximately:

30 September 2022 to 11 September 2026

Historical results from the current implementation:

Metric	HRP + NN	Benchmark
Cumulative Return	170.01%	171.73%
Annualized Return	29.13%	29.35%
Annualized Volatility	12.34%	13.12%
Sharpe Ratio	2.07	1.96
Maximum Drawdown	-17.05%	-18.95%

These figures are historical backtest results, not guaranteed future returns or actual investment performance.

6. Technology Stack
Quantitative / Machine Learning
Python
Pandas
NumPy
SciPy
PyTorch
ARCH
scikit-learn
yfinance
Backend
Java
Spring Boot
Maven
REST API
Frontend
React
Vite
Recharts
JavaScript
CSS
Development
Git
GitHub
VS Code / IntelliJ IDEA
7. Project Structure
Code_Group_Project/
|
├── modules/
│   ├── backtest.py
│   ├── cvar.py
│   ├── data_loader.py
│   ├── dcc.py
│   ├── features.py
│   ├── garch.py
│   ├── hrp.py
│   ├── loss.py
│   ├── neural_network.py
│   ├── nn_optimizer.py
│   ├── portfolio.py
│   ├── returns.py
│   ├── stock_selector.py
│   └── train_nn.py
|
├── backend/
│   ├── pom.xml
│   └── src/
│       └── main/
│           └── java/
│               └── com/portfolio/riskengine/
│                   ├── controller/
│                   ├── model/
│                   └── service/
|
├── frontend/
│   ├── package.json
│   └── src/
│       ├── App.jsx
│       ├── App.css
│       └── main.jsx
|
├── run_final_backtest.py
├── sector_wise_stock_selection_yfinance.ipynb
├── final_selection_30_per_sector.csv
├── stage3_selected_universe.csv
├── final_hrp_nn_metrics.csv
├── final_hrp_nn_performance.csv
├── final_hrp_nn_weights.csv
└── README.md
8. Backend API

The Spring Boot backend exposes portfolio results through REST endpoints.

Endpoint	Purpose
GET /api/portfolio/health	API health check
GET /api/portfolio/metrics	Portfolio and benchmark metrics
GET /api/portfolio/performance	Historical performance data
GET /api/portfolio/allocation	Latest sector allocation

The backend currently reads the generated quantitative-engine output files and exposes them to the frontend.

9. Dashboard

The React dashboard provides:

Cumulative portfolio performance
Annualized return
Volatility
Sharpe ratio
Maximum drawdown
Portfolio configuration
Methodology overview
Portfolio vs benchmark performance
Latest sector allocation
Benchmark comparison

The dashboard is intended as the presentation and monitoring layer for the quantitative engine.

10. Running the Project
Python Quant Engine

Create and activate a Python virtual environment and install the required dependencies.

Then run:

python run_final_backtest.py

The backtest generates portfolio performance, metrics, and allocation CSV files.

Spring Boot Backend

From the backend directory:

mvnw spring-boot:run

The API runs on:

http://localhost:8080
React Frontend

From the frontend directory:

npm install
npm run dev

The dashboard runs on:

http://localhost:5173
11. Current Limitations
The current system is a historical backtesting and research system, not a live trading platform.
Current portfolio outputs persist sector-level weights; historical stock-level neural-network weights are not currently exposed by the final output file.
Market data depends on the available historical data source and its data quality.
Backtest results are sensitive to model parameters, data availability, transaction-cost assumptions, and the selected universe.
Historical performance does not guarantee future performance.
12. Future Enhancements

Potential extensions include:

Persisting historical stock-level NN allocations.
Refreshing market data automatically.
Adding a database layer for portfolio and market data.
Building a live portfolio-rebalancing workflow.
Adding authentication and user-specific portfolios.
Adding richer risk analytics such as rolling CVaR and exposure analysis.
Improving deployment with Docker and cloud infrastructure.
Extending the dashboard with historical portfolio holdings and rebalance history.
13. Key Takeaway

The project is designed around one central question:

Given a large stock universe, how can we construct and evaluate a diversified portfolio while explicitly considering volatility, correlation, concentration, and downside risk?

The system combines quantitative finance, machine learning, backend engineering, and frontend visualization into one end-to-end portfolio management workflow.