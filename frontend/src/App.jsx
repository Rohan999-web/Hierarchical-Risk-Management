import { useEffect, useState } from "react";
import {
  BarChart,
  Bar,
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from "recharts";
import "./App.css";

function App() {
  const [metrics, setMetrics] = useState(null);
  const [performance, setPerformance] = useState([]);
  const [allocation, setAllocation] = useState([]);
  const [error, setError] = useState(null);

  useEffect(() => {
    Promise.all([
      fetch("http://localhost:8080/api/portfolio/metrics"),
      fetch("http://localhost:8080/api/portfolio/performance"),
      fetch("http://localhost:8080/api/portfolio/allocation"),
    ])
      .then(async ([metricsResponse, performanceResponse, allocationResponse]) => {
        if (
          !metricsResponse.ok ||
          !performanceResponse.ok ||
          !allocationResponse.ok
        ) {
          throw new Error("Failed to fetch portfolio data");
        }

        const metricsData = await metricsResponse.json();
        const performanceData = await performanceResponse.json();
        const allocationData = await allocationResponse.json();

        return {
          metricsData,
          performanceData,
          allocationData,
        };
      })
      .then(({ metricsData, performanceData, allocationData }) => {
        setMetrics(metricsData);

        const formattedPerformance = performanceData.map((row) => ({
          date: row[0],
          hrpCumulative: parseFloat(row[3]) * 100,
          benchmarkCumulative: parseFloat(row[4]) * 100,
        }));

        setPerformance(formattedPerformance);

        const formattedAllocation = Object.entries(allocationData).map(
          ([sector, weight]) => ({
            sector,
            weight: weight * 100,
          })
        );

        setAllocation(formattedAllocation);
      })
      .catch((error) => {
        setError(error.message);
      });
  }, []);

  if (error) {
    return (
      <div className="dashboard">
        <h1>Portfolio Risk Dashboard</h1>
        <div className="error">Error: {error}</div>
      </div>
    );
  }

  if (!metrics || performance.length === 0 || allocation.length === 0) {
    return (
      <div className="dashboard">
        <h1>Portfolio Risk Dashboard</h1>
        <p>Loading portfolio data...</p>
      </div>
    );
  }

  const comparisonData = [
    {
      metric: "Return",
      HRP: metrics.hrpAnnualizedReturn * 100,
      Benchmark: metrics.benchmarkAnnualizedReturn * 100,
    },
    {
      metric: "Volatility",
      HRP: metrics.hrpVolatility * 100,
      Benchmark: metrics.benchmarkVolatility * 100,
    },
    {
      metric: "Sharpe",
      HRP: metrics.hrpSharpeRatio,
      Benchmark: metrics.benchmarkSharpeRatio,
    },
  ];

  return (
    <div className="dashboard">

      {/* HEADER */}
      <header className="header">
  <div>
    <p className="subtitle">HIERARCHICAL RISK MANAGEMENT</p>

    <h1>Portfolio Risk Dashboard</h1>

    <p className="description">
      HRP + Neural Network portfolio analysis
    </p>
  </div>
</header>


      {/* METRIC CARDS */}
      <section className="metrics-grid">

        <div className="metric-card">
          <p>Cumulative Return</p>
          <h2>
            {(metrics.hrpCumulativeReturn * 100).toFixed(2)}%
          </h2>
          <span>HRP + Neural Network</span>
        </div>

        <div className="metric-card">
          <p>Annualized Return</p>
          <h2>
            {(metrics.hrpAnnualizedReturn * 100).toFixed(2)}%
          </h2>
          <span>Annualized</span>
        </div>

        <div className="metric-card">
          <p>Volatility</p>
          <h2>
            {(metrics.hrpVolatility * 100).toFixed(2)}%
          </h2>
          <span>Annualized</span>
        </div>

        <div className="metric-card">
          <p>Sharpe Ratio</p>
          <h2>
            {metrics.hrpSharpeRatio.toFixed(2)}
          </h2>
          <span>Risk-adjusted return</span>
        </div>

        <div className="metric-card">
          <p>Maximum Drawdown</p>
          <h2>
            {(metrics.hrpMaximumDrawdown * 100).toFixed(2)}%
          </h2>
          <span>Peak-to-trough decline</span>
        </div>

      </section>

      {/* PORTFOLIO OVERVIEW */}
<section className="overview-panel panel">

  <div className="panel-header">
    <h2>Portfolio Overview</h2>
    <p>Configuration of the current risk-management pipeline</p>
  </div>

  <div className="overview-grid">

    <div className="overview-item">
      <span>Initial Universe</span>
      <strong>180 Stocks</strong>
    </div>

    <div className="overview-item">
      <span>Sectors</span>
      <strong>6</strong>
    </div>

    <div className="overview-item">
      <span>Stocks / Sector</span>
      <strong>15</strong>
    </div>

    <div className="overview-item">
      <span>Feature Lookback</span>
      <strong>100 Days</strong>
    </div>

    <div className="overview-item">
      <span>Training Window</span>
      <strong>2520 Days</strong>
    </div>

    <div className="overview-item">
      <span>Rebalance Frequency</span>
      <strong>20 Days</strong>
    </div>

  </div>

</section>

{/* METHODOLOGY */}
<section className="panel methodology-panel">

  <div className="panel-header">
    <h2>Portfolio Methodology</h2>
    <p>Risk-aware portfolio construction pipeline</p>
  </div>

  <div className="methodology-grid">

    <div className="methodology-step">
      <div className="step-number">01</div>
      <h3>Stock Selection</h3>
      <p>
        Fundamental and liquidity filters are used to create
        the eligible stock universe.
      </p>
    </div>

    <div className="methodology-step">
      <div className="step-number">02</div>
      <h3>Risk Modeling</h3>
      <p>
        GARCH estimates volatility while DCC models
        time-varying correlations.
      </p>
    </div>

    <div className="methodology-step">
      <div className="step-number">03</div>
      <h3>Hierarchical Risk Parity</h3>
      <p>
        Correlated assets are clustered and capital is
        allocated using hierarchical risk principles.
      </p>
    </div>

    <div className="methodology-step">
      <div className="step-number">04</div>
      <h3>Neural Network</h3>
      <p>
        The neural network determines intra-sector stock
        weights while incorporating return and tail-risk objectives.
      </p>
    </div>

    <div className="methodology-step">
      <div className="step-number">05</div>
      <h3>Walk-Forward Backtest</h3>
      <p>
        Portfolio performance is evaluated using rolling
        out-of-sample data.
      </p>
    </div>

  </div>

</section>

      {/* PERFORMANCE CURVE */}
      <section className="panel performance-panel">

        <div className="panel-header">

          <h2>Portfolio Performance</h2>

          <p>
            Cumulative return during the out-of-sample backtest
          </p>

        </div>

        <ResponsiveContainer width="100%" height={420}>

          <LineChart data={performance}>

            <CartesianGrid strokeDasharray="3 3" />

            <XAxis
              dataKey="date"
              tickFormatter={(value) => value.substring(0, 7)}
            />

            <YAxis
              tickFormatter={(value) => `${value.toFixed(0)}%`}
            />

            <Tooltip
              formatter={(value) => `${value.toFixed(2)}%`}
            />

            <Line
              type="monotone"
              dataKey="hrpCumulative"
              name="HRP + NN"
              dot={false}
            />

            <Line
              type="monotone"
              dataKey="benchmarkCumulative"
              name="Benchmark"
              dot={false}
            />

          </LineChart>

        </ResponsiveContainer>

      </section>


      {/* LOWER SECTION */}
      <section className="content-grid">

        {/* COMPARISON CHART */}
        <div className="panel">

          <div className="panel-header">

            <h2>HRP + NN vs Benchmark</h2>

            <p>
              Annualized performance comparison
            </p>

          </div>

          <ResponsiveContainer width="100%" height={350}>

            <BarChart data={comparisonData}>

              <CartesianGrid strokeDasharray="3 3" />

              <XAxis dataKey="metric" />

              <YAxis />

              <Tooltip />

              <Bar
                dataKey="HRP"
                name="HRP + NN"
              />

              <Bar
                dataKey="Benchmark"
                name="Benchmark"
              />

            </BarChart>

          </ResponsiveContainer>

        </div>


        {/* SECTOR ALLOCATION */}
        <div className="panel">

          <div className="panel-header">

            <h2>Sector Allocation</h2>

            <p>
              Latest HRP portfolio weights
            </p>

          </div>

          <ResponsiveContainer width="100%" height={350}>

            <BarChart
              data={allocation}
              layout="vertical"
              margin={{
                top: 10,
                right: 20,
                left: 20,
                bottom: 10,
              }}
            >

              <CartesianGrid strokeDasharray="3 3" />

              <XAxis
                type="number"
                tickFormatter={(value) => `${value}%`}
              />

              <YAxis
                type="category"
                dataKey="sector"
              />

              <Tooltip
                formatter={(value) => `${value.toFixed(2)}%`}
              />

              <Bar
                dataKey="weight"
                name="Allocation"
              />

            </BarChart>

          </ResponsiveContainer>

        </div>

      </section>


      {/* BENCHMARK TABLE */}
      <section className="panel benchmark-panel">

        <div className="panel-header">

          <h2>Benchmark Comparison</h2>

          <p>
            Final backtest summary
          </p>

        </div>

        <div className="comparison">

          <div>
            <span>Metric</span>
            <strong>HRP + NN</strong>
            <strong>Benchmark</strong>
          </div>

          <div>
            <span>Cumulative Return</span>

            <strong>
              {(metrics.hrpCumulativeReturn * 100).toFixed(2)}%
            </strong>

            <strong>
              {(metrics.benchmarkCumulativeReturn * 100).toFixed(2)}%
            </strong>
          </div>

          <div>
            <span>Annualized Return</span>

            <strong>
              {(metrics.hrpAnnualizedReturn * 100).toFixed(2)}%
            </strong>

            <strong>
              {(metrics.benchmarkAnnualizedReturn * 100).toFixed(2)}%
            </strong>
          </div>

          <div>
            <span>Volatility</span>

            <strong>
              {(metrics.hrpVolatility * 100).toFixed(2)}%
            </strong>

            <strong>
              {(metrics.benchmarkVolatility * 100).toFixed(2)}%
            </strong>
          </div>

          <div>
            <span>Sharpe Ratio</span>

            <strong>
              {metrics.hrpSharpeRatio.toFixed(2)}
            </strong>

            <strong>
              {metrics.benchmarkSharpeRatio.toFixed(2)}
            </strong>
          </div>

          <div>
            <span>Maximum Drawdown</span>

            <strong>
              {(metrics.hrpMaximumDrawdown * 100).toFixed(2)}%
            </strong>

            <strong>
              {(metrics.benchmarkMaximumDrawdown * 100).toFixed(2)}%
            </strong>
          </div>

        </div>

      </section>

    </div>
  );
}

export default App;