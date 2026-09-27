package com.portfolio.riskengine.model;

public class PortfolioMetrics {

    private double hrpCumulativeReturn;
    private double benchmarkCumulativeReturn;
    private double hrpAnnualizedReturn;
    private double benchmarkAnnualizedReturn;
    private double hrpVolatility;
    private double benchmarkVolatility;
    private double hrpSharpeRatio;
    private double benchmarkSharpeRatio;
    private double hrpMaximumDrawdown;
    private double benchmarkMaximumDrawdown;

    public PortfolioMetrics(
            double hrpCumulativeReturn,
            double benchmarkCumulativeReturn,
            double hrpAnnualizedReturn,
            double benchmarkAnnualizedReturn,
            double hrpVolatility,
            double benchmarkVolatility,
            double hrpSharpeRatio,
            double benchmarkSharpeRatio,
            double hrpMaximumDrawdown,
            double benchmarkMaximumDrawdown) {

        this.hrpCumulativeReturn = hrpCumulativeReturn;
        this.benchmarkCumulativeReturn = benchmarkCumulativeReturn;
        this.hrpAnnualizedReturn = hrpAnnualizedReturn;
        this.benchmarkAnnualizedReturn = benchmarkAnnualizedReturn;
        this.hrpVolatility = hrpVolatility;
        this.benchmarkVolatility = benchmarkVolatility;
        this.hrpSharpeRatio = hrpSharpeRatio;
        this.benchmarkSharpeRatio = benchmarkSharpeRatio;
        this.hrpVolatility = hrpVolatility;
        this.benchmarkVolatility = benchmarkVolatility;
        this.hrpSharpeRatio = hrpSharpeRatio;
        this.benchmarkSharpeRatio = benchmarkSharpeRatio;
        this.hrpMaximumDrawdown = hrpMaximumDrawdown;
        this.benchmarkMaximumDrawdown = benchmarkMaximumDrawdown;
    }

    public double getHrpCumulativeReturn() {
        return hrpCumulativeReturn;
    }

    public double getBenchmarkCumulativeReturn() {
        return benchmarkCumulativeReturn;
    }

    public double getHrpAnnualizedReturn() {
        return hrpAnnualizedReturn;
    }

    public double getBenchmarkAnnualizedReturn() {
        return benchmarkAnnualizedReturn;
    }

    public double getHrpVolatility() {
        return hrpVolatility;
    }

    public double getBenchmarkVolatility() {
        return benchmarkVolatility;
    }

    public double getHrpSharpeRatio() {
        return hrpSharpeRatio;
    }

    public double getBenchmarkSharpeRatio() {
        return benchmarkSharpeRatio;
    }

    public double getHrpMaximumDrawdown() {
        return hrpMaximumDrawdown;
    }

    public double getBenchmarkMaximumDrawdown() {
        return benchmarkMaximumDrawdown;
    }
}