package com.portfolio.riskengine.service;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import org.springframework.stereotype.Service;

import com.portfolio.riskengine.model.PortfolioMetrics;

@Service
public class PortfolioService {

    private static final Path METRICS_FILE =
            Path.of("..", "final_hrp_nn_metrics.csv");

    private static final Path PERFORMANCE_FILE =
            Path.of("..", "final_hrp_nn_performance.csv");

    private static final Path WEIGHTS_FILE =
            Path.of("..", "final_hrp_nn_weights.csv");


    // =========================================================
    // PORTFOLIO METRICS
    // =========================================================

    public PortfolioMetrics getMetrics() throws IOException {

        List<String> lines = Files.readAllLines(METRICS_FILE);

        double hrpCumulativeReturn = 0;
        double benchmarkCumulativeReturn = 0;

        double hrpAnnualizedReturn = 0;
        double benchmarkAnnualizedReturn = 0;

        double hrpVolatility = 0;
        double benchmarkVolatility = 0;

        double hrpSharpeRatio = 0;
        double benchmarkSharpeRatio = 0;

        double hrpMaximumDrawdown = 0;
        double benchmarkMaximumDrawdown = 0;


        for (String line : lines) {

            String[] parts = line.split(",");

            if (parts.length < 2) {
                continue;
            }

            String metric = parts[0].trim();

            // Skip CSV header
            if (metric.isEmpty()
                    || parts[1].trim().equalsIgnoreCase("Value")) {
                continue;
            }

            double value = Double.parseDouble(parts[1].trim());


            switch (metric) {

                case "HRP Cumulative Return" ->
                        hrpCumulativeReturn = value;

                case "Benchmark Cumulative Return" ->
                        benchmarkCumulativeReturn = value;

                case "HRP Annualized Return" ->
                        hrpAnnualizedReturn = value;

                case "Benchmark Annualized Return" ->
                        benchmarkAnnualizedReturn = value;

                case "HRP Annualized Volatility" ->
                        hrpVolatility = value;

                case "Benchmark Annualized Volatility" ->
                        benchmarkVolatility = value;

                case "HRP Sharpe Ratio" ->
                        hrpSharpeRatio = value;

                case "Benchmark Sharpe Ratio" ->
                        benchmarkSharpeRatio = value;

                case "HRP Maximum Drawdown" ->
                        hrpMaximumDrawdown = value;

                case "Benchmark Maximum Drawdown" ->
                        benchmarkMaximumDrawdown = value;
            }
        }


        return new PortfolioMetrics(
                hrpCumulativeReturn,
                benchmarkCumulativeReturn,
                hrpAnnualizedReturn,
                benchmarkAnnualizedReturn,
                hrpVolatility,
                benchmarkVolatility,
                hrpSharpeRatio,
                benchmarkSharpeRatio,
                hrpMaximumDrawdown,
                benchmarkMaximumDrawdown
        );
    }


    // =========================================================
    // PORTFOLIO PERFORMANCE
    // =========================================================

    public List<String[]> getPerformance() throws IOException {

        List<String> lines = Files.readAllLines(PERFORMANCE_FILE);

        return lines.stream()
                .skip(1)
                .filter(line -> !line.trim().isEmpty())
                .map(line -> line.split(","))
                .toList();
    }


    // =========================================================
    // LATEST SECTOR ALLOCATION
    // =========================================================

    public Map<String, Double> getLatestAllocation() throws IOException {

        List<String> lines = Files.readAllLines(WEIGHTS_FILE);

        if (lines.size() < 2) {
            throw new IOException("Portfolio weights file is empty.");
        }

        // First row contains column names
        String[] headers = lines.get(0).split(",");

        // Last row contains the latest portfolio allocation
        String[] values = lines.get(lines.size() - 1).split(",");

        Map<String, Double> allocation = new LinkedHashMap<>();

        for (int i = 1; i < headers.length; i++) {

            String sector = headers[i].trim();

            double weight = Double.parseDouble(values[i].trim());

            allocation.put(sector, weight);
        }

        return allocation;
    }
}