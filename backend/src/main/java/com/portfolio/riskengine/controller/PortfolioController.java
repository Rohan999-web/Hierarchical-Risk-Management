package com.portfolio.riskengine.controller;

import java.io.IOException;
import java.util.List;

import java.util.Map;
import org.springframework.web.bind.annotation.CrossOrigin;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import com.portfolio.riskengine.model.PortfolioMetrics;
import com.portfolio.riskengine.service.PortfolioService;

@RestController
@CrossOrigin(origins = "http://localhost:5173")
@RequestMapping("/api/portfolio")
public class PortfolioController {

    private final PortfolioService portfolioService;

    public PortfolioController(PortfolioService portfolioService) {
        this.portfolioService = portfolioService;
    }

    @GetMapping("/health")
    public String health() {
        return "Portfolio Risk Engine API is running";
    }

    @GetMapping("/metrics")
    public PortfolioMetrics metrics() throws IOException {
        return portfolioService.getMetrics();
    }

    @GetMapping("/performance")
    public List<String[]> performance() throws IOException {
        return portfolioService.getPerformance();
    }
    @GetMapping("/allocation")
    public Map<String, Double> allocation() throws IOException {
        return portfolioService.getLatestAllocation();
}
}