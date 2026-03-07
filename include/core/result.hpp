#pragma once
#include <vector>
#include <string>
#include <Eigen/Dense>

struct OptimResult
{
    Eigen::VectorXd optimal_x; // Optimal point
    double optimal_f; // Value of function at optimal point
    int iterations; // Number of iterations that ran
    bool converged; // Check if stopping criteria is satisfied
    std::string message; // Status
    std::vector<double> history; // Value of function at each point
};

