#pragma once
#include <vector>
#include <string>
#include <Eigen/Dense>

enum class Status
{
    CONVERGED, // Improvement is below tolerance
    STAGNATED, // no Improvement for N steps
    MAX_EPOCHS, // Epochs exhausted before convergence
    FAILED // numerical failure
};
struct OptimResult
{
    Eigen::VectorXd optimal_x; // Optimal point
    double optimal_f; // Value of function at optimal point
    int iterations; // Number of iterations that ran
    Status status; // Check for stopping condition
    std::string message; // Status
    std::vector<double> history; // Value of function at each point
};

