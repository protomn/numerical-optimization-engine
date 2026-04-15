#pragma once
#include <vector>
#include <string>
#include <Eigen/Dense>

enum class Status
{
    /*
    Reason for terminating optimization loop.
    To be checked by callers before trusting optimal_x/optimal_f
    */

    CONVERGED, // Improvement is below tolerance
    STAGNATED, // no Improvement for N steps, stagnation limit hit.
    MAX_EPOCHS, // Epochs exhausted before convergence
    FAILED // numerical failure (NaN/Inf)
};
struct OptimResult
{
    /*
    result to be returned by Optimizer::optimize() 
    optimal_x/optimal_f always have a value
    history vector populated only of <Optimizer>Config::history == true.
    */

    Eigen::VectorXd optimal_x; // Optimal point
    double optimal_f; // Value of function at optimal point
    int iterations; // Number of iterations that ran
    Status status; // Check for stopping condition
    std::string message; // string description of stopping condition
    std::vector<double> history; // f(x) at every epoch
};

