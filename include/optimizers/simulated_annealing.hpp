#pragma once
#include <Eigen/Dense>
#include "core/objective.hpp"
#include "core/result.hpp"
#include "core/optimizer.hpp"

struct SAConfig
{
    /*
    config for simulated annealing (dual_annealing in scipy)

    behaves like pure hill-climb when temperature = 0
    tracks best position seen across entire markov chain
    */

    double initial_temp{1000.0}; // starting temperature
    double cooling_rate{0.995}; // decay factor for temperature
                                // slower cooling, more exploration
    double min_temp{1e-8}; // temp floor, is T < min_temp, Optimizer::stats == Status::CONVERGED
    double step_size{0.1}; // stddev of gaussian pertub
    int max_epochs{10000}; // max number of iterations the optimizer runs
    [[maybe_unused]] double tol{1e-6}; // unused for SA, task taken up by min_temp
    bool history{false}; // records f(x) at every epoch
};

class SA : public Optimizer
{
    /*
    simulated annealing optimizer, global search, no derivatives
    best for non-convex problems with several local optima
    
    return best point across all iterations.
    */

    public:

        explicit SA(SAConfig config = {}) : config_(config) { }

        OptimResult optimize(ObjectiveFunction &f,
                             const Eigen::VectorXd &x_0) override;

    private:

        SAConfig config_;
};