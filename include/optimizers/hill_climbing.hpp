#pragma once
#include <Eigen/Dense>
#include "core/objective.hpp"
#include "core/result.hpp"
#include "core/optimizer.hpp"

struct HCConfig
{
    /*
    config for stochastic hill climbing

    giverned by 2 independent stopping criteria:
        - improvement based: stop when accepted step improves f by less than tol
        - stagnation based: stop when no improvement found for <stagnation_limit> steps
    */

    int max_epochs{1000}; // max number of iters the optimizer will run
    double step_size{0.01}; // stddev of gaussian pertubation, larger the value
                            // bigger the tradeoff between exploration and exploitation
    double tol{1e-6}; // convergence threshold
    int stagnation_limit{100}; // number of consec rejected step before OptimResult::status == Status::STAGNATED
    bool history{false}; // records f(x) at every epoch and stores in a vector
};

class HC : public Optimizer
{
    /*
    stochastic hill-climbing optimizer

    has no probabilistic acceptance criteria
    works well on unimodal problems (eg. sphere function)
    */
    
    public:

        explicit HC(HCConfig config = {}) : config_(config) { }

        OptimResult optimize(ObjectiveFunction &f,
                             const Eigen::VectorXd &x_0) override;

    private:

            HCConfig config_;
};