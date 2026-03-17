#pragma once
#include <Eigen/Dense>
#include "core/objective.hpp"
#include "core/result.hpp"
#include "core/optimizer.hpp"

struct SAConfig
{
    double initial_temp = 1000.0;
    double cooling_rate = 0.995;
    double min_temp = 1e-8;
    double step_size = 0.1;
    int max_epochs = 10000;
    double tol = 1e-6;
    bool history = false;
};

class SA : public Optimizer
{
    public:

        explicit SA(SAConfig config = {}) : config_(config) { }

        OptimResult optimize(ObjectiveFunction &f,
                             const Eigen::VectorXd &x_0) override;

    private:

        SAConfig config_;
};