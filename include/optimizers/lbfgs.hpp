#pragma once

#include <Eigen/Dense>
#include "core/optimizer.hpp"
#include "core/result.hpp"
#include "core/objective.hpp"

struct LBFGSConfig
{
    int m{10};
    double tol{1e-6};
    int max_epochs{1000};
    bool history{false};
    double c1{1e-4};
    double c2{0.1};
};

class LBFGS : public Optimizer
{
    public:

        explicit LBFGS(LBFGSConfig config = {}) : config_(config) {}

        OptimResult optimize(ObjectiveFunction &f,
                             const Eigen::VectorXd &x_0) override;

    private:

        LBFGSConfig config_;
};