#pragma once
#include <Eigen/Dense>
#include "core/optimizer.hpp"
#include "core/result.hpp"
#include <stdexcept>

struct GDConfig
{
    double learning_rate{0.01}; // to be used as initial alpha is backtracking is disabled
    double tol{1e-6};
    int max_epochs{10000};
    bool history{false};
    bool backtracking{true};
    double c1{1e-4}; // Armijo constant
    double rho{0.5}; // Shrinkage factor
};

class GradientDescent : public Optimizer
{
    public:

        explicit GradientDescent(GDConfig config = {}) : config_{config} { }

        OptimResult optimize(ObjectiveFunction &f,
                             const Eigen::VectorXd &x_0) override;

    private:

        GDConfig config_;
};