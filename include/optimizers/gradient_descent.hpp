#pragma once
#include <Eigen/Dense>
#include "core/optimizer.hpp"
#include "core/result.hpp"
#include <stdexcept>

struct GDConfig
{
    double learning_rate{0.01};
    double tol{1e-6};
    int max_epochs{10000};
    bool history{false};
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