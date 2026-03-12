#pragma once
#include <Eigen/Dense>
#include "core/objective.hpp"
#include "core/result.hpp"
#include "core/optimizer.hpp"

struct NewtonConfig
{
    double tol = 1e-6;
    int max_epochs = 1000;
    bool history = false;
};

class Newton : public Optimizer
{
    public:

        explicit Newton(NewtonConfig config = {}) : config_(config) { }

        OptimResult optimize(ObjectiveFunction &f,
                             const Eigen::VectorXd &x_0) override;

    private:

            NewtonConfig config_;
};