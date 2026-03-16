#pragma once
#include <Eigen/Dense>
#include "core/objective.hpp"
#include "core/result.hpp"
#include "core/optimizer.hpp"

struct HCConfig
{
    int max_epochs = 1000;
    double step_size = 0.01;
    double tol = 1e-6;
    bool history = false;
};

class HC : public Optimizer
{
    public:

        explicit HC(HCConfig config = {}) : config_(config) { }

        OptimResult optimize(ObjectiveFunction &f,
                             const Eigen::VectorXd &x_0) override;

    private:

            HCConfig config_;
};