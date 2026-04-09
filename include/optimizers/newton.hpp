#pragma once
#include <Eigen/Dense>
#include "core/objective.hpp"
#include "core/result.hpp"
#include "core/optimizer.hpp"

struct NewtonConfig
{
    double tol{1e-6};
    int max_epochs{1000};
    bool history{false};
    double c1{1e-4}; // Armijo constant
    double rho{0.5}; // shrinkage factor
    double initial_mu{1e-4}; // initial diagonal shift
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