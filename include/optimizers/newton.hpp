#pragma once
#include <Eigen/Dense>
#include "core/objective.hpp"
#include "core/result.hpp"
#include "core/optimizer.hpp"

struct NewtonConfig
{
    /*
    config for newton descent optimizer
    method has modified cholesky and wolfe line search
    
    diagonal shift implemented to regularize indefinite hessians
    forces hessians to be pos-def
    */

    double tol{1e-6}; // convergence tolerance threshold
    int max_epochs{1000}; // max number of iterations the optimizer will run, if not converged.
    bool history{false}; // records f(x) at each epoch and stores in a vector
    double c1{1e-4}; // wolfe sufficient decrease const
    double c2{0.9}; // curvature const
    double initial_mu{1e-4}; // initial diagonal shift, only when hess is indef/singular
    bool backtracking{true}; // if true, uses wolfe search to scale newton step
};

class Newton : public Optimizer
{
    /*
    second ord newton optimizer with modified cholesky regularization
    d_n = (hess(f(x_n)) + mu*I)^(-1) * grad(f(x_n))
    x_{n + 1} = x_n - alpha_n * d_n

    hess is guaranteed to be positive def, if not numerical method fails.

    converges quadratically near optimum, faster than gd/l-bfgs on smooth
    well conditioned problems.

    requires finite-definite hess, very expensive per epoch for high
    dimensional problems.
    */

    public:

        explicit Newton(NewtonConfig config = {}) : config_(config) { }

        OptimResult optimize(ObjectiveFunction &f,
                             const Eigen::VectorXd &x_0) override;

    private:

            NewtonConfig config_;
};