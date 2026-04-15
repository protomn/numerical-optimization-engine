#pragma once

#include <Eigen/Dense>
#include "core/optimizer.hpp"
#include "core/result.hpp"
#include "core/objective.hpp"

struct LBFGSConfig
{
    /*
    config for l-fbgs optimizer
    */

    int m{10}; // history size, number of curvature pairs retained
    double tol{1e-6}; // covergence threshold
    int max_epochs{1000}; // max number of iterations the optimizer runs
    bool history{false}; // records f(x) at every epoch
    double c1{1e-4}; // sufficient decrease const
    double c2{0.1}; // curvature constant (0.1 for lbfgs, 0.9 for gd)
};

class LBFGS : public Optimizer
{
    /*
    l-fbgs optimizer; default for smooth, unconstrained convex problems

    search direction computed via two loop recursion
    hessian inverse approximated using the m most recent curvature pairs

    convergence is superlinear in practice on smooth problems
    */

    public:

        explicit LBFGS(LBFGSConfig config = {}) : config_(config) {}

        OptimResult optimize(ObjectiveFunction &f,
                             const Eigen::VectorXd &x_0) override;

    private:

        LBFGSConfig config_;
};