#pragma once
#include <Eigen/Dense>
#include "core/optimizer.hpp"
#include "core/result.hpp"
#include <stdexcept>

struct GDConfig
{
    /*
    Config for gradient descent
    when backtrackign is enables (default), the learning_rate param is unused.
    step size calculated using wolfe line search at every epoch
    disable backtracting for fixed step results. 
    */

    double learning_rate{0.01}; // to be used as initial alpha if backtracking is disabled
    double tol{1e-6}; // convergence threshold on grad.norm()
    int max_epochs{10000}; // max number of iterations to run the optimization loop
    bool history{false}; // records f(x) at each epoch in a vector
    bool backtracking{true}; // if true, uses wolfe line search instead of fixed step
    double c1{1e-4}; // sufficient decrease constant (only when backtracking = true)
    double c2{0.9}; //wolfe curvature constant (only when backtracking = true)
};

class GradientDescent : public Optimizer
{
    public:

        /*
        first order grad descent optimizer
        x_{n + 1} = x_n - alpha_n * grad(f(x_n))

        with backtracking enabled, step size chosen such that strong wolfe
        conditions are satisfied.

        convergence is checked on gradient norm
        optimizer converges slowly on ill-conditioned problems (non-convex)
        */

        explicit GradientDescent(GDConfig config = {}) : config_{config} { }

        OptimResult optimize(ObjectiveFunction &f,
                             const Eigen::VectorXd &x_0) override;

    private:

        GDConfig config_;
};