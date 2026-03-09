#include <stdexcept>
#include <string>
#include <Eigen/Dense>
#include <iostream>
#include <vector>
#include "optimizers/gradient_descent.hpp"

OptimResult GradientDescent::optimize(ObjectiveFunction &f,
                          const Eigen::VectorXd &x_0)
{
    OptimResult result;
    result.converged = false;
    result.iterations = 0;
    Eigen::VectorXd x{x_0};
    Eigen::VectorXd grad_;

    for(int i{}; i < config_.max_epochs; ++i)
    {
        grad_ = f.grad(x);

        if(grad_.norm() < config_.tol)
        {
            result.converged = true;
            result.message = std::string("Converged.");
            break;
        }

        x = x - config_.learning_rate * grad_;

        if(config_.history)
        {
            result.history.push_back(f.evaluate(x));
        }

        result.iterations = i + 1;
    }

    result.optimal_x = x;
    result.optimal_f = f.evaluate(x);

    if(!result.converged)
    {
        result.message = std::string("Stopped. Epochs exhausted.");
    }

    return result;
}