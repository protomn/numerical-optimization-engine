#include <stdexcept>
#include <string>
#include <Eigen/Dense>
#include <iostream>
#include <vector>
#include "optimizers/gradient_descent.hpp"
#include "core/line_search.hpp"

OptimResult GradientDescent::optimize(ObjectiveFunction &f,
                          const Eigen::VectorXd &x_0)
{
    OptimResult result;
    result.status = Status::MAX_EPOCHS;
    result.iterations = 0;
    Eigen::VectorXd x{x_0};
    Eigen::VectorXd grad_;

    for(int i{}; i < config_.max_epochs; ++i)
    {
        grad_ = f.grad(x);

        if(grad_.norm() < config_.tol)
        {
            result.status = Status::CONVERGED;
            result.message = std::string("Converged.");
            break;
        }

        double alpha{1.0};

        if(config_.backtracking)
        {
            WolfeConfig wc;
            wc.c1 = config_.c1;
            wc.c2 = config_.c2;
            alpha = lineSearch(f, x, -grad_, grad_, wc);
        }

        else
        {
            alpha = config_.learning_rate;
        }
        
        x = x - alpha * grad_;

        if(config_.history)
        {
            result.history.push_back(f.evaluate(x));
        }

        result.iterations = i + 1;
    }

    result.optimal_x = x;
    result.optimal_f = f.evaluate(x);

    if(result.status == Status::MAX_EPOCHS)
    {
        result.message = std::string("Stopped. Epochs exhausted.");
    }

    return result;
}