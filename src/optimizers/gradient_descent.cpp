#include <stdexcept>
#include <string>
#include <Eigen/Dense>
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
        double f_x = f.evaluate(x);

        /*
        relative gradient norm: scale tol by |f(x)| so convergence
        behaves consistently across functions w different magnitudes.
        */
        if(grad_.norm() < config_.tol * std::max(1.0, std::abs(f_x)))
        {
            result.status = Status::CONVERGED;
            result.message = std::string("Converged.");
            break;
        }

        double alpha{1.0};

        /*
        wolfe line search to find a step size satisfying sufficient decrease and curvature
        conditions, prevents divergence on ill-conditions surfaces

        fixed learning rate fall back for debugging, initial implementation, iterated 3 times
        fixed step -> armijo -> wolfe
        */
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
        
        // steppest descent step, step size either from line_search or config, depending on
        // backtracking conditional
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