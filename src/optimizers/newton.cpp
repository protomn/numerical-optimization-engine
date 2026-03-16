#include <string>
#include <Eigen/Dense>
#include <vector>
#include <optimizers/newton.hpp>

OptimResult Newton::optimize(ObjectiveFunction &f,
                             const Eigen::VectorXd &x_0)
{
    OptimResult result;
    result.converged = false;
    result.iterations = 0;
    Eigen::VectorXd x{x_0};
    Eigen::VectorXd grad_;
    Eigen::MatrixXd hess;

    for(int i{}; i < config_.max_epochs; ++i)
    {
        grad_ = f.grad(x);

        if(grad_.norm() < config_.tol)
        {
            result.converged = true;
            result.message = std::string("Converged.");
            break;
        }

        hess = f.hessian(x);

        auto ldlt = hess.ldlt();

        if(ldlt.info() != Eigen::Success)
        {
            result.message = std::string("Hessian decomposition failed.");
            break;
        }

        Eigen::VectorXd d = ldlt.solve(grad_);

        x = x - d;

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