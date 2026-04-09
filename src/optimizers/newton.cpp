#include <string>
#include <Eigen/Dense>
#include <vector>
#include <optimizers/newton.hpp>

OptimResult Newton::optimize(ObjectiveFunction &f,
                             const Eigen::VectorXd &x_0)
{
    OptimResult result;
    result.status = Status::MAX_EPOCHS;
    result.iterations = 0;
    Eigen::VectorXd x{x_0};
    Eigen::VectorXd grad_;
    Eigen::MatrixXd hess;

    for(int i{}; i < config_.max_epochs; ++i)
    {
        grad_ = f.grad(x);

        if(grad_.norm() < config_.tol)
        {
            result.status = Status::CONVERGED;
            result.message = std::string("Converged.");
            break;
        }

        double mu{config_.initial_mu};
        Eigen::MatrixXd hess_mod;
        Eigen::LLT<Eigen::MatrixXd> llt;
        hess = f.hessian(x);
        bool proceed{true};

        while(proceed)
        {
            hess_mod = hess + mu * Eigen::MatrixXd::Identity(hess.rows(), hess.cols());
            llt = hess_mod.llt();

            if(llt.info() == Eigen::Success)
            {
                break;
            }

            mu *= 2.0;

            if(mu >= 1e10)
            {
                result.status = Status::FAILED;
                result.message = std::string("Hessian modification failed.");
                proceed = false;
            }
        }

        if(!proceed)
        {
            break;
        }

        Eigen::VectorXd d = llt.solve(grad_);

        double alpha{1.0};
        double f_x = f.evaluate(x);
        double grad_sq = grad_.squaredNorm();

        while(f.evaluate(x - alpha * d) > f_x - config_.c1 * alpha * grad_sq)
        {
            alpha *= config_.rho;
            
            if(alpha < 1e-10)
            {
                break;
            }
        }

        x = x - alpha * d;

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