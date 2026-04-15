#include <string>
#include <Eigen/Dense>
#include <vector>
#include "optimizers/newton.hpp"
#include "core/line_search.hpp"

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
        double f_x = f.evaluate(x);

        /*
        relative grad norm convergence, same as GD and L-BFGS for same
        behaviour across optimizers
        */
        if(grad_.norm() < config_.tol * std::max(1.0, std::abs(f_x)))
        {
            result.status = Status::CONVERGED;
            result.message = std::string("Converged.");
            break;
        }

        /*
        modified cholesky regularization
        if raw hessian is indef far from the optimum, add mu * I
        and double the value of mu until cholesky factorization succeeds,
        guaranteeing hessian matrix is PD.
        */

        double mu{config_.initial_mu};
        Eigen::MatrixXd hess_mod;
        Eigen::LLT<Eigen::MatrixXd> llt;
        hess = f.hessian(x);
        bool proceed{true};

        /*
        each iter doubles mu until LLT succeeds or shift is too large for
        the problem to be numerically stable.
        */
        while(proceed)
        {
            /*
            as mu grows, hess_mod grows and approaches mu * I and 
            d approaches the direction of steepest descent, safe fallback
            */
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

        /*
        solve for newton dirn d
        uses cholesky factor, no extra computation
        */
        Eigen::VectorXd d = llt.solve(grad_);

        double alpha{1.0};

        if(config_.backtracking)
        {
            /*
            damped newton step, full step can overshoot far from optimum
            wolfe line search scales the step size back whenever required.
            */

            WolfeConfig wc;
            wc.c1 = config_.c1;
            wc.c2 = config_.c2;
            alpha = lineSearch(f, x, -d, grad_, wc);
        }

        x = x - alpha * d; // update rule
        f_x = f.evaluate(x);

        if(config_.history)
        {
            result.history.push_back(f_x);
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