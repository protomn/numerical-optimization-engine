#include <Eigen/Dense>
#include "optimizers/lbfgs.hpp"
#include "core/line_search.hpp"
#include <deque>
#include <string>
#include <vector>

static Eigen::VectorXd two_loop(
    const Eigen::VectorXd &grad,
    const std::deque<Eigen::VectorXd> &s_hist,
    const std::deque<Eigen::VectorXd> &y_hist)
{
    /*
    computes lfbgs search direction using 2 loop recursion
    
    when no history exists (1st epoch), return grad directly
    equivalent to steepest descent when hessian = I
    */

    auto k = s_hist.size();
    Eigen::VectorXd q{grad};
    std::vector<double> alpha(k);

    if(k == 0)
    {
        return grad;
    }

    for(int i{(int)k - 1}; i >= 0; --i)
    {
        /*
        forward pass loop
        compute and store alpha[i], then update q
        walk back through history (recent pairs first)
        */
        double rho_i = 1.0 / y_hist[i].dot(s_hist[i]);
        alpha[i] = rho_i * s_hist[i].dot(q);
        q = q - alpha[i] * y_hist[i];
    }

    /*
    scale the initial hess approximation using last curvature pair
    to ensure search direction has the right magnitude without
    storing the full hessian 
    */

    double gamma{1.0};
    if(k > 0)
    {
        gamma = s_hist.back().dot(y_hist.back()) /
                y_hist.back().dot(y_hist.back());
    }

    Eigen::VectorXd r = gamma * q;

    for(int i{}; i < (int)k; ++i)
    {
        /*
        backward pass loop
        compute beta_i then update r, the scaled quasi-newton direction
        */

        double rho_i = 1.0 / y_hist[i].dot(s_hist[i]);
        double beta_i = rho_i * y_hist[i].dot(r);
        r = r + s_hist[i] * (alpha[i] - beta_i);
    }

    return r;
}

OptimResult LBFGS::optimize(ObjectiveFunction &f,
                            const Eigen::VectorXd &x_0)
{
    OptimResult result;
    result.status = Status::MAX_EPOCHS;
    result.iterations = 0;
    Eigen::VectorXd x{x_0};
    Eigen::VectorXd grad_ = f.grad(x);

    std::deque<Eigen::VectorXd> s_history;
    std::deque<Eigen::VectorXd> y_history;

    Eigen::VectorXd x_prev{};
    Eigen::VectorXd grad_prev{};

    for(int i{}; i < config_.max_epochs; ++i)
    {
        double f_x = f.evaluate(x);

        if(grad_.norm() < config_.tol * std::max(1.0, std::abs(f_x)))
        {
            result.status = Status::CONVERGED;
            result.message = std::string("Converged.");
            break;
        }

        Eigen::VectorXd d = two_loop(grad_, s_history, y_history);

        if(d.dot(grad_) <= 0)
        {
            /*
            safety check
            if 2 loop direction is not a descent related vect
            fallback to the grad itself
            this can happen if curvature pairs become stale
            or history is empty
            */

            d = grad_;
        }

        WolfeConfig wc;
        wc.c2 = config_.c2;

        /*
        wolfe search with c2 = 0.1 (tight curvature)
        guarantees necessary condition is met for bfgs
        update to remain psd 
        */

        double alpha = lineSearch(f, x, -d, grad_, wc);

        if(!std::isfinite(alpha) || alpha < 1e-15)
        {
            /*
            line search fallback
            small fixed step allows epoch to continue and
            potentially recover at next step 
            */

            alpha = 1e-4;
        }

        x_prev = x;
        grad_prev = grad_;
        x = x - alpha * d;
        grad_ = f.grad(x);

        if(!x.allFinite() || !grad_.allFinite())
        {
            /*
            nan check
            detect numerical breakdown before storing corrupt curvature
            pairs
            */

            result.status = Status::FAILED;
            result.message = std::string("Numerical breakdown.");
            break;
        }

        Eigen::VectorXd s = x - x_prev;
        Eigen::VectorXd y = grad_ - grad_prev;

        if(s.dot(y) > 1e-8 * s.norm() * y.norm())
        {
            /*
            only store curvature pair if s.dot(y) is sufficiently 
            positive relative to the pair magnitudes. 
            near zero or negative curvatures indicate a bad step
            and can corrupt inv-hess approx
            */

            if((int)s_history.size() == config_.m)
            {
                /*
                circular buffer, discard oldest pair when history buffer is full
                */
                s_history.pop_front();
                y_history.pop_front();
            }

            s_history.push_back(s);
            y_history.push_back(y);
        }
        

        if(config_.history)
        {
            result.history.push_back(f.evaluate(x));
        }

        result.iterations += 1;
    }

    result.optimal_x = x;
    result.optimal_f = f.evaluate(x);

    if(result.status == Status::MAX_EPOCHS)
    {
        result.message = std::string("Stopped. Epochs exhausted.");
    }

    return result;
}