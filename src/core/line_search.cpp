#include "core/line_search.hpp"
#include <Eigen/Dense>
#include <cmath>
#include <algorithm>

static double cubic_interpolation(
    double a_lo, double a_hi,
    double f_lo, double f_hi,
    double g_lo, double g_hi
)
{
    /*
    fits a cubic ppolynomial through (a_lo, f_lo, g_lo) and (a_hi, f_hi, g_hi)
    and returns its minimizer a_star.
    used by zoom() to accelerate bracket reduction
    falls back to bisection if discriminant is negative or 
    polynomial is degen cubic (denominator is zero)
    result is clamped to [a_lo, a_hi] by caller (zoom())
    */
    
    /*
    cubic minimizer formula
    d1 and d2 are intermediates in closed form solution
    */
    double d1 = g_lo + g_hi - 3.0 * (f_lo - f_hi)/(a_lo - a_hi);
    double disc = (d1 * d1) - (g_lo * g_hi);
    
    if(disc < 0.0)
    {
        /*
        guard against complex roots
        */
        return (a_lo + a_hi) / 2.0;
    }

    double d2 = std::sqrt(disc);
    double denom = g_hi - g_lo + 2.0 * d2;

    if(std::abs(denom) < 1e-10)
    {
        /*
        guard against 0 denom
        */
        return (a_lo + a_hi) / 2.0;
    }

    double a_star = a_hi - (a_hi - a_lo) * (g_hi + d2 - d1) / denom;

    double a_min = std::min(a_lo, a_hi);
    double a_max = std::max(a_lo, a_hi);
    a_star = std::max(a_min, std::min(a_max, a_star));

    return a_star;
}

static double zoom(
    double a_lo, double a_hi,
    double phi_lo, double phi_hi,
    double dphi_lo, double dphi_hi,
    ObjectiveFunction &f,
    const Eigen::VectorXd &x,
    const Eigen::VectorXd &d,
    double phi_0, double dphi_0,
    double c1, double c2,
    double max_iters)
{
    /*
    called when bracket [a_lo, a_hi] is known to contain a wolfe point
    invariants:
        - a_lo satisfies armijo
        - algorithm descends at a_lo
        - a_hi is the worst endpoint

    narrows the bracket search via cubic_interpolation() until strong Wolfe
    conditions are satisfied or epochs are exhausted
    */

    for(int i{}; i < max_iters; ++i)
    {
        double a_j = cubic_interpolation(a_lo, a_hi, phi_lo, phi_hi, dphi_lo, dphi_hi);

        /*
        clamp trial point to inner 80% of the bracket to prevent
        near-boundary steps, may stall convergence
        */
        double margin{0.1 * (a_hi - a_lo)};
        double lo = std::min(a_lo, a_hi) + margin;
        double hi = std::max(a_lo, a_hi) - margin;

        if(lo < hi) 
        {
            a_j = std::clamp(a_j, lo, hi);
        }

        else 
        {
            a_j = (a_lo + a_hi) / 2.0;
        }

        double phi_j = f.evaluate(x + a_j * d);

        if((phi_j > phi_0 + c1 * a_j * dphi_0) || (phi_j >= phi_lo))
        {
            /*
            a_j violates armijo or is worse than a_lo
            shrink the ub
            */

            a_hi = a_j;
            phi_hi = phi_j;
        }

        else
        {
            /*
            armijo satisfied, check for strong curvature condition
            grad calc is deferred to avoid computational cost when armijo
            already fails
            */

            double dphi_j = f.grad(x + a_j * d).dot(d);

            if(std::abs(dphi_j) <= c2 * std::abs(dphi_0))
            {
                /*
                strong wolfe condition is met
                a_j is accepted step
                */
                return a_j;
            }

            if (dphi_j * (a_hi - a_lo) >= 0)
            {
                /*
                directional deriv points towards a_hi, move a_hi to a_lo
                preserve zoom invariant, then update a_lo
                */

                a_hi = a_lo;
                phi_hi = phi_lo;
            }

            a_lo = a_j;
            phi_lo = phi_j;
            dphi_lo = dphi_j;
        }
    }

    return a_lo;
}

double lineSearch(
    ObjectiveFunction &f,
    const Eigen::VectorXd &x,
    const Eigen::VectorXd &d,
    const Eigen::VectorXd &grad,
    WolfeConfig config_)
{
    /*
    bracket phase
    expands alpha until:
        - armijo is violated or phi increases
        - strong curvature condition met, returns alpha directly
        - directional deriv turns non-neg before updating a_lo
    alpha = 1.0 is accepted directly for well scaled problems.
    */

    double phi_0 = f.evaluate(x);
    double dphi_0 = grad.dot(d); // directional deriv along d at alpha = 0
                                 // must be negative for line search to work

    double alpha{1.0};
    double alpha_prev{};
    double phi_prev{phi_0};
    double dphi_prev{dphi_0};

    for(int i{}; i < config_.max_epochs; ++i)
    {
        double phi_a = f.evaluate(x + alpha * d);

        if((phi_a > phi_0 + config_.c1 * alpha * dphi_0) || ((phi_a >= phi_prev) && (i > 0)))
        {
            /*
            armijo violated or phi increased from previous alpha - a wolfe point exists
            in [a_prev, alpha], delegate zoom() to find it
            */

            double dphi_a = f.grad(x + alpha * d).dot(d);
            return zoom(alpha_prev, alpha, 
                        phi_prev, phi_a, 
                        dphi_prev, dphi_a, 
                        f, x, d, phi_0, dphi_0,
                        config_.c1, config_.c2, config_.max_epochs);
        }

        double dphi_a = f.grad(x + alpha * d).dot(d);

        if(std::abs(dphi_a) <= config_.c2 * std::abs(dphi_0))
        {
            /*
            strong curvature condition satisfied, return alpha without entering zoom
            */
            return alpha;
        }

        if(dphi_a >= 0)
        {
            /*
            directional deriv has turned non-neg, min is between a_prev and alpha
            zoom with arguments reversed
            */

            return zoom(alpha, alpha_prev,
                        phi_a, phi_prev,
                        dphi_a, dphi_prev, 
                        f, x, d, phi_0, dphi_0,
                        config_.c1, config_.c2, config_.max_epochs);
        }

        alpha_prev = alpha;
        phi_prev = phi_a;
        dphi_prev = dphi_a;
        alpha = std::min(2.0 * alpha, config_.alpha_max); // neither condition triggered, expand bracket by doubling alpha
                                                          // alpha capped at alpha_max, continues searching for a wolfe interval
    }

    return alpha;
}