#pragma once
#include <Eigen/Dense>
#include <stdexcept>
#include "core/objective.hpp"

class Rosenbrock : public ObjectiveFunction
{
    /*
    2d rosenbrock functions, non-convex
    f(x, y) = (a - x)^2 + b(y - x^2)^2, where a = 1, b = 100
    global minimum @ (0, 0)

    handled well by newton and l-bfgs, gd struggles
    */

    public:

        explicit Rosenbrock(int dims) : n_(dims)
        {
            if(dims != 2)
            {
                throw std::invalid_argument("Rosenbrock is 2D.");
            }
        }

        double evaluate(const Eigen::VectorXd &x) const override
        {
            double t{x(1) - x(0) * x(0)}; // cached to avoid recomputation 
            return (1 - x(0)) * (1 - x(0)) + 100 * t * t;
        }

        Eigen::VectorXd grad(const Eigen::VectorXd &x) const override
        {
            Eigen::VectorXd g(2);
            double t{x(1) - x(0) * x(0)}; // cached
            g(0) = -2 * (1-x(0)) - 400 * x(0) * t;
            g(1) = 200 * t;

            return g;
        }

        Eigen::MatrixXd hessian(const Eigen::VectorXd &x) const override
        {
            Eigen::MatrixXd h(2, 2);
            h(0, 0) = 2 - (400 * x(1)) + (1200 * x(0) * x(0));
            h(0, 1) = -400 * x(0);
            h(1, 0) = - 400 * x(0);
            h(1, 1) = 200;

            return h;
        }

        int dim() const override
        {
            return 2;
        }

    private:
        
        [[maybe_unused]]int n_;
};