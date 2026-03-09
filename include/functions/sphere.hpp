#pragma once
#include <Eigen/Dense>
#include <stdexcept>
#include "core/objective.hpp"

class Sphere : public ObjectiveFunction
{
    public: 

        explicit Sphere(int dims) : n_(dims)
        {
            if(dims <= 0)
            {
                throw std::invalid_argument("Dimensions must be positive.");
            }
        }

        double evaluate(const Eigen::VectorXd &x) const override
        {
            return x.squaredNorm();
        }

        Eigen::VectorXd grad(const Eigen::VectorXd &x) const override
        {
            return 2 * x;
        }

        Eigen::MatrixXd hessian(const Eigen::VectorXd &/*x*/) const override
        {
            return 2 * Eigen::MatrixXd::Identity(n_, n_);
        }

        int dim() const override
        {
            return n_;
        }

    private:

        int n_;

};