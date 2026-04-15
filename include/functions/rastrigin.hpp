#pragma once
#include <Eigen/Dense>
#include <stdexcept>
#include <cmath>
#include <numbers>
#include "core/objective.hpp"

class Rastrigin : public ObjectiveFunction
{
    /*
    n-dim rastrigin function
    A = 10 (standard value)

    hessian is a diagonal matrix, off diagonal entries are 0
    rastrigin is separable across dimensions
    */
    public:

        explicit Rastrigin(int dims) : n_(dims)
        {
            if(dims < 1)
            {
                throw std::invalid_argument("Dimensions must be positive.");
            }
        }

        double evaluate(const Eigen::VectorXd &x) const override
        {
            if(n_ != x.size())
            {
                throw std::invalid_argument("Dimension mismatch!");
                // Dimension check to guard against mismatched start point @ runtime
            }

            double res{};

            res += A_ * n_;

            for(int i{}; i < n_; ++i)
            {
                res += std::pow(x[i], 2) - (A_ * std::cos(2.0 * std::numbers::pi * x[i]));
            }

            return res;
        }

        Eigen::VectorXd grad(const Eigen::VectorXd &x) const override
        {
            if(n_ != x.size())
            {
                throw std::invalid_argument("Dimension mismatch!");
            }
            
            Eigen::VectorXd grad_(n_);

            for(int i{}; i < n_; ++i)
            {
                grad_[i] = 2.0 * x[i] + (2.0 * std::numbers::pi * A_ * std::sin(2.0 * std::numbers::pi * x[i]));
            }

            return grad_;
        }

        Eigen::MatrixXd hessian(const Eigen::VectorXd &x) const override
        {
            if(n_ != x.size())
            {
                throw std::invalid_argument("Dimension mismatch!");
            }
            
            double constant{4.0 * std::pow(std::numbers::pi, 2) * A_};
            Eigen::MatrixXd hess_ = Eigen::MatrixXd::Zero(n_, n_);

            for(int i{}; i < n_; ++i)
            {
                hess_.diagonal()[i] = 2.0 + (constant * std::cos(2.0 * std::numbers::pi * x[i]));
            }

            return hess_;
        }

        int dim() const override
        {
            return n_;
        }

    private:

        int n_;
        static constexpr double A_ = 10.0; // amplitude of cos term
};