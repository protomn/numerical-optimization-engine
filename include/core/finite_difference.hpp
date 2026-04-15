#pragma once

#include <stdexcept>
#include <Eigen/Dense>
#include "core/objective.hpp"

class FiniteDiffWrapper : public ObjectiveFunction
{
    /*
    wraps any ObjectiveFunction to numerically approximate derivatives (grad and hess)
    useful when analytical grad()/hessian() is computationally expensive
    
    grad() uses central difference, 2n evaluations per call
    hessian() diag uses second difference, 2n + 1 evals per call
    hessian() off diag uses 4-pt mixed formula, 4 evals per pair

    f_ is held by reference, caller must ensure wrapped object outlives the wrapper object.
    */

    public:

        explicit FiniteDiffWrapper(ObjectiveFunction &f, double h = 1e-5) : f_(f), h_(h) {}

        double evaluate(const Eigen::VectorXd &x) const override
        {
            return f_.evaluate(x);
        }

        Eigen::VectorXd grad(const Eigen::VectorXd &x) const override
        {
            /*
            approximates 1st derivative via central differences.
            */

            size_t dim = x.size();

            Eigen::VectorXd grad_(dim);

            for(size_t i{}; i < dim; ++i)
            {
                Eigen::VectorXd x_plus = x;
                Eigen::VectorXd x_minus = x;

                x_plus(i) += h_;
                x_minus(i) -= h_;

                double f_plus = f_.evaluate(x_plus);
                double f_minus = f_.evaluate(x_minus);

                grad_(i) = (f_plus - f_minus) / (2.0 * h_);
            }

            return grad_;
        }

        Eigen::MatrixXd hessian(const Eigen::VectorXd &x) const override
        {
            /*
            approximates second deriv via finite difference
            computationally very expensive
            symmetry hardcoded
            only called by newton's descent.
            */

            size_t dim = x.size();
            Eigen::MatrixXd hess(dim, dim);

            double f_x = f_.evaluate(x);

            for(size_t i{}; i < dim; ++i)
            {
                for(size_t j{i}; j < dim; ++j)
                {
                    if(i == j)
                    {
                        Eigen::VectorXd x_plus = x;
                        Eigen::VectorXd x_minus = x;

                        x_plus(i) += h_;
                        x_minus(i) -= h_;

                        double f_plus = f_.evaluate(x_plus);
                        double f_minus = f_.evaluate(x_minus);

                        hess(i, i) = (f_plus - (2.0 * f_x) + f_minus) / (h_ * h_);
                    }

                    else
                    {
                        Eigen::VectorXd x_pp = x;
                        Eigen::VectorXd x_pm = x;
                        Eigen::VectorXd x_mp = x;
                        Eigen::VectorXd x_mm = x;

                        x_pp(i) += h_;
                        x_pp(j) += h_;

                        x_pm(i) += h_;
                        x_pm(j) -= h_;

                        x_mp(i) -= h_;
                        x_mp(j) += h_;

                        x_mm(i) -= h_;
                        x_mm(j) -= h_;

                        double f_pp = f_.evaluate(x_pp);
                        double f_pm = f_.evaluate(x_pm);
                        double f_mp = f_.evaluate(x_mp);
                        double f_mm = f_.evaluate(x_mm);

                        hess(i, j) = (f_pp - f_pm - f_mp + f_mm) / (4.0 * h_ * h_);
                        hess(j, i) = hess(i, j); 
                    }
                }
            }

            return hess;
        }

        int dim() const override
        {
            return f_.dim();
        }

    private:

        ObjectiveFunction &f_;
        double h_; // finite difference step size
};