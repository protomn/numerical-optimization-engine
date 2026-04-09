#pragma once

#include <stdexcept>
#include <Eigen/Dense>
#include "core/objective.hpp"

class FiniteDiffWrapper : public ObjectiveFunction
{
    public:

        explicit FiniteDiffWrapper(ObjectiveFunction &f, double h = 1e-5) : f_(f), h_(h) {}

        double evaluate(const Eigen::VectorXd &x) const override
        {
            return f_.evaluate(x);
        }

        Eigen::VectorXd grad(Eigen::VectorXd &x) const override
        {
            size_t dim{x.size()};

            Eigen::VectorXd grad_(dim);

            
            double f_x = f_.evaluate(x);

            for(int i{}; i < dim; ++i)
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
            size_t dim{x.size()};
            Eigen::MatrixXd hess(dim, dim);

            double f_x = f_.evaluate(x);

            for(int i{}; i < dim; ++i)
            {
                for(int j{i}; j < dim; ++j)
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
        double h_;
};