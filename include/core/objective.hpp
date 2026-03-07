#pragma once
#include <Eigen/Dense>

class ObjectiveFunction
{
    public:

        virtual ~ObjectiveFunction() = default;

        virtual double evalutate(const Eigen::VectorXd &x) const = 0;
        virtual Eigen::VectorXd grad(const Eigen::VectorXd &x) const;
        virtual Eigen::MatrixXd hessian(const Eigen::VectorXd &x) const;
        virtual int dim() const = 0;
};
