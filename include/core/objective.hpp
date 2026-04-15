#pragma once
#include <Eigen/Dense>

class ObjectiveFunction
{
    /*
    Abstract base class for objective that is to be minimized.
    Concrete functions (Sphere, Rosenbrock, Rastrigin) inherit from
    this and provide analytical derivatives.
    Functions without analystical derivatives use FiniteDiffWrapper
    */
    public:

        virtual ~ObjectiveFunction() = default;

        virtual double evaluate(const Eigen::VectorXd &x) const = 0;
        /*
        Returns f(x). Defined for all x in the domain.
        */

        virtual Eigen::VectorXd grad(const Eigen::VectorXd &x) const = 0;
        /*
        Returns grad as column vector (dim(), )
        Dimension must be consistent with evaluate().
        */

        virtual Eigen::MatrixXd hessian(const Eigen::VectorXd &x) const = 0;
        /*
        Returns the hessian matrix of dimension (dim(), dim()).
        Required for Newton Descent, not called by L-BFGS or GD.
        */

        virtual int dim() const = 0;
        /*
        Returns dimensionality of input space.
        Used by optimizer to validate starting point size.
        */
};
