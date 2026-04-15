#pragma once
#include <Eigen/Dense>
#include "objective.hpp"
#include "result.hpp"

class Optimizer
{
        /*
        Abstract base class for all optimizers.
        All optimizers have their respective implementations.
        They inherit from this base class and implement their own optimize.
        */
    public:

        virtual ~Optimizer() = default;

        virtual OptimResult optimize(
                ObjectiveFunction &f,
                const Eigen::VectorXd &x_0
        ) = 0;
        /*
        Minimizes f from a starting point x_0.
        Returns an OptimResult struct containing result details.
        f is a non-const ref as line search evaluates f at every candidate point.
        */
};