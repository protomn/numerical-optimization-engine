#pragma once
#include "core/objective.hpp"
#include <Eigen/Dense>

struct WolfeConfig
{
    /*
    config for wolfe line search
    tradeoff between step-quality and cost controlled by
        - armijo condition
        - curvature

    (c1 < c2) should always be true.
    */

    double c1{1e-4}; // sufficient decrease constant (Armijo)
    double c2{0.9}; // curvature condition constant (0.9 for GD, Newton; 0.1 for L-BFGS)
    double alpha_max{1.0}; // ub for step size.
    int max_epochs{100}; // max iterations
};

double lineSearch(ObjectiveFunction &f,
                  const Eigen::VectorXd &x,
                  const Eigen::VectorXd &d, // direction of search
                  const Eigen::VectorXd &grad, // gradient at x
                  WolfeConfig config = {});
                /*
                finds step that satisfies strong wolfe conditions via more-thuente bracket & zoom
                algorithm. 
                bracket: expands alpha until a wolfe satisfying interval is found
                zoom: narrows the interval via cubic_interpolation until both conditions are met
                or epochs are exhausted.

                returns accepted step length alpha.
                */