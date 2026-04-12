#pragma once
#include "core/objective.hpp"
#include <Eigen/Dense>

struct WolfeConfig
{
    double c1{1e-4};
    double c2{0.9};
    double alpha_max{1.0};
    int max_epochs{100};
};

double lineSearch(ObjectiveFunction &f,
                  const Eigen::VectorXd &x,
                  const Eigen::VectorXd &d, // direction of search
                  const Eigen::VectorXd &grad, // gradient at x
                  WolfeConfig config = {});