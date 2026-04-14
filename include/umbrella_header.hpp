#pragma once

// Core headers
#include <core/objective.hpp>
#include <core/optimizer.hpp>
#include <core/result.hpp>
#include <core/finite_difference.hpp>
#include <core/line_search.hpp>

// Function headers
#include <functions/rosenbrock.hpp>
#include <functions/sphere.hpp>
#include <functions/rastrigin.hpp>

// Optimizer headers
#include <optimizers/gradient_descent.hpp>
#include <optimizers/hill_climbing.hpp>
#include <optimizers/newton.hpp>
#include <optimizers/simulated_annealing.hpp>
#include <optimizers/lbfgs.hpp>