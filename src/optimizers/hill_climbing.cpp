#include <string>
#include <vector>
#include <Eigen/Dense>
#include <random>
#include <cmath>
#include "optimizers/hill_climbing.hpp"

OptimResult HC::optimize(ObjectiveFunction &f,
                         const Eigen::VectorXd &x_0)
{
    OptimResult result;
    result.status = Status::FAILED;
    result.iterations = 0;
    Eigen::VectorXd x{x_0};

    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<double> dist(0.0, config_.step_size);

    double f_prev{f.evaluate(x)};
    int stagnation_count{};

    for(int i{}; i < config_.max_epochs; ++i)
    {
        Eigen::VectorXd pertubation_vect(x.size());

        for(int j{}; j < x.size(); j++)
        {
            pertubation_vect[j] = dist(gen);
        }

        Eigen::VectorXd cand;
        
        cand = x + pertubation_vect;

        auto f_cand = f.evaluate(cand);

        if(f_cand < f_prev)
        {
            double improvement = std::abs(f_prev - f_cand);
            x = cand;
            f_prev = f_cand;
            stagnation_count = 0;

            if(improvement < config_.tol)
            {
                result.status = Status::CONVERGED;
                result.message = std::string("Converged.");
                break;
            }
        }

        else
        {
            stagnation_count++;

            if(stagnation_count >= config_.stagnation_limit)
            {
                result.status = Status::STAGNATED;
                result.message = std::string("Stopped. Stagnation limit reached.");
                break;
            }
        }

        if(config_.history)
        {
            result.history.push_back(f.evaluate(x));
        }

        result.iterations = i + 1;
    }

    result.optimal_x = x;
    result.optimal_f = f.evaluate(x);


    if(result.status == Status::FAILED)
    {
        result.status = Status::MAX_EPOCHS;
        result.message = std::string("Stopped. Epochs exhausted.");
    }

    return result;
}