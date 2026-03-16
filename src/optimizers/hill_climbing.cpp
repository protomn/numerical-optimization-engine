#include <string>
#include <vector>
#include <Eigen/Dense>
#include <random>
#include "optimizers/hill_climbing.hpp"

OptimResult HC::optimize(ObjectiveFunction &f,
                         const Eigen::VectorXd &x_0)
{
    OptimResult result;
    result.converged = false;
    result.iterations = 0;
    Eigen::VectorXd x{x_0};

    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<double> dist(0.0, config_.step_size);

    
    for(int i{}; i < config_.max_epochs; ++i)
    {
        Eigen::VectorXd pertubation_vect(x.size());

        for(int j{}; j < x.size(); j++)
        {
            pertubation_vect[j] = dist(gen);
        }

        Eigen::VectorXd cand;
        
        cand = x + pertubation_vect;

        auto f_curr = f.evaluate(x);
        auto f_cand = f.evaluate(cand);

        if(f_cand < f_curr)
        {
            x = cand;
            f_curr = f_cand;

            if(f_curr < config_.tol)
            {
                result.converged = true;
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


    if(!result.converged)
    {
        result.message = std::string("Stopped. Epochs exhausted.");
    }

    return result;
}