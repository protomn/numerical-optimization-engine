#include <string>
#include <vector>
#include <Eigen/Dense>
#include <random>
#include "optimizers/simulated_annealing.hpp"

OptimResult SA::optimize(ObjectiveFunction &f,
                         const Eigen::VectorXd &x_0)
{
    OptimResult result;
    result.converged = false;
    result.iterations = 0;
    Eigen::VectorXd x{x_0};
    double T = config_.initial_temp;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dist_2(0.0, 1.0);

    for(int i{}; i < config_.max_epochs; ++i)
    {
        //double current_step = config_.step_size * (T / config_.initial_temp);
        std::normal_distribution<double> dist_1(0.0, config_.step_size);

        if(T < config_.min_temp)
        {
            result.converged = true;
            result.message = std::string("Converged.");
            break;
        }

        Eigen::VectorXd pertubation_vect(x.size());

        for(int j{}; j < x.size(); ++j)
        {
            pertubation_vect[j] = dist_1(gen);
        }

        Eigen::VectorXd cand = x + pertubation_vect;

        auto f_cand = f.evaluate(cand);
        auto f_x = f.evaluate(x);
        auto delta_f = f_cand - f_x;

        double p{};

        if(delta_f < 0)
        {
            x = cand;
            f_x = f_cand;
        }

        else
        {
            p = std::exp(-delta_f / T);

            if(dist_2(gen) < p)
            {
                x = cand;
                f_x = f_cand;
            }
        }

        T = T * config_.cooling_rate;

        if(config_.history)
        {
            result.history.push_back(f.evaluate(x));
        }

        result.iterations += 1;
    }

    result.optimal_x = x;
    result.optimal_f = f.evaluate(x);

    if(!result.converged)
    {
        result.message = std::string("Stopped. Epochs exhausted.");
    }

    return result;
}