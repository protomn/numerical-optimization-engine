#include <string>
#include <vector>
#include <Eigen/Dense>
#include <random>
#include "optimizers/simulated_annealing.hpp"

OptimResult SA::optimize(ObjectiveFunction &f,
                         const Eigen::VectorXd &x_0)
{
    OptimResult result;
    result.status = Status::MAX_EPOCHS;
    result.iterations = 0;
    Eigen::VectorXd x{x_0};
    double T = config_.initial_temp;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<double> dist_1(0.0, config_.step_size);
    std::uniform_real_distribution<double> dist_2(0.0, 1.0);

    double f_x = f.evaluate(x);
    Eigen::VectorXd best_x = x;
    double best_f{f_x};

    for(int i{}; i < config_.max_epochs; ++i)
    {
        //double current_step = config_.step_size * (T / config_.initial_temp);

        if(T < config_.min_temp)
        {
            result.status = Status::CONVERGED;
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
        auto delta_f = f_cand - f_x;

        double p{};

        if(delta_f < 0)
        {
            x = cand;
            f_x = f_cand;

            if(f_x < best_f)
            {
                best_f = f_x;
                best_x = x;
            }
        }

        else
        {
            double exp = -delta_f / T;
            p = (exp < -500.0) ? 0.0 : std::exp(exp);

            if(dist_2(gen) < p)
            {
                x = cand;
                f_x = f_cand;

                if(f_x < best_f)
                {
                    best_f = f_x;
                    best_x = x;
                }
            }
        }

        T = T * config_.cooling_rate;

        if(config_.history)
        {
            result.history.push_back(f.evaluate(x));
        }

        result.iterations += 1;
    }

    result.optimal_x = best_x;
    result.optimal_f = best_f;

    if(result.status == Status::MAX_EPOCHS)
    {
        result.message = std::string("Stopped. Epochs exhausted.");
    }

    return result;
}