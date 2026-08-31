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

    /*
    config_.seed < 0 draws a non-deterministic seed, so each run gives a
    different markov chain, as needed for stochastic global optimization.

    a non-negative seed makes the chain reproducible.
    */
    std::mt19937 gen(config_.seed < 0
                         ? std::random_device{}()
                         : static_cast<std::mt19937::result_type>(config_.seed));
    std::normal_distribution<double> dist_1(0.0, config_.step_size);
    std::uniform_real_distribution<double> dist_2(0.0, 1.0);

    double f_x = f.evaluate(x); // track best solution seen across the entire run
                                // chain may accept worse solutions
    Eigen::VectorXd best_x{x};
    double best_f{f_x};

    for(int i{}; i < config_.max_epochs; ++i)
    {
        //double current_step = config_.step_size * (T / config_.initial_temp);

        if(T < config_.min_temp)
        {
            /*
            primary stopping criterion, temp cooled below min_temp
            acceptance probability for any worsening move is virtually 0
            search converges to local exploitation
            */
            result.status = Status::CONVERGED;
            result.message = std::string("Converged.");
            break;
        }

        Eigen::VectorXd pertubation_vect(x.size()); // sample candidate vector

        for(int j{}; j < x.size(); ++j)
        {
            pertubation_vect[j] = dist_1(gen);
        }

        Eigen::VectorXd cand = x + pertubation_vect;

        auto f_cand = f.evaluate(cand);
        auto delta_f = f_cand - f_x;
        /*
        delta_f < 0: improvement, accept
        delta_f >= 0: worsening, accept based on probability
        high temp -> all moves accepted (exploration)
        low temp -> worsening moves are rejected (exploitation)
        */

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
            double exp = -delta_f / T; // to clamp exp value to avoid overflow
                                       // double overflows for x > ~600
                                       // guard against very large magnitudes when temp approches min_temp
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

        T = T * config_.cooling_rate; // cooling update rule

        if(config_.history)
        {
            result.history.push_back(f.evaluate(x));
        }

        result.iterations += 1;
    }

    // return best seen, not last position in the chain
    result.optimal_x = best_x;
    result.optimal_f = best_f;

    if(result.status == Status::MAX_EPOCHS)
    {
        result.message = std::string("Stopped. Epochs exhausted.");
    }

    return result;
}