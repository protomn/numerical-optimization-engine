#include <iostream>
#include <chrono>
#include <iomanip>
#include <string>
#include <Eigen/Dense>
#include "umbrella_header.hpp"

template<typename OptimizerType, typename FunctionType>
void run_benchmark(const std::string &name,
                   const std::string &function_name,
                   OptimizerType &optimizer,
                   FunctionType &function,
                   const Eigen::VectorXd &x0,
                   int runs = 10)
{
    double total_ms{};
    OptimResult result;

    for(int i{}; i < runs; ++i)
    {
        auto start = std::chrono::high_resolution_clock::now();
        result = optimizer.optimize(function, x0);
        auto end = std::chrono::high_resolution_clock::now();
        total_ms += std::chrono::duration<double, std::milli>(end - start).count();
    }

    double avg_ms{total_ms / runs};

    std::cout << std::left
              << std::setw(22) << name
              << std::setw(15) << function_name
              << std::setw(12) << std::fixed << std::setprecision(3) << avg_ms
              << std::setw(10) << result.iterations
              << std::setw(12) << std::scientific << std::setprecision(2) << result.optimal_f
              << std::setw(12) << (result.status == Status::CONVERGED ? "CONVERGED" :
                                   result.status == Status::STAGNATED ? "STAGNATED" :
                                   result.status == Status::FAILED ? "FAILED" : "MAX_EPOCHS")
              << '\n';
}

int main()
{
    Eigen::VectorXd xs_0(2);
    xs_0 << 5.0, 5.0;
    Eigen::VectorXd xr_0(2);
    xr_0 << 1.0, -1.0;
    Eigen::VectorXd x_rast(4);
    x_rast << 2.0, 3.0, 4.0, 5.0;

    std::cout << "\n";
    std::cout << "Benchmark:\n";
    std::cout << std::left
              << std::setw(22) << "Optimizer"
              << std::setw(15) << "Function"
              << std::setw(12) << "Time(ms)"
              << std::setw(10) << "Epochs"
              << std::setw(12) << "f(x_opt)"
              << std::setw(12) << "Status" << '\n';

    Sphere sphere(2);
    Rosenbrock rosenbrock(2);
    Rastrigin rastrigin(4);

    // Benchmark for GD
    GDConfig gd_config;
    gd_config.learning_rate = 0.01;
    gd_config.max_epochs = 10000;
    GradientDescent gd(gd_config);
    run_benchmark("Gradient Descent", "Sphere", gd, sphere, xs_0);
    run_benchmark("Gradient Descent", "Rosenbrock", gd, rosenbrock, xr_0);
    run_benchmark("Gradient Descent", "Rastrigin", gd, rastrigin, x_rast);

    //Benchmark for Newton
    NewtonConfig newton_config;
    Newton newton(newton_config);
    run_benchmark("Newton Descent", "Sphere", newton, sphere, xs_0);
    run_benchmark("Newton Descent", "Rosenbrock", newton, rosenbrock, xr_0);
    run_benchmark("Newton Descent", "Rastrigin", newton, rastrigin, x_rast);

    //Benchmark for Hill Climbing
    HCConfig hc_config;
    hc_config.max_epochs = 100000;
    hc_config.step_size = 0.1;
    HC hc(hc_config);
    run_benchmark("Hill Climbing", "Sphere", hc, sphere, xs_0);
    run_benchmark("Hill Climbing", "Rosenbrock", hc, rosenbrock, xr_0);
    run_benchmark("Hill Climbing", "Rastrigin", hc, rastrigin, x_rast);

    //Benchmark for Simulated Annealing
    SAConfig sa_config;
    SA sa(sa_config);
    run_benchmark("Simulated Annealing", "Sphere", sa, sphere, xs_0);
    run_benchmark("Simulated Annealing", "Rosenbrock", sa, rosenbrock, xr_0);
    run_benchmark("Simulated Annealing", "Rastrigin", sa, rastrigin, x_rast);
}