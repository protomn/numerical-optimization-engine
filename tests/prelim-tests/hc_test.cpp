#define CATCH_CONFIG_MAIN
#include "catch2/catch.hpp"
#include "functions/sphere.hpp"
#include "functions/rosenbrock.hpp"
#include "optimizers/hill_climbing.hpp"

TEST_CASE("Hill Climbing on Sphere", "[hill_climbing]")
{
    Sphere test_sphere(2);
    Rosenbrock test_rosenbrock(2);
    HCConfig test_config;

    SECTION("Converges from [5, 5]")
    {
        test_config.max_epochs = 100000;
        test_config.step_size = 0.01;
        HC test_hc(test_config);

        Eigen::VectorXd test_x(2);
        test_x << 5.0, 5.0;
        OptimResult test_result = test_hc.optimize(test_sphere, test_x);

        CHECK(test_result.converged == true);
        CHECK(test_result.optimal_f < 1e-6);
        CHECK(test_result.optimal_x.norm() < 1e-3);
    }

    SECTION("Fails.")
    {
        test_config.max_epochs = 10000;
        test_config.step_size = 1.0;
        HC test_hc(test_config);

        Eigen::VectorXd test_x(2);
        test_x << 1.0, -1.0;
        OptimResult test_result = test_hc.optimize(test_rosenbrock, test_x);

        CHECK(test_result.converged == false);
        CHECK(test_result.iterations == 10000);
    }
}