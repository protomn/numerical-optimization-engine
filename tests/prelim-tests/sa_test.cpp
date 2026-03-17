#define CATCH_CONFIG_MAIN
#include "catch2/catch.hpp"
#include "functions/sphere.hpp"
#include "functions/rosenbrock.hpp"
#include "optimizers/simulated_annealing.hpp"

TEST_CASE("Simulated Annealing on Sphere", "[simulated_annealing]")
{
    Sphere test_sphere(2);
    Rosenbrock test_rosenbrock(2);
    SAConfig test_config;

    SECTION("Converges from [5.0, 5.0]")
    {
        test_config.max_epochs = 100000;
        test_config.history = true;
        SA test_sa(test_config);

        Eigen::VectorXd test_x0(2);
        test_x0 << 5.0, 5.0;
        OptimResult test_result = test_sa.optimize(test_sphere, test_x0);

        CHECK(test_result.converged == true);
        CHECK(test_result.optimal_f < 1e-3);
        CHECK(test_result.optimal_x.norm() < 1e-1);
    }

    SECTION("Fails")
    {
        test_config.max_epochs = 1000;
        test_config.step_size = 1.0;
        SA test_sa(test_config);

        Eigen::VectorXd test_x0(2);
        test_x0 << 1.0, -1.0;
        OptimResult test_result = test_sa.optimize(test_rosenbrock, test_x0);

        CHECK(test_result.converged == false);
        CHECK(test_result.iterations == 1000);
    }
}