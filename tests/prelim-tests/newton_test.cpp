#define CATCH_CONFIG_MAIN
#include "catch2/catch.hpp"
#include "functions/sphere.hpp"
#include "functions/rosenbrock.hpp"
#include "optimizers/newton.hpp"

TEST_CASE("Newton on Sphere", "[newton]")
{
    Sphere test_sphere(2);
    Rosenbrock test_rosenbrock(2);
    NewtonConfig test_config;

    SECTION("Converges from [5, 5]")
    {
        test_config.max_epochs = 1000;
        test_config.history = true;
        Newton test_newton(test_config);

        Eigen::VectorXd test_x(2);
        test_x << 5.0, 5.0;
        OptimResult test_result = test_newton.optimize(test_sphere, test_x);

        CHECK(test_result.converged == true);
        CHECK(test_result.optimal_f < 1e-8);
        CHECK(test_result.optimal_x.norm() < 1e-4);
    }

    SECTION("Fails.")
    {
        test_config.max_epochs = 3;
        test_config.tol = 1e-20;
        Newton test_newton(test_config);

        Eigen::VectorXd test_x(2);
        test_x << -1.0, 1.0;
        OptimResult test_result = test_newton.optimize(test_rosenbrock, test_x);

        CHECK(test_result.converged == false);
        CHECK(test_result.iterations == 3);
    }
}