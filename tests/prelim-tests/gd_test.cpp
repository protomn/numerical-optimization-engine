#define CATCH_CONFIG_MAIN
#include "catch2/catch.hpp"
#include "functions/sphere.hpp"
#include "optimizers/gradient_descent.hpp"

TEST_CASE("GD on Sphere", "[gradient_descent]")
{
    Sphere test_sphere(2);
    GDConfig test_config;

    SECTION("Converges from [5, 5]")
    {   
        test_config.learning_rate = 0.01;
        test_config.max_epochs = 10000;
        GradientDescent test_gd(test_config);

        Eigen::VectorXd test_x0(2);
        test_x0 << 5.0, 5.0;
        OptimResult test_result = test_gd.optimize(test_sphere, test_x0);

        CHECK(test_result.converged == true);
        CHECK(test_result.optimal_f < 1e-8);
        CHECK(test_result.optimal_x.norm() < 1e-4);
    }

    SECTION("Fails.")
    {
        test_config.learning_rate = 10;
        test_config.max_epochs = 100;
        GradientDescent test_gd(test_config);

        Eigen::VectorXd test_x0(2);
        test_x0 << 5.0, 5.0;
        OptimResult test_result = test_gd.optimize(test_sphere, test_x0);

        CHECK(test_result.converged == false);
        CHECK(test_result.iterations == 100);
    }
}