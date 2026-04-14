#define CATCH_CONFIG_MAIN
#include "catch2/catch.hpp"
#include "functions/sphere.hpp"
#include "functions/rosenbrock.hpp"
#include "functions/rastrigin.hpp"
#include "optimizers/lbfgs.hpp"

TEST_CASE("LBFGS on Sphere", "[lbfgs]")
{
    Sphere test_sphere(2);
    LBFGSConfig test_config;
    
    SECTION("Converges from [5, 5]")
    {
        test_config.max_epochs = 1000;
        test_config.history = true;
        LBFGS test_lbfgs(test_config);

        Eigen::VectorXd test_x(2);
        test_x << 5.0, 5.0;
        OptimResult test_result = test_lbfgs.optimize(test_sphere, test_x);

        CHECK(test_result.status == Status::CONVERGED);
        CHECK(test_result.optimal_f < 1e-7);
        CHECK(test_result.optimal_x.norm() < 1e-4);
    }

    SECTION("Fails on Rastrigin")
    {
        test_config.max_epochs = 3;
        test_config.tol = 1e-30;
        LBFGS test_lbfgs(test_config);
        
        Rastrigin test_rastrigin(10);
        Eigen::VectorXd test_x = Eigen::VectorXd::Constant(10, 3.0);
        OptimResult test_result = test_lbfgs.optimize(test_rastrigin, test_x);

        CHECK(test_result.status == Status::MAX_EPOCHS);
        CHECK(test_result.iterations == 3);
    }
}