#include <Eigen/Dense>
#include "objective.hpp"
#include "result.hpp"

class Optimizer
{
    public:

        virtual ~Optimizer() = default;

        virtual OptimResult optimize(
                ObjectiveFunction &f,
                const Eigen::VectorXd &x_0
        ) = 0;
};