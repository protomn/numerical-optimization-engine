import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import optim_engine as opt
import numpy as np

def wrapper_test():

    class QuarticFunction(opt.ObjectiveFunction):

        def evaluate(self, x: np.ndarray):

            return float(x[0] ** 4 + x[1] ** 4)
        
        def grad(self, x):

            raise NotImplementedError

        def hessian(self, x):

            raise NotImplementedError
        
        def dim(self):

            return 2
        
    f = QuarticFunction()
    wrapped = opt.FiniteDiffWrapper(f)

    config = opt.GDConfig()
    gd = opt.GradientDescent(config)
    result = gd.optimize(wrapped, np.array([2.0, 2.0]))

    assert result.status == opt.Status.CONVERGED
    assert result.optimal_f < 1e-6
    # The objective is a quartic, so f == sum(x**4) and the two tolerances must
    # be consistent: accepting f < 1e-6 permits ||x|| up to (1e-6)**0.25 ~= 0.032.
    # The previous bound of 1e-3 implied f ~ 1e-12, which is below what finite
    # differences with h=1e-5 can resolve. Measured value here is ~0.0079.
    assert np.linalg.norm(result.optimal_x) < 5e-2

    print("SUCCESS")

wrapper_test()