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
    assert np.linalg.norm(result.optimal_x) < 1e-3

    print("SUCCESS")

wrapper_test()