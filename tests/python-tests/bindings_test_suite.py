import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import optim_engine as opt
import numpy as np

def test_gradient_descent():

    sphere = opt.Sphere(2)
    config = opt.GDConfig()
    gd = opt.GradientDescent(config)
    result = gd.optimize(sphere, np.array([5.0, 5.0]))
    assert result.status == opt.Status.CONVERGED
    assert result.optimal_f < 1e-8
    print("GD:  PASSED")

def test_newton():

    sphere = opt.Sphere(2)
    config = opt.NewtonConfig()
    newton = opt.Newton(config)
    result = newton.optimize(sphere, np.array([5.0, 5.0]))
    assert result.status == opt.Status.CONVERGED
    assert result.optimal_f < 1e-8
    print("Newton:  PASSED")

def test_hill_climbing():

    sphere = opt.Sphere(2)
    config = opt.HCConfig()
    config.max_epochs = 100000
    hc = opt.HC(config)
    result = hc.optimize(sphere, np.array([5.0, 5.0]))
    assert result.status == opt.Status.CONVERGED
    assert result.optimal_f < 1e-4
    print("HC:  PASSED")

def test_simulated_annealing():

    sphere = opt.Sphere(2)
    config = opt.SAConfig()
    config.max_epochs = 100000
    sa = opt.SA(config)
    result = sa.optimize(sphere, np.array([5.0, 5.0]))
    assert result.status == opt.Status.CONVERGED
    assert result.optimal_f < 1e-3
    print("SA:  PASSED")

def test_custom_function():

    class CustomSphere(opt.ObjectiveFunction):
        def evaluate(self, x):
            return float(np.dot(x, x))
        def grad(self, x):
            return 2.0 * x
        def hessian(self, x):
            return 2.0 * np.eye(len(x))
        def dim(self):
            return 2

    f = CustomSphere()
    config = opt.GDConfig()
    gd = opt.GradientDescent(config)
    result = gd.optimize(f, np.array([3.0, 3.0]))
    assert result.optimal_f < 1e-8
    print("Custom Function:  PASSED")

if __name__ == "__main__":
    test_gradient_descent()
    test_newton()
    test_hill_climbing()
    test_simulated_annealing()
    test_custom_function()