import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import optim_engine as opt
import numpy as np

sphere = opt.Sphere(2)
config = opt.GDConfig()
gd = opt.GradientDescent(config)
x0 = np.array([5.0, 5.0])
result = gd.optimize(sphere, x0)

print(f"Converged: {result.converged}")
print(f"Optimal x: {result.optimal_x}")
print(f"Optimal f: {result.optimal_f}")