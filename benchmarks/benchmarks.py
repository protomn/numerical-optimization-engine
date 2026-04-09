import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import optim_engine as opt
import scipy.optimize as sciop
from scipy.optimize import dual_annealing as da
import time
from typing import Callable, Any

def benchmark(name: str, func: Callable, runs: int = 10) -> tuple[float, Any]:

    runtimes = []
    res = None

    for _ in range(runs):

        start = time.perf_counter()
        res = func()
        end = time.perf_counter()

        runtimes.append((end - start) * 1000)

    av_time = sum(runtimes) / len(runtimes)

    return av_time, res

# Gradient Descent

x_0 = np.array([5.0, 5.0])
sphere = opt.Sphere(2)
gd_config = opt.GDConfig()
gd = opt.GradientDescent(gd_config)
fn_gd = lambda: gd.optimize(sphere, x_0)
av_time_gd, res_gd_ = benchmark("Gradient Descent", fn_gd)

fn_gd_sci = lambda: sciop.minimize(
    fun = lambda x: float(np.dot(x, x)),
    x0 = np.array([5.0, 5.0]),
    jac = lambda x: 2.0 * x,
    method = 'CG'
)

av_time_scigd, res_scigd = benchmark("SciPy GD", fn_gd_sci)

# Newton

newton_config = opt.NewtonConfig()
newton = opt.Newton(newton_config)
fn_newton = lambda: newton.optimize(sphere, x_0)
av_time_newt, res_newt_ = benchmark("Newton Descent", fn_newton)

fn_newt_sci = lambda: sciop.minimize(
    fun = lambda x: float(np.dot(x, x)),
    x0 = x_0,
    jac = lambda x: 2.0 * x,
    hess = lambda x: 2.0 * np.eye(len(x)),
    method = "Newton-CG"
)

av_time_scinew, res_scinew = benchmark("SciPy Newton", fn_newt_sci)

# Hill Climbing
'''
Since SciPy doesn't have a direct Hill Climbing method, 
the implemented C++ method is compared against SciPy's 
Nelder-Mead Method.
'''
hc_config = opt.HCConfig()
hc_config.max_epochs = 100000
hc_config.step_size = 0.1
hc = opt.HC(hc_config)
fn_hc = lambda: hc.optimize(sphere, x_0)
av_time_hc, res_hc_ = benchmark("Hill Climbing", fn_hc)

fn_nm_sci = lambda: sciop.minimize(
    fun = lambda x: float(np.dot(x, x)),
    x0 = x_0,
    method = "Nelder-Mead"
)

av_time_scinm, res_scinm = benchmark("SciPy Nelder-Mead", fn_nm_sci)

# Simulated Annealing
sa_config = opt.SAConfig()
sa = opt.SA(sa_config)
fn_sa = lambda: sa.optimize(sphere, x_0)
av_time_sa, res_sa_ = benchmark("Simulated Annealing", fn_sa)

fn_sa_sci = lambda: da(
    func = lambda x: float(np.dot(x, x)),
    bounds = [(-10, 10), (-10, 10)],
    x0 = x_0
)

av_time_scisa, res_scisa = benchmark("SciPy Dual Annealing", fn_sa_sci)

print("Results:")
print(f"{'Optimizer':<20} {'Time(ms)':<12} {'Iters':<10} {'f(x*)'}\n")
print(f"{'Gradient Descent':<20} {av_time_gd:<12.3f} {res_gd_.iterations:<10} {res_gd_.optimal_f:.2e}\n")
print(f"{"SciPy GD":<20} {av_time_scigd:<12.3f} {res_scigd.nit:<10} {res_scigd.fun:.2e}\n")
print(f"{'Newton Descent':<20} {av_time_newt:<12.3f} {res_newt_.iterations:<10} {res_newt_.optimal_f:2e}\n")
print(f"{"SciPy Newton":<20} {av_time_scinew:<12.3f} {res_scinew.nit:<10} {res_scinew.fun:.2e}\n")
print(f"{'Hill Climbing':<20} {av_time_hc:<12.3f} {res_hc_.iterations:<10} {res_hc_.optimal_f:.2e}\n")
print(f"{"SciPy Nelder-Mead":<20} {av_time_scinm:<12.3f} {res_scinm.nit:<10} {res_scinm.fun:.2e}\n")
print(f"{'Simulated Annealing':<20} {av_time_sa:<12.3f} {res_sa_.iterations:<10} {res_sa_.optimal_f:.2e}\n")
print(f"{"SciPy Dual Annealing":<20} {av_time_scisa:<12.3f} {res_scisa.nit:<10} {res_scisa.fun:.2e}\n")