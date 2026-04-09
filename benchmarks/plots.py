import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import optim_engine as opt
import scipy.optimize as sciop
from scipy.optimize import dual_annealing as da
import matplotlib.pyplot as plt
import time

root_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(root_dir, "..", "docs", "images")
os.makedirs(images_dir, exist_ok = True)

sphere = opt.Sphere(2)
x_0 = np.array([5.0, 5.0])

gd_config = opt.GDConfig()
gd_config.history = True
gd = opt.GradientDescent(gd_config)
res_gd = gd.optimize(sphere, x_0)
gd_hist = res_gd.history

sci_hist = []

def callback(x: float) -> None:

    sci_hist.append(float(np.dot(x, x)))

def sa_callback(_: np.ndarray, f: float, __: int) -> None:

    sci_hist.append(f)

sciop.minimize(
    fun = lambda x: float(np.dot(x, x)),
    x0 = x_0,
    jac = lambda x: 2.0 * x,
    method = "CG",
    callback = callback
)

fig_1, axes_1 = plt.subplots(1, 2, figsize = (12, 5))
axes_1[0].plot(gd_hist, label = "C++ GD", color = "blue")
axes_1[1].plot(sci_hist, label = "SciPy GD", color = "red")
axes_1[0].set_xlabel("Epochs")
axes_1[0].set_ylabel("f(x)")
axes_1[1].set_xlabel("Epochs")
axes_1[1].set_ylabel("f(x)")
fig_1.suptitle("Gradient Descent Convergence (C++ v. SciPy)")
axes_1[0].legend()
axes_1[0].set_yscale("log")
axes_1[1].legend()
axes_1[1].set_yscale("log")
plt.tight_layout()
plt.savefig(os.path.join(images_dir, "gd_comparison.png"), dpi = 150, bbox_inches = "tight")
plt.show()
plt.close()
sci_hist.clear()

"""
Newton Descent converges in a very small number of epochs.
This makes it difficult to compare optimizer performance via
a convergence plot.
So instead I compared the optimizer speed, now fast does one
optimizer converge as compared to the other. This is plotted in a
histogram.
"""

newton_config = opt.NewtonConfig()
newton_config.history = True
newton = opt.Newton(newton_config)

cpp_times = []
for _ in range(1000):

    start = time.perf_counter()
    newton.optimize(sphere, x_0)
    end = time.perf_counter()
    cpp_times.append((end - start) * 1000) # Time is in ms

sci_times = []
for _ in range(1000):

    start = time.perf_counter()
    sciop.minimize(
        fun = lambda x: float(np.dot(x, x)),
        x0 = x_0,
        jac = lambda x: 2.0 * x,
        hess = lambda x: 2.0 * np.eye(len(x)),
        method = "Newton-CG",
        callback = callback
    )
    end = time.perf_counter()
    sci_times.append((end - start) * 1000)

fig_2, ax_2 = plt.subplots(figsize = (10, 5))
ax_2.hist(cpp_times, bins = 50, alpha = 0.6, label = "C++ Newton", color = "blue")
ax_2.hist(sci_times, bins = 50, alpha = 0.6, label = "SciPy Newton", color = "red")
ax_2.set_xlabel("Time (s)")
ax_2.set_ylabel("Frequency")
ax_2.set_title("Newton Convergence Time Distribution (1000 runs)")
ax_2.legend()
plt.xscale("log")
plt.tight_layout()
plt.savefig(os.path.join(images_dir, "newton_comparison.png"), dpi = 150, bbox_inches = "tight")
plt.close()
sci_hist.clear()

hc_config = opt.HCConfig()
hc_config.max_epochs = 100000
hc_config.step_size = 0.1
hc_config.history = True
hc = opt.HC(hc_config)
res_hc = hc.optimize(sphere, x_0)
hc_hist = res_hc.history

sciop.minimize(
    fun = lambda x: float(np.dot(x, x)),
    x0 = x_0,
    method = "Nelder-Mead",
    callback = callback
)

fig_3, axes_3 = plt.subplots(1, 2, figsize = (12, 5))
axes_3[0].plot(hc_hist, label = "C++ HC", color = "blue")
axes_3[1].plot(sci_hist, label = "SciPy HC", color = "red")
axes_3[0].set_xlabel("Epochs")
axes_3[0].set_ylabel("f(x)")
axes_3[1].set_xlabel("Epochs")
axes_3[1].set_ylabel("f(x)")
fig_3.suptitle("Hill Climbing Convergence (C++ v. SciPy [Nelder-Mead])")
axes_3[0].legend()
axes_3[0].set_yscale("log")
axes_3[1].legend()
axes_3[1].set_yscale("log")
plt.tight_layout()
plt.savefig(os.path.join(images_dir, "hc_comparison.png"), dpi = 150, bbox_inches = "tight")
plt.show()
plt.close()
sci_hist.clear()

sa_config = opt.SAConfig()
sa_config.history = True
sa = opt.SA(sa_config)
res_sa = sa.optimize(sphere, x_0)
sa_hist = res_sa.history

da(
    func = lambda x: float(np.dot(x, x)),
    bounds = [(-10, 10), (-10, 10)],
    x0 = x_0,
    callback = sa_callback
)

fig_4, axes_4 = plt.subplots(1, 2, figsize = (12, 5))
axes_4[0].plot(sa_hist, label = "C++ SA", color = "blue")
axes_4[1].plot(sci_hist, label = "SciPy SA", color = "red")
axes_4[0].set_xlabel("Epochs")
axes_4[0].set_ylabel("f(x)")
axes_4[1].set_xlabel("Epochs")
axes_4[1].set_ylabel("f(x)")
fig_4.suptitle("Simulated Annealing Convergence (C++ v. SciPy)")
axes_4[0].legend()
axes_4[0].set_yscale("log")
axes_4[1].legend()
axes_4[1].set_yscale("log")
plt.tight_layout()
plt.savefig(os.path.join(images_dir, "sa_comparison.png"), dpi = 150, bbox_inches = "tight")
plt.show()
plt.close()
sci_hist.clear()