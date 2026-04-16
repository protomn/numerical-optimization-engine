# Benchmarks

## Environment

| Field        | Value |
|-------------|-------|
| CPU         | Apple M2 |
| OS          | macOS 26.3.1 (Build 25D2128) |
| Compiler    | Apple Clang 21.0.0 (`-std=c++20 -O2`) |
| Python      | 3.14.2 |
| Eigen       | 3.4+ |
| Build       | `make clean && make all` |

---

## Benchmark Setup

**Test functions:**

| Function | Dimensions | Global minimum | Character |
|----------|-----------|---------------|-----------|
| Sphere | 2 | f(0,0) = 0 | Convex, isotropic, well-conditioned |
| Rosenbrock | 2 | f(1,1) = 0 | Non-convex, narrow curved valley, ill-conditioned (~2500 condition number near minimum) |
| Rastrigin | 4 | f(0,0,0,0) = 0 | Non-convex, ~10^4 local minima on the search domain |

**Starting points:**

| Function | x_0 |
|----------|-----|
| Sphere | [5.0, 5.0] |
| Rosenbrock | [1.0, -1.0] |
| Rastrigin | [2.5, 3.1, 4.7, 5.3] |

**Termination criteria:**

- Gradient-based (GD, Newton, L-BFGS): `||grad f(x)|| < tol * max(1, |f(x)|)` with `tol = 1e-6`
- Derivative-free (HC, SA): improvement below `tol = 1e-6` (HC) or `T < min_temp = 1e-8` (SA)
- Hard cap: `max_epochs` per optimizer (see config table below)

**Config used:**

| Optimizer | Key parameters |
|-----------|---------------|
| Gradient Descent | `learning_rate=0.01`, `backtracking=true`, `max_epochs=10000` |
| Newton | defaults: `initial_mu=1e-4`, `backtracking=true`, `max_epochs=1000` |
| L-BFGS | `m=10`, `c2=0.1`, `max_epochs=100000` |
| Hill Climbing | `step_size=0.1`, `stagnation_limit=1000000`, `max_epochs=100000` |
| Simulated Annealing | `initial_temp=1000`, `cooling_rate=0.995`, `min_temp=1e-8`, `step_size=0.1` |

**Timing**: wall-clock average over 10 runs per configuration, reported in milliseconds.

---

## C++ Results

| Optimizer | Function | Iterations | f(x*) | Time (ms) | Status |
|-----------|----------|-----------|-------|-----------|--------|
| Gradient Descent | Sphere | 1 | 0.00e+00 | 0.008 | CONVERGED |
| Gradient Descent | Rosenbrock | 10000 | 8.32e-09 | 5.447 | MAX_EPOCHS |
| Gradient Descent | Rastrigin | 17 | 6.87e+01 | 0.013 | CONVERGED |
| Newton | Sphere | 2 | 3.12e-16 | 0.001 | CONVERGED |
| Newton | Rosenbrock | 2 | 1.05e-19 | 0.001 | CONVERGED |
| Newton | Rastrigin | 6 | 5.97e+01 | 0.016 | CONVERGED |
| L-BFGS | Sphere | 1 | 0.00e+00 | 0.001 | CONVERGED |
| L-BFGS | Rosenbrock | 31 | 6.36e-18 | 0.071 | CONVERGED |
| L-BFGS | Rastrigin | 11 | 6.87e+01 | 0.031 | CONVERGED |
| Hill Climbing | Sphere | 23386 | 1.56e-06 | 1.312 | CONVERGED |
| Hill Climbing | Rosenbrock | 100000 | 2.23e-06 | 6.849 | MAX_EPOCHS |
| Hill Climbing | Rastrigin | 100000 | 6.27e+01 | 13.900 | MAX_EPOCHS |
| Simulated Annealing | Sphere | 5054 | 9.27e-06 | 0.493 | CONVERGED |
| Simulated Annealing | Rosenbrock | 5054 | 4.05e-06 | 0.488 | CONVERGED |
| Simulated Annealing | Rastrigin | 5054 | 3.26e+01 | 0.799 | CONVERGED |

Iteration count and runtime represent a single representative run. Stochastic optimizers (HC, SA) will produce different values across runs.

---

## Python / SciPy Comparison

All comparisons are performed on Sphere (2D), x_0 = [5.0, 5.0], to avoid conflating implementation differences with algorithmic behavior on ill-conditioned or multimodal functions. Timings are averages over 10 runs via `benchmarks/benchmarks.py`. All comparisons are performed under consistent termination criteria where applicable; differences in default stopping conditions across libraries may affect iteration counts.

The SciPy methods are not algorithmic equivalents: CG is conjugate gradient rather than steepest descent, Newton-CG uses an iterative Hessian-vector product rather than a direct solve, and dual annealing is a hybrid SA/local-search variant. They are included as scale references on the same function, not as head-to-head algorithm comparisons. Nelder-Mead is included only as a reference for derivative-free performance on this problem and is not directly comparable to Hill Climbing.

| Optimizer | Time (ms) | Iterations | f(x*) |
|-----------|-----------|-----------|-------|
| C++ GD | 0.061 | 1 | 0.00e+00 |
| SciPy CG | 0.258 | 3 | 4.08e-16 |
| C++ Newton | 0.023 | 2 | 3.12e-16 |
| SciPy Newton-CG | 0.134 | 2 | 0.00e+00 |
| C++ HC | 0.040 | 361 | 8.85e-05 |
| SciPy Nelder-Mead | 0.648 | 44 | 1.48e-09 |
| C++ SA | 0.533 | 5054 | 1.12e-05 |
| SciPy Dual Annealing | 47.318 | 1000 | 5.04e-17 |
| C++ L-BFGS | 0.002 | 1 | 0.00e+00 |
| SciPy L-BFGS-B | 0.062 | 2 | 7.73e-29 |

The observed runtime differences are primarily a reflection of implementation context rather than algorithmic efficiency. The C++ implementations execute natively through pybind11 bindings with minimal per-call overhead; SciPy introduces Python-level dispatch, convergence bookkeeping, and robustness checks that add latency independent of iteration count. These numbers should be read as rough scale indicators, not as a performance comparison between implementations. The SA comparison is not meaningful in accuracy terms — SciPy's dual annealing is a fundamentally different algorithm that combines simulated annealing with local search restarts.

---

## Test Results

All 5 C++ test suites (Catch2) passed: `gd_test`, `newton_test`, `lbfgs_test`, `hc_test`, `sa_test`.

Python binding tests (`bindings_test_suite.py`): GD, Newton, SA, and custom-function tests passed. HC test is non-deterministic — the assertion `status == CONVERGED` can fail when the optimizer returns `STAGNATED`, which is a valid termination for a stochastic optimizer with a finite stagnation limit. The `optimal_f` result is within tolerance regardless of status.

---

## Observations

**Sphere:**
Gradient Descent and L-BFGS both converge in a single iteration. For this quadratic, the line search recovers the exact minimiser along the descent direction in one step. Newton converges in 2 iterations from this starting point, reflecting quadratic convergence once near the optimum — the first iteration takes the step, the second checks the gradient norm and exits.

**Rosenbrock:**
The primary differentiator across optimizers. Gradient Descent with Wolfe line search does not converge within 10,000 iterations, reaching f = 8.32e-09. The narrow valley causes persistent zigzagging — this is the known pathology of steepest descent on ill-conditioned problems regardless of step size strategy. Newton converges in 2 iterations to f = 1.05e-19, exploiting the exact Hessian. L-BFGS converges in 31 iterations to f = 6.36e-18, building a sufficient inverse-Hessian approximation after the first few steps. Hill Climbing exhausts its 100,000-iteration budget (stochastic — may converge on other runs); the narrow valley is difficult to navigate with isotropic random perturbations. Simulated Annealing reaches f = 4.05e-06 within its fixed cooling schedule (5,054 steps to min_temp).

**Rastrigin:**
All gradient-based methods (GD, Newton, L-BFGS) converge to a local minimum near the starting point, not the global minimum at the origin. This is expected: the basin of attraction of any gradient method is determined by the starting point relative to the local minima grid. GD and L-BFGS converge to f = 6.87e+01 (the same local minimum, reached in 17 and 11 iterations respectively). Newton reaches f = 5.97e+01 in 6 iterations, landing in a different basin. The local Hessian structure in Rastrigin is dominated by the cosine oscillation term (`4π²A·cos(2πxᵢ)`) and does not reflect global geometry, which limits the usefulness of second-order information beyond local refinement. Hill Climbing and Simulated Annealing also converge to local minima; SA reaches f = 3.26e+01 within its cooling schedule — lower than Hill Climbing's final value, consistent with the probabilistic acceptance allowing escape from some local minima during the high-temperature phase.

**Iteration efficiency:**
On smooth, well-conditioned problems (Sphere), all gradient-based methods are equivalent in iteration count. The separation appears on ill-conditioned problems: Newton is the most iteration-efficient when an exact Hessian is available (quadratic convergence near the minimum), followed by L-BFGS (superlinear convergence via Hessian approximation), followed by GD (linear convergence, sensitive to conditioning). The derivative-free methods require orders of magnitude more iterations to reach comparable accuracy.

**Runtime:**
All gradient-based methods complete in under 10 ms on these 2–4 dimensional problems. The dominant cost for HC and SA is the number of function evaluations rather than per-evaluation arithmetic. L-BFGS at 0.071 ms for Rosenbrock reflects the two-loop recursion overhead per iteration relative to GD, but this is offset by requiring 320x fewer iterations.