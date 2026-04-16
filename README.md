# numerical-optimization-engine

A from-scratch numerical optimization engine in C++ with Python bindings via pybind11.

---

## Overview

This library implements five optimization algorithms over a shared abstract interface, supporting both analytic and finite-difference derivatives. The design separates algorithm logic from objective function definitions, allowing any optimizer to be applied to any function without modification.

Core design principles:

- Uniform optimizer interface: all algorithms implement a single `optimize(f, x0)` method.
- Separation of objectives and algorithms: functions subclass `ObjectiveFunction`; optimizers subclass `Optimizer`.
- Numerical robustness: explicit handling of indefinite Hessians, curvature pair degeneracy, line search bracket escape, and floating-point overflow in acceptance probability.
- Python interoperability: full binding of all types and configs, including support for Python-defined objective functions.

---

## Architecture

### Optimizer Abstraction

All optimizers inherit from `Optimizer` and implement:

```cpp
OptimResult optimize(ObjectiveFunction &f, const Eigen::VectorXd &x_0);
```

Objective functions inherit from `ObjectiveFunction` and implement `evaluate`, `grad`, `hessian`, and `dim`. The line search and finite difference wrapper operate on the same interface.

### Optimizers

**Gradient Descent** (`src/optimizers/gradient_descent.cpp`)

Standard steepest descent: `x_{n+1} = x_n - alpha_n * grad f(x_n)`. Step size is determined by Wolfe line search when `backtracking = true` (default), or a fixed `learning_rate` otherwise. Convergence is checked on relative gradient norm.

**Newton's Method** (`src/optimizers/newton.cpp`)

Computes `d_n = (H + mu*I)^{-1} * grad f(x_n)` where `H = hessian f(x_n)`. The diagonal shift `mu` is initialized at `initial_mu` and doubled until the Cholesky factorization of `H + mu*I` succeeds, guaranteeing a positive definite system and a valid descent direction regardless of Hessian conditioning. The Newton step is then scaled by Wolfe line search.

**L-BFGS** (`src/optimizers/lbfgs.cpp`)

Limited-memory quasi-Newton method. The inverse Hessian approximation `H_n` is built implicitly from the `m` most recent displacement pairs `(s_n, y_n)` and applied via the two-loop recursion (Nocedal 1980). Curvature pairs with `s^T y <= 1e-8 * ||s|| * ||y||` are skipped to prevent near-singular updates. The line search uses `c2 = 0.1` (tighter than the GD default of `0.9`) to guarantee `y^T s > 0`, which is required for the BFGS update to remain positive definite. Convergence is checked on relative gradient norm: `||grad f|| < tol * max(1, |f(x)|)`.

**Hill Climbing** (`src/optimizers/hill_climbing.cpp`)

Derivative-free stochastic local search. At each step, a candidate `x' = x + eps` is drawn where `eps ~ N(0, step_size^2 * I)`. The candidate is accepted only if `f(x') < f(x)`. Two stopping criteria run in parallel: improvement below `tol` (CONVERGED), or no accepted step for `stagnation_limit` consecutive iterations (STAGNATED).

**Simulated Annealing** (`src/optimizers/simulated_annealing.cpp`)

Metropolis-Hastings global search with geometric cooling `T_{n+1} = cooling_rate * T_n`. Worsening moves are accepted with probability `exp(-delta_f / T)`, clamped to prevent overflow when `T` is near `min_temp`. The best solution seen across all iterations is returned, not the final chain position.

### Supporting Components

**Wolfe Line Search** (`src/core/line_search.cpp`)

Implements the More-Thuente bracket-and-zoom algorithm (Nocedal & Wright). The bracket phase expands `alpha` from `1.0` until a Wolfe-satisfying interval is identified. The zoom phase narrows the interval via cubic interpolation, with the trial point clamped to the inner 80% of the bracket to prevent degenerate near-boundary steps. The cubic interpolation falls back to bisection when the discriminant is negative or the denominator is near-zero.

**Finite Difference Wrapper** (`include/core/finite_difference.hpp`)

Wraps any `ObjectiveFunction` to provide numerically approximated derivatives when analytic forms are unavailable. Uses central differences for the gradient (O(h^2), 2n evaluations) and second/mixed differences for the Hessian (O(h^2), O(n^2) evaluations). Default step size `h = 1e-5` balances truncation error against floating-point cancellation for double precision. Symmetry of the Hessian is enforced explicitly.

### Numerical Safeguards

- **Newton**: Cholesky-based indefinite Hessian detection with iterative diagonal shift; fails gracefully if shift exceeds `1e10`.
- **L-BFGS**: Relative curvature guard `s^T y > 1e-8 * ||s|| * ||y||` on each pair before storage.
- **Line search**: Cubic interpolation result clamped to bracket interior; denominator near-zero check with bisection fallback.
- **Simulated Annealing**: Acceptance exponent clamped at `-500` before `exp()` to prevent IEEE 754 overflow.

---

## Python Bindings

Python bindings are implemented with pybind11 (`python/bindings.cpp`). The module is compiled as a shared library (`optim_engine.cpython-*.so`) and importable from the project root.

**Trampoline classes** (`PyObjectiveFunction`, `PyOptimizer`) enable Python subclassing of the abstract C++ base classes. Each virtual method is forwarded via `PYBIND11_OVERRIDE_PURE`, allowing Python-defined objective functions to be passed directly to C++ optimizers.

**Exposed to Python:**

- All five optimizer classes and their config structs
- `ObjectiveFunction` base class (subclassable)
- `FiniteDiffWrapper` with `py::keep_alive<1,2>()` to prevent dangling reference to the wrapped object
- `OptimResult` (read-only fields)
- `Status` enum
- `WolfeConfig` and the `lineSearch` free function
- `Sphere`, `Rosenbrock`, `Rastrigin`

---

## Build Instructions

### Dependencies

| Dependency | Version tested |
|------------|---------------|
| Compiler   | Apple Clang 21 / GCC 12+ with C++20 support |
| Eigen      | 3.4+ (`/opt/homebrew/include/eigen3` or equivalent) |
| pybind11   | 3.0+ |
| Python     | 3.10+ (headers required) |
| Catch2     | v2 (vendored in `vendor/catch2/`) |

### Commands

```bash
# Build everything: optimizer objects, test binaries, benchmarks, Python extension
make all

# Individual targets
make src            # compile src/ objects only
make benchmarks     # build bin/benchmarks
make prelim-tests   # build bin/prelim-tests/<name> for each test
make python         # build optim_engine.cpython-*.so
make clean          # remove build/ and bin/
```

### Expected outputs

```
bin/benchmarks
bin/prelim-tests/{gd_test,newton_test,lbfgs_test,hc_test,sa_test}
optim_engine.cpython-*.so
```

If `SDKROOT` is stale (common after macOS SDK updates), add `unexport SDKROOT` to the top of the Makefile (already present) or run `unset SDKROOT` in your shell before building.

---

## Usage

### C++

```cpp
#include "umbrella_header.hpp"

Rosenbrock f(2);

LBFGSConfig config;
config.tol = 1e-10;
LBFGS optimizer(config);

Eigen::VectorXd x0(2);
x0 << 1.0, -1.0;

OptimResult result = optimizer.optimize(f, x0);
// result.optimal_x  -- Eigen::VectorXd
// result.optimal_f  -- double
// result.status     -- Status::CONVERGED | MAX_EPOCHS | STAGNATED | FAILED
// result.iterations -- int
```

### Python

```python
import optim_engine as opt
import numpy as np

f = opt.Rosenbrock(2)
config = opt.LBFGSConfig()
config.tol = 1e-10

optimizer = opt.LBFGS(config)
result = optimizer.optimize(f, np.array([1.0, -1.0]))

print(result.optimal_f)   # float
print(result.optimal_x)   # numpy array
print(result.status)      # opt.Status enum value
print(result.iterations)  # int
```

### Custom Objective Function

**C++:**

```cpp
class MyFunction : public ObjectiveFunction {
public:
    double evaluate(const Eigen::VectorXd &x) const override { ... }
    Eigen::VectorXd grad(const Eigen::VectorXd &x) const override { ... }
    Eigen::MatrixXd hessian(const Eigen::VectorXd &x) const override { ... }
    int dim() const override { return n_; }
private:
    int n_;
};
```

**Python with analytic derivatives:**

```python
class MyFunction(opt.ObjectiveFunction):
    def evaluate(self, x): return float(np.dot(x, x))
    def grad(self, x):     return 2.0 * x
    def hessian(self, x):  return 2.0 * np.eye(len(x))
    def dim(self):         return 2
```

**Python without derivatives (finite differences):**

```python
class MyFunction(opt.ObjectiveFunction):
    def evaluate(self, x): return float(x[0]**4 + x[1]**4)
    def grad(self, x):     raise NotImplementedError
    def hessian(self, x):  raise NotImplementedError
    def dim(self):         return 2

f = MyFunction()
wrapped = opt.FiniteDiffWrapper(f, h=1e-5)  # f must outlive wrapped

result = opt.GradientDescent(opt.GDConfig()).optimize(wrapped, np.array([2.0, 2.0]))
```

---

## Testing

### C++ tests (Catch2)

```bash
make run-tests          # build and run all test binaries sequentially

# or individually:
./bin/prelim-tests/gd_test
./bin/prelim-tests/newton_test
./bin/prelim-tests/lbfgs_test
./bin/prelim-tests/hc_test
./bin/prelim-tests/sa_test
```

### Python tests

```bash
python3 tests/python-tests/bindings_test_suite.py
python3 tests/python-tests/fdw_test.py
```

### Coverage

Each optimizer has two C++ test cases: convergence on a unimodal function and a defined non-convergence scenario. Python tests cover the binding layer for all five optimizers, a Python-subclassed objective, and the finite difference wrapper.

Hill Climbing and Simulated Annealing are stochastic. Individual runs may return `STAGNATED` rather than `CONVERGED` under default config; this does not indicate a correctness failure. The final `optimal_f` value is the relevant metric for these optimizers.

---

## Project Structure

```
include/
  core/           # Abstract interfaces and supporting utilities
    objective.hpp
    optimizer.hpp
    result.hpp
    line_search.hpp
    finite_difference.hpp
  optimizers/     # Config structs and class declarations
  functions/      # Sphere, Rosenbrock, Rastrigin (header-only)

src/
  core/
    line_search.cpp
  optimizers/     # Algorithm implementations

python/
  bindings.cpp    # pybind11 module

tests/
  prelim-tests/   # Catch2 C++ tests, one file per optimizer
  python-tests/   # Python binding and integration tests

benchmarks/
  benchmarks.cpp  # C++ benchmark runner
  benchmarks.py   # Python comparison benchmark
  plots.py        # Convergence plotting utilities

vendor/
  catch2/         # Vendored Catch2 v2 single-header
```

---

## Notes on Numerical Stability

**Line search**: The More-Thuente zoom phase maintains the standard bracket invariants. Cubic interpolation results are clamped to the inner 80% of the bracket to prevent degenerate near-boundary trial steps. A denominator near-zero check and discriminant less than zero falls back to bisection.

**L-BFGS curvature pairs**: The relative threshold `s^T y > 1e-8 * ||s|| * ||y||` is scale-invariant. An absolute threshold (`s^T y > 0`) accepts near-orthogonal pairs that produce large `rho = 1/(s^T y)` values, amplifying numerical error in subsequent two-loop iterations.

**Newton regularization**: The modified Cholesky approach doubles `mu` on each failed Cholesky factorization. At `mu >= 1e10` the step is abandoned and `Status::FAILED` is returned, rather than silently producing a non-descent direction.

**Simulated annealing overflow**: The exponent in `exp(-delta_f / T)` is clamped to `[-500, 0]` before evaluation, corresponding to a minimum acceptance probability near `7e-218`.