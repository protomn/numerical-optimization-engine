#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <Eigen/Dense>
#include <string>
#include <umbrella_header.hpp>

namespace py = pybind11;

/*
trampoline class required by pybind11 to support python subclassing of abstract cpp classes
without this python class inheriting from cpp base classes would not be able to override
their pure virtual methods
*/

class PyObjectiveFunction : public ObjectiveFunction
{
    public:

        double evaluate(const Eigen::VectorXd &x) const override
        {
            PYBIND11_OVERRIDE_PURE(
                double,
                ObjectiveFunction,
                evaluate,
                x
            );
        }

        Eigen::VectorXd grad(const Eigen::VectorXd &x) const override
        {
            PYBIND11_OVERRIDE_PURE(
                Eigen::VectorXd,
                ObjectiveFunction,
                grad,
                x
            );
        }

        Eigen::MatrixXd hessian(const Eigen::VectorXd &x) const override
        {
            PYBIND11_OVERRIDE_PURE(
                Eigen::MatrixXd,
                ObjectiveFunction,
                hessian,
                x
            );
        }

        int dim() const override
        {
            PYBIND11_OVERRIDE_PURE(
                int, 
                ObjectiveFunction,
                dim
            );
        }
};

class PyOptimizer : public Optimizer
{
    public: 

        OptimResult optimize(
            ObjectiveFunction &f,
            const Eigen::VectorXd &x_0
        ) override
        {
            PYBIND11_OVERRIDE_PURE(
                OptimResult,
                Optimizer,
                optimize,
                f, x_0
            );
        }
};

/*
module entry point
exposes the entire optimization engine to python as optim_engine
registration order:
- bases classes
- everything else (enums, structs, derived classes)
*/

PYBIND11_MODULE(optim_engine, m)
{
    m.doc() = std::string("Numerical Optimization Engine");

    //Exposing the Objective Function

    py::class_<ObjectiveFunction, PyObjectiveFunction>(m, "ObjectiveFunction")
        .def(py::init<>())
        .def("evaluate", &ObjectiveFunction::evaluate)
        .def("grad", &ObjectiveFunction::grad)
        .def("hessian", &ObjectiveFunction::hessian)
        .def("dim", &ObjectiveFunction::dim);

    py::class_<FiniteDiffWrapper, ObjectiveFunction>(m, "FiniteDiffWrapper")
    /*
    py::keep_alive<1, 2>() keeps the wrapped ObjectiveFunction alive for at least as
    long as the FiniteDiffWrapper object is alive.
    this is to prevent python's gc from destroying the inner function while the 
    wrapper still holds a raw reference to it -> can lead to seg fault.
    */
        .def(py::init<ObjectiveFunction &, double>(),
             py::arg("f"), py::arg("h") = 1e-5,
             py::keep_alive<1, 2>())
        .def("evaluate", &FiniteDiffWrapper::evaluate)
        .def("grad", &FiniteDiffWrapper::grad)
        .def("hessian", &FiniteDiffWrapper::hessian)
        .def("dim", &FiniteDiffWrapper::dim);

    // Exposing status enum

    py::enum_<Status>(m, "Status")
        .value("CONVERGED", Status::CONVERGED)
        .value("STAGNATED", Status::STAGNATED)
        .value("MAX_EPOCHS", Status::MAX_EPOCHS)
        .value("FAILED", Status::FAILED)
        .export_values();

    // Exposing the OptimResult struct

    py::class_<OptimResult>(m, "OptimResult")
        .def(py::init<>())
        .def_readonly("optimal_x", &OptimResult::optimal_x)
        .def_readonly("optimal_f", &OptimResult::optimal_f)
        .def_readonly("iterations", &OptimResult::iterations)
        .def_readonly("status", &OptimResult::status)
        .def_readonly("message", &OptimResult::message)
        .def_readonly("history", &OptimResult::history);
        
    
    py::class_<Optimizer, PyOptimizer>(m, "Optimizer")
        .def(py::init<>())
        .def("optimize", &Optimizer::optimize);

    /*
    config structs exposed with readwrite access so parameters can be tuned via python
    */

    py::class_<GDConfig>(m, "GDConfig")
        .def(py::init<>())
        .def_readwrite("learning_rate", &GDConfig::learning_rate)
        .def_readwrite("max_epochs", &GDConfig::max_epochs)
        .def_readwrite("tol", &GDConfig::tol)
        .def_readwrite("history", &GDConfig::history)
        .def_readwrite("backtracking", &GDConfig::backtracking)
        .def_readwrite("c1", &GDConfig::c1)
        .def_readwrite("c2", &GDConfig::c2);

    py::class_<NewtonConfig>(m, "NewtonConfig")
        .def(py::init<>())
        .def_readwrite("tol", &NewtonConfig::tol)
        .def_readwrite("max_epochs", &NewtonConfig::max_epochs)
        .def_readwrite("history", &NewtonConfig::history)
        .def_readwrite("c1", &NewtonConfig::c1)
        .def_readwrite("c2", &NewtonConfig::c2)
        .def_readwrite("initial_mu", &NewtonConfig::initial_mu)
        .def_readwrite("backtracking", &NewtonConfig::backtracking);

    py::class_<HCConfig>(m, "HCConfig")
        .def(py::init<>())
        .def_readwrite("max_epochs", &HCConfig::max_epochs)
        .def_readwrite("step_size", &HCConfig::step_size)
        .def_readwrite("tol", &HCConfig::tol)
        .def_readwrite("stagnation_limit", &HCConfig::stagnation_limit)
        .def_readwrite("history", &HCConfig::history)
        .def_readwrite("seed", &HCConfig::seed);

    py::class_<SAConfig>(m, "SAConfig")
        .def(py::init<>())
        .def_readwrite("initial_temp", &SAConfig::initial_temp)
        .def_readwrite("cooling_rate", &SAConfig::cooling_rate)
        .def_readwrite("min_temp", &SAConfig::min_temp)
        .def_readwrite("step_size", &SAConfig::step_size)
        .def_readwrite("max_epochs", &SAConfig::max_epochs)
        .def_readwrite("tol", &SAConfig::tol)
        .def_readwrite("history", &SAConfig::history)
        .def_readwrite("seed", &SAConfig::seed);

    py::class_<LBFGSConfig>(m, "LBFGSConfig")
        .def(py::init<>())
        .def_readwrite("m", &LBFGSConfig::m)
        .def_readwrite("tol", &LBFGSConfig::tol)
        .def_readwrite("max_epochs", &LBFGSConfig::max_epochs)
        .def_readwrite("history", &LBFGSConfig::history)
        .def_readwrite("c1", &LBFGSConfig::c1)
        .def_readwrite("c2", &LBFGSConfig::c2);

    py::class_<WolfeConfig>(m, "WolfeConfig")
        .def(py::init<>())
        .def_readwrite("c1", &WolfeConfig::c1)
        .def_readwrite("c2", &WolfeConfig::c2)
        .def_readwrite("alpha_max", &WolfeConfig::alpha_max)
        .def_readwrite("max_epochs", &WolfeConfig::max_epochs);

    // Optimizer bindings

    /*
    exposed optimizers, each takes a config struct as an argument at construction
    and exposes a single optimizer method that matches the core code interface.
    */

    py::class_<GradientDescent, Optimizer>(m, "GradientDescent")
        .def(py::init<GDConfig>())
        .def("optimize", &GradientDescent::optimize);

    py::class_<Newton, Optimizer>(m, "Newton")
        .def(py::init<NewtonConfig>())
        .def("optimize", &Newton::optimize);

    py::class_<HC, Optimizer>(m, "HC")
        .def(py::init<HCConfig>())
        .def("optimize", &HC::optimize);

    py::class_<SA, Optimizer>(m ,"SA")
        .def(py::init<SAConfig>())
        .def("optimize", &SA::optimize);

    py::class_<LBFGS, Optimizer>(m, "LBFGS")
        .def(py::init<LBFGSConfig>())
        .def("optimize", &LBFGS::optimize);

    py::class_<Sphere, ObjectiveFunction>(m, "Sphere")
        .def(py::init<int>())
        .def("evaluate", &Sphere::evaluate)
        .def("grad", &Sphere::grad)
        .def("hessian", &Sphere::hessian)
        .def("dim", &Sphere::dim);

    py::class_<Rosenbrock, ObjectiveFunction>(m, "Rosenbrock")
        .def(py::init<int>())
        .def("evaluate", &Rosenbrock::evaluate)
        .def("grad", &Rosenbrock::grad)
        .def("hessian", &Rosenbrock::hessian)
        .def("dim", &Rosenbrock::dim);

    py::class_<Rastrigin, ObjectiveFunction>(m, "Rastrigin")
        .def(py::init<int>())
        .def("evaluate", &Rastrigin::evaluate)
        .def("grad", &Rastrigin::grad)
        .def("hessian", &Rastrigin::hessian)
        .def("dim", &Rastrigin::dim);

    m.def("lineSearch", &lineSearch,
          py::arg("f"),
          py::arg("x"),
          py::arg("d"),
          py::arg("grad"),
          py::arg("config") = WolfeConfig(),
          "Performs line search to satisfy Wolfe criterion.");
}