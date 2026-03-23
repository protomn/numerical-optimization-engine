#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <Eigen/Dense>
#include <string>
#include <umbrella_header.hpp>

namespace py = pybind11;

/*
Creating a trampoline class to be able to expose and override
virtual and pure virtual methods.
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

    // Exposing the OptimResult struct

    py::class_<OptimResult>(m, "OptimResult")
        .def(py::init<>())
        .def_readonly("optimal_x", &OptimResult::optimal_x)
        .def_readonly("optimal_f", &OptimResult::optimal_f)
        .def_readonly("iterations", &OptimResult::iterations)
        .def_readonly("converged", &OptimResult::converged)
        .def_readonly("message", &OptimResult::message)
        .def_readonly("history", &OptimResult::history);
        
    
    py::class_<Optimizer, PyOptimizer>(m, "Optimizer")
        .def(py::init<>())
        .def("optimize", &Optimizer::optimize);


    py::class_<GDConfig>(m, "GDConfig")
        .def(py::init<>())
        .def_readwrite("learning_rate", &GDConfig::learning_rate)
        .def_readwrite("max_epochs", &GDConfig::max_epochs)
        .def_readwrite("tol", &GDConfig::tol)
        .def_readwrite("history", &GDConfig::history);

    py::class_<NewtonConfig>(m, "NewtonConfig")
        .def(py::init<>())
        .def_readwrite("tol", &NewtonConfig::tol)
        .def_readwrite("max_epochs", &NewtonConfig::max_epochs)
        .def_readwrite("history", &NewtonConfig::history);

    py::class_<HCConfig>(m, "HCConfig")
        .def(py::init<>())
        .def_readwrite("max_epochs", &HCConfig::max_epochs)
        .def_readwrite("step_size", &HCConfig::step_size)
        .def_readwrite("tol", &HCConfig::tol)
        .def_readwrite("history", &HCConfig::history);

    py::class_<SAConfig>(m, "SAConfig")
        .def(py::init<>())
        .def_readwrite("initial_temp", &SAConfig::initial_temp)
        .def_readwrite("cooling_rate", &SAConfig::cooling_rate)
        .def_readwrite("min_temp", &SAConfig::min_temp)
        .def_readwrite("step_size", &SAConfig::step_size)
        .def_readwrite("max_epochs", &SAConfig::max_epochs)
        .def_readwrite("tol", &SAConfig::tol)
        .def_readwrite("history", &SAConfig::history);

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
}