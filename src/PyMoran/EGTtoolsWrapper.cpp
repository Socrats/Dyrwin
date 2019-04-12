//
// Created by Elias Fernandez on 2019-02-11.
//

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include <vector>
#include "../../include/Dyrwin/PyMoran/PDImitation.h"
#include "../../include/Dyrwin/PyMoran/StochDynamics.h"
#include "../../include/Dyrwin/PyMoran/TraulsenMoran.h"

//PYBIND11_MAKE_OPAQUE(std::vector<float>);

namespace py = pybind11;
using namespace egt_tools;

PYBIND11_MODULE(EGTtools, m) {
    m.doc() = R"pbdoc(
        EGTtools: library with tools for efficient Evolutionary Game theory methods in python
        -----------------------
        .. currentmodule:: EGTtools
        .. autosummary::
           :toctree: _generate
           add
           subtract
    )pbdoc";

//    py::class_<std::vector<float>>(m, "FloatVector")
//            .def(py::init<>())
//            .def("clear", &std::vector<int>::clear)
//            .def("pop_back", &std::vector<int>::pop_back)
//            .def("__len__", [](const std::vector<int> &v) { return v.size(); })
//            .def("__iter__", [](std::vector<int> &v) {
//                return py::make_iterator(v.begin(), v.end());
//            }, py::keep_alive<0, 1>()); /* Keep vector alive while iterator is used */

    py::class_<PDImitation>(m, "PDImitation")
            .def(py::init<unsigned int, unsigned int, float, float, float, Eigen::Ref<const MatrixXd>>())
            .def_property("generations", &PDImitation::generations, &PDImitation::set_generations)
            .def_property("pop_size", &PDImitation::pop_size, &PDImitation::set_pop_size)
            .def_property_readonly("nb_coop", &PDImitation::nb_coop)
            .def_property("mu", &PDImitation::mu, &PDImitation::set_mu)
            .def_property("beta", &PDImitation::beta, &PDImitation::set_beta)
            .def_property("coop_freq", &PDImitation::coop_freq, &PDImitation::set_coop_freq)
            .def_property_readonly("result_coop_freq", &PDImitation::result_coop_freq)
            .def_property("payoff_matrix", &PDImitation::payoff_matrix, &PDImitation::set_payoff_matrix)
            .def("update_payoff_matrix", &PDImitation::set_payoff_matrix, py::return_value_policy::reference_internal)
            .def("evolve", static_cast<float (PDImitation::*)(float)>(&PDImitation::evolve),
                 "Execute the moran process with imitation once.")
            .def("evolve", static_cast<float (PDImitation::*)(unsigned int, float)>(&PDImitation::evolve),
                 "Find the stationary distribution for beta.")
            .def("evolve", static_cast<std::vector<float> (PDImitation::*)(std::vector<float>,
                                                                           unsigned int) >(&PDImitation::evolve),
                 "Find the stationary distribution for a range of betas.");

    py::class_<TraulsenMoran>(m, "TraulsenMoran")
            .def(py::init<uint64_t, unsigned int, unsigned int, double, double, double, double, Eigen::Ref<const MatrixXd>>())
            .def_property("generations", &TraulsenMoran::generations, &TraulsenMoran::set_generations)
            .def_property("n", &TraulsenMoran::group_size, &TraulsenMoran::set_group_size)
            .def_property("m", &TraulsenMoran::nb_groups, &TraulsenMoran::set_nb_groups)
            .def_property("pop_size", &TraulsenMoran::pop_size, &TraulsenMoran::set_pop_size)
            .def_property("nb_coop", &TraulsenMoran::nb_coop, &TraulsenMoran::set_nb_coop)
            .def_property_readonly("group_cooperation", &TraulsenMoran::group_cooperation)
            .def_property("mu", &TraulsenMoran::mu, &TraulsenMoran::set_mu)
            .def_property("beta", &TraulsenMoran::beta, &TraulsenMoran::set_beta)
            .def_property("coop_freq", &TraulsenMoran::coop_freq, &TraulsenMoran::set_coop_freq)
            .def_property_readonly("result_coop_freq", &TraulsenMoran::result_coop_freq)
            .def_property("payoff_matrix", &TraulsenMoran::payoff_matrix, &TraulsenMoran::set_payoff_matrix)
            .def("update_payoff_matrix", &TraulsenMoran::set_payoff_matrix, py::return_value_policy::reference_internal)
            .def("evolve", static_cast<double (TraulsenMoran::*)(double)>(&TraulsenMoran::evolve),
                 "Execute the moran process with imitation once.")
            .def("evolve", static_cast<double (TraulsenMoran::*)(unsigned int, double)>(&TraulsenMoran::evolve),
                 "Find the stationary distribution for beta.")
            .def("evolve", static_cast<std::vector<double> (TraulsenMoran::*)(std::vector<double>,
                                                                              unsigned int) >(&TraulsenMoran::evolve),
                 "Find the stationary distribution for a range of betas.");

    py::class_<StochDynamics>(m, "StochDynamics")
            .def(py::init<unsigned int, unsigned int, Eigen::Ref<MatrixXd>>())
            .def_property("pop_size", &StochDynamics::pop_size, &StochDynamics::set_pop_size)
            .def_property("nb_strategies", &StochDynamics::nb_strategies, &StochDynamics::set_nb_strategies)
            .def_property("payoff_matrix", &StochDynamics::payoff_matrix, &StochDynamics::set_payoff_matrix)
            .def("prop_increase_decrease", &StochDynamics::probIncreaseDecrease,
                 "Calculate the probability of incresing and decreasing the number of invaders")
            .def("calculate_transition_fixations", &StochDynamics::calculate_transition_fixations,
                 "Calculate the transition probabilities and the stationary distribution");

}