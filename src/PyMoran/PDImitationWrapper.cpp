//
// Created by Elias Fernandez on 2019-02-11.
//

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>
#include "../../include/Dyrwin/PyMoran/PDImitation.h"

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
            .def(py::init<unsigned int, unsigned int, float, float, float, std::vector<float>>())
            .def_property("generations", &PDImitation::generations, &PDImitation::set_generations)
            .def_property("pop_size", &PDImitation::pop_size, &PDImitation::set_pop_size)
            .def_property_readonly("nb_coop", &PDImitation::nb_coop)
            .def_property("mu", &PDImitation::mu, &PDImitation::set_mu)
            .def_property("beta", &PDImitation::beta, &PDImitation::set_beta)
            .def_property("coop_freq", &PDImitation::coop_freq, &PDImitation::set_coop_freq)
            .def_property_readonly("result_coop_freq", &PDImitation::result_coop_freq)
            .def_property("payoff_matrix", &PDImitation::payoff_matrix, &PDImitation::set_payoff_matrix)
            .def("evolve", static_cast<float (PDImitation::*)(float)>(&PDImitation::evolve),
                 "Execute the moran process with imitation once.")
            .def("evolve", static_cast<float (PDImitation::*)(unsigned int, float)>(&PDImitation::evolve),
                 "Find the stationary distribution for beta.")
            .def("evolve", static_cast<std::vector<float> (PDImitation::*)(std::vector<float>,
                                                                           unsigned int) >(&PDImitation::evolve),
                 "Find the stationary distribution for a range of betas.");
}