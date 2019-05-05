//
// Created by Elias Fernandez on 2019-02-11.
//

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include <Dyrwin/SED/PDImitation.h>
#include <Dyrwin/SED/StochDynamics.h>
#include <Dyrwin/SED/TraulsenMoran.h>
#include <Dyrwin/SED/MoranProcess.hpp>
#include <Dyrwin/SED/MLS.hpp>
#include <Dyrwin/SeedGenerator.h>

namespace py = pybind11;
using namespace EGTTools;

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

    // Use this function to get access to the singleton
//    m.def("get_seed_generator_instance",
//          &Random::SeedGenerator::getInstance,
//          py::return_value_policy::reference,
//          "Get reference to the seed generator singleton");

    py::class_<Random::SeedGenerator>(m, "Random")
            .def("init", []() {
                return std::unique_ptr<Random::SeedGenerator, py::nodelete>(&Random::SeedGenerator::getInstance());
            })
            .def("getSeed",
                 &Random::SeedGenerator::getMainSeed, "seed",
                 "Returns current seed"
            )
            .def("seed", &Random::SeedGenerator::getMainSeed,
                 "Set main seed");

//    py::class_<Random::SeedGenerator>(m, "Random")
//    .def("init", [](){
//        return std::unique_ptr<Random::SeedGenerator, py::nodelete>(&Random::SeedGenerator::getInstance());
//    });

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
            .def_property("mu", &TraulsenMoran::mu, &TraulsenMoran::set_mu)
            .def_property("beta", &TraulsenMoran::beta, &TraulsenMoran::set_beta)
            .def_property("coop_freq", &TraulsenMoran::coop_freq, &TraulsenMoran::set_coop_freq)
            .def_property_readonly("result_coop_freq", &TraulsenMoran::result_coop_freq)
            .def_property("payoff_matrix", &TraulsenMoran::payoff_matrix, &TraulsenMoran::set_payoff_matrix)
            .def("update_payoff_matrix", &TraulsenMoran::set_payoff_matrix, py::return_value_policy::reference_internal)
            .def("evolve", static_cast<double (TraulsenMoran::*)(double)>(&TraulsenMoran::evolve),
                 "Execute the moran process with imitation once.")
            .def("evolve", static_cast<double (TraulsenMoran::*)(unsigned int, double)>(&TraulsenMoran::evolve),
                 py::call_guard<py::gil_scoped_release>(),
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

    py::class_<MoranProcess>(m, "MoranProcess")
            .def(py::init<size_t, size_t, size_t, double, Eigen::Ref<const Vector>, Eigen::Ref<const Matrix2D>>())
            .def(py::init<size_t, size_t, size_t, size_t, double, Eigen::Ref<const Vector>, Eigen::Ref<const Matrix2D>>())
            .def(py::init<size_t, size_t, size_t, size_t, double, double, Eigen::Ref<const Vector>, Eigen::Ref<const Matrix2D>>())
            .def(py::init<size_t, size_t, size_t, size_t, double, double, double, Eigen::Ref<const Vector>, Eigen::Ref<const Matrix2D>>())
            .def_property("generations", &MoranProcess::generations, &MoranProcess::set_generations)
            .def_property_readonly("nb_strategies", &MoranProcess::nb_strategies)
            .def_property("n", &MoranProcess::group_size, &MoranProcess::set_group_size)
            .def_property("m", &MoranProcess::nb_groups, &MoranProcess::set_nb_groups)
            .def_property("pop_size", &MoranProcess::pop_size, &MoranProcess::set_pop_size)
            .def_property("mu", &MoranProcess::mu, &MoranProcess::set_mu)
            .def_property("beta", &MoranProcess::beta, &MoranProcess::set_beta)
            .def_property("init_freq", &MoranProcess::init_strategy_freq, &MoranProcess::set_strategy_freq)
            .def_property("init_state", &MoranProcess::init_strategy_count, &MoranProcess::set_strategy_count)
            .def_property_readonly("payoff_matrix", &MoranProcess::payoff_matrix)
            .def("update_payoff_matrix", &MoranProcess::set_payoff_matrix, py::return_value_policy::reference_internal)
            .def("evolve", static_cast<Vector (MoranProcess::*)(double)>(&MoranProcess::evolve),
                 "Execute the moran process with imitation once.")
            .def("evolve", static_cast<Vector (MoranProcess::*)(size_t, double)>(&MoranProcess::evolve),
                 py::call_guard<py::gil_scoped_release>(),
                 "Find the stationary distribution for beta.")
            .def("__repr__", &MoranProcess::toString);

    py::class_<SED::MLS<SED::Group>>(m, "MLS")
            .def(py::init<size_t, size_t, size_t, size_t, double, const Eigen::Ref<const Vector> &, const Eigen::Ref<const Matrix2D> &>())
            .def_property("generations", &SED::MLS<SED::Group>::generations, &SED::MLS<SED::Group>::set_generations)
            .def_property_readonly("nb_strategies", &SED::MLS<SED::Group>::nb_strategies)
            .def_property("n", &SED::MLS<SED::Group>::group_size, &SED::MLS<SED::Group>::set_group_size)
            .def_property("m", &SED::MLS<SED::Group>::nb_groups, &SED::MLS<SED::Group>::set_nb_groups)
            .def_property("init_freq", &SED::MLS<SED::Group>::init_strategy_freq,
                          &SED::MLS<SED::Group>::set_strategy_freq)
            .def_property("init_state", &SED::MLS<SED::Group>::init_strategy_count,
                          &SED::MLS<SED::Group>::set_strategy_count)
            .def_property_readonly("payoff_matrix", &SED::MLS<SED::Group>::payoff_matrix)
            .def_property_readonly("max_pop_size", &SED::MLS<SED::Group>::max_pop_size)
            .def("update_payoff_matrix", &SED::MLS<SED::Group>::set_payoff_matrix,
                 py::return_value_policy::reference_internal)
            .def("evolve", &SED::MLS<SED::Group>::evolve)
            .def("fixation_probability",
                 static_cast<double (SED::MLS<SED::Group>::*)(size_t, size_t, size_t, double,
                                                              double)>( &SED::MLS<SED::Group>::fixationProbability),
                 py::call_guard<py::gil_scoped_release>(),
                 "Calculates the fixation probability given a beta.")
            .def("fixation_probability",
                 static_cast<double (SED::MLS<SED::Group>::*)(size_t, size_t, size_t, double, double,
                                                              double)>( &SED::MLS<SED::Group>::fixationProbability),
                 py::call_guard<py::gil_scoped_release>(),
                 "Calculates the fixation probability given a beta and lambda.")
            .def("fixation_probability",
                 static_cast<Vector (SED::MLS<SED::Group>::*)(size_t, const Eigen::Ref<const VectorXui> &, size_t,
                                                              double,
                                                              double)>( &SED::MLS<SED::Group>::fixationProbability),
                 py::call_guard<py::gil_scoped_release>(),
                 "Calculates the fixation probability given a beta and an initial state.")
//            .def("fixation_probability",
//                 static_cast<double (SED::MLS<SED::Group>::*)(size_t, size_t, size_t, size_t, double, double, double,
//                                                              double)>(&SED::MLS<SED::Group>::fixationProbability),
//                 py::call_guard<py::gil_scoped_release>(),
//                 "Calculates the fixation probability given a beta, lambda and mu.")
            .def("gradient_selection",
                 static_cast<Vector (SED::MLS<SED::Group>::*)(size_t, size_t, size_t, double,
                                                              double)>(&SED::MLS<SED::Group>::gradientOfSelection),
                 py::call_guard<py::gil_scoped_release>(),
                 "Calculates the gradient of selection between two strategies.")
            .def("gradient_selection",
                 static_cast<Vector (SED::MLS<SED::Group>::*)(size_t, size_t, const Eigen::Ref<const VectorXui> &,
                                                              size_t, double,
                                                              double)>(&SED::MLS<SED::Group>::gradientOfSelection),
                 py::call_guard<py::gil_scoped_release>(),
                 "Calculates the gradient of selection for an invading strategy and any initial state.")
            .def("__repr__", &SED::MLS<SED::Group>::toString);

}