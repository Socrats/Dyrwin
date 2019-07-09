//
// Created by Elias Fernandez on 2019-07-08.
//

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include <Dyrwin/SED/PairwiseMoran.hpp>
#include <Dyrwin/SED/games/AbstractGame.hpp>
#include <Dyrwin/SED/games/CrdGame.hpp>
#include <Dyrwin/SED/games/CrdGameTU.hpp>
#include <Dyrwin/SED/behaviors/CrdBehaviors.hpp>
#include <Dyrwin/LruCache.hpp>

namespace py = pybind11;
using PairwiseComparison = EGTTools::SED::PairwiseMoran<EGTTools::Utils::LRUCache<std::string, double>>;

PYBIND11_MODULE(EGTtools, m) {
    py::class_<EGTTools::SED::AbstractGame>(m, "AbstractGame")
            .def("play", &EGTTools::SED::AbstractGame::play)
            .def("calculate_payoffs", &EGTTools::SED::AbstractGame::calculate_payoffs)
            .def("calculate_fitness", &EGTTools::SED::AbstractGame::calculate_fitness)
            .def("to_string", &EGTTools::SED::AbstractGame::toString)
            .def("type", &EGTTools::SED::AbstractGame::type)
            .def("payoffs", &EGTTools::SED::AbstractGame::payoffs)
            .def("save_payoffs", &EGTTools::SED::AbstractGame::save_payoffs);

    // Now we define a submodule
    auto mCRD = m.def_submodule("CRD");

    py::class_<EGTTools::SED::CRD::CrdGame, EGTTools::SED::AbstractGame>(mCRD, "CRDGame")
            .def(py::init<size_t, size_t, size_t, size_t, double>(), "Collective-risk game", py::arg("endowment"),
                 py::arg("threshold"), py::arg("nb_rounds"), py::arg("group_size"), py::arg("risk"))
            .def("play", &EGTTools::SED::CRD::CrdGame::play)
            .def("calculate_payoffs", &EGTTools::SED::CRD::CrdGame::calculate_payoffs)
            .def("calculate_fitness", &EGTTools::SED::CRD::CrdGame::calculate_fitness)
            .def("to_string", &EGTTools::SED::CRD::CrdGame::toString)
            .def("type", &EGTTools::SED::CRD::CrdGame::type)
            .def("payoffs", &EGTTools::SED::CRD::CrdGame::payoffs)
            .def("save_payoffs", &EGTTools::SED::CRD::CrdGame::save_payoffs);

    py::class_<EGTTools::SED::CRD::CrdGameTU, EGTTools::SED::AbstractGame>(mCRD, "CRDGameTU")
            .def(py::init<size_t, size_t, size_t, size_t, double, EGTTools::TimingUncertainty<std::mt19937_64> &>(),
                 "Collective-risk with Timing uncertainty",
                 py::arg("endowment"),
                 py::arg("threshold"), py::arg("nb_rounds"), py::arg("group_size"), py::arg("risk"),
                 py::arg("timing_uncertainty_object"), py::keep_alive<1, 7>())
            .def("play", &EGTTools::SED::CRD::CrdGameTU::play)
            .def("calculate_payoffs", &EGTTools::SED::CRD::CrdGameTU::calculate_payoffs)
            .def("calculate_fitness", &EGTTools::SED::CRD::CrdGameTU::calculate_fitness)
            .def("to_string", &EGTTools::SED::CRD::CrdGameTU::toString)
            .def("type", &EGTTools::SED::CRD::CrdGameTU::type)
            .def("payoffs", &EGTTools::SED::CRD::CrdGameTU::payoffs)
            .def("save_payoffs", &EGTTools::SED::CRD::CrdGameTU::save_payoffs);

    py::class_<PairwiseComparison>(m, "PairwiseMoran")
            .def(py::init<size_t, EGTTools::SED::AbstractGame &>(),
                 "Runs a moran process with pairwise comparison and calculates fitness according to game",
                 py::arg("pop_size"), py::arg("game"), py::keep_alive<1, 3>())
            .def("evolve", &PairwiseComparison::evolve, py::keep_alive<1, 5>(),
                 "evolves the strategies for a maximum of nb_generations", py::arg("nb_generations"), py::arg("beta"),
                 py::arg("mu"), py::arg("init_state"))
            .def("fixation_probability", &PairwiseComparison::fixationProbability,
                 "Estimates the fixation probability of an strategy in the population.",
                 py::arg("mutant"), py::arg("resident"), py::arg("nb_runs"), py::arg("nb_generations"), py::arg("beta"),
                 py::arg("mu"))
            .def("stationary_distribution", &PairwiseComparison::stationaryDistribution,
                 py::call_guard<py::gil_scoped_release>(),
                 "Estimates the stationary distribution of the population of strategies given the game.",
                 py::arg("nb_runs"), py::arg("nb_generations"), py::arg("beta"), py::arg("mu"))
            .def_property_readonly("nb_strategies", &PairwiseComparison::nb_strategies)
            .def_property_readonly("payoffs", &PairwiseComparison::payoffs)
            .def_property("pop_size", &PairwiseComparison::population_size,
                          &PairwiseComparison::set_population_size);

    mCRD.def("cooperator", &EGTTools::SED::CRD::cooperator, "returns the actions of a cooperator player.",
             py::arg("prev_donation"), py::arg("threshold"), py::arg("current_round"));
    mCRD.def("defector", &EGTTools::SED::CRD::defector, "returns the actions of a defector player.",
             py::arg("prev_donation"), py::arg("threshold"), py::arg("current_round"));
    mCRD.def("altruist", &EGTTools::SED::CRD::altruist, "returns the actions of a altruist player.",
             py::arg("prev_donation"), py::arg("threshold"), py::arg("current_round"));
    mCRD.def("reciprocal", &EGTTools::SED::CRD::reciprocal, "returns the actions of a reciprocal player.",
             py::arg("prev_donation"), py::arg("threshold"), py::arg("current_round"));
    mCRD.def("compensator", &EGTTools::SED::CRD::compensator, "returns the actions of a compensator player.",
             py::arg("prev_donation"), py::arg("threshold"), py::arg("current_round"));

    // Bind the enum class so that it's clear from python which are the indexes of each strategy
    py::enum_<EGTTools::SED::CRD::CRDBehaviors>(mCRD, "CRDBehaviors", py::arithmetic(), "Indexes of CRD behaviors")
            .value("Cooperator", EGTTools::SED::CRD::CRDBehaviors::cooperator)
            .value("Defector", EGTTools::SED::CRD::CRDBehaviors::defector)
            .value("Altruist", EGTTools::SED::CRD::CRDBehaviors::altruist)
            .value("Reciprocal", EGTTools::SED::CRD::CRDBehaviors::reciprocal)
            .value("Compensator", EGTTools::SED::CRD::CRDBehaviors::compensator);
}