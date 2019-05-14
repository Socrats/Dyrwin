//
// Created by Elias Fernandez on 2019-02-11.
//

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include <Dyrwin/SeedGenerator.h>
#include <Dyrwin/SED/PDImitation.h>
#include <Dyrwin/SED/StochDynamics.h>
#include <Dyrwin/SED/TraulsenMoran.h>
#include <Dyrwin/SED/MoranProcess.hpp>
#include <Dyrwin/SED/MLS.hpp>
#include <Dyrwin/RL/Agent.h>
#include <Dyrwin/RL/BatchQLearningAgent.h>
#include <Dyrwin/RL/QLearningAgent.h>
#include <Dyrwin/RL/RothErevAgent.h>
#include <Dyrwin/RL/HistericQLearningAgent.hpp>
#include <Dyrwin/RL/CRDGame.h>
#include <Dyrwin/RL/CRDDemocracy.h>
#include <Dyrwin/RL/CRDConditional.h>
#include <Dyrwin/RL/CrdSim.hpp>

//PYBIND11_MAKE_OPAQUE(EGTTools::RL::PopContainer);

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
    py::class_<Random::SeedGenerator, std::unique_ptr<Random::SeedGenerator, py::nodelete>>(m, "Random")
            .def("init", []() {
                return std::unique_ptr<Random::SeedGenerator, py::nodelete>(&Random::SeedGenerator::getInstance());
            })
            .def("getSeed",
                 &Random::SeedGenerator::getMainSeed, "seed",
                 "Returns current seed"
            )
            .def("seed", &Random::SeedGenerator::getMainSeed,
                 "Set main seed");

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
            .def_property_readonly("payoff_matrix", &SED::MLS<SED::Group>::payoff_matrix,
                                   py::return_value_policy::reference_internal)
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

    // Now we define a submodule
    auto mRL = m.def_submodule("RL");
    py::class_<RL::Agent>(mRL, "Agent")
            .def(py::init<size_t, size_t, size_t, double>(), "Implementation of a Roth-Erev RL without forgetting",
                 py::arg("nb_states"), py::arg("nb_actions"), py::arg("episode_length"), py::arg("endowment"))
            .def_property("endowment", &RL::Agent::endowment, &RL::Agent::set_endowment)
            .def_property("nb_states", &RL::Agent::nb_states, &RL::Agent::set_nb_states)
            .def_property("nb_actions", &RL::Agent::nb_actions, &RL::Agent::set_nb_actions)
            .def_property("episode_length", &RL::Agent::episode_length, &RL::Agent::set_episode_length)
            .def_property_readonly("payoff", &RL::Agent::payoff)
            .def_property_readonly("trajectory_states", &RL::Agent::trajectoryStates,
                                   py::return_value_policy::reference_internal)
            .def_property_readonly("trajectory_actions", &RL::Agent::trajectoryActions,
                                   py::return_value_policy::reference_internal)
            .def_property_readonly("q_values", &RL::Agent::qValues, py::return_value_policy::reference_internal)
            .def_property_readonly("policy", &RL::Agent::policy, py::return_value_policy::reference_internal)
            .def("reset", &RL::Agent::reset)
            .def("set_q_values", &RL::Agent::set_q_values)
            .def("set_policy", &RL::Agent::set_policy)
            .def("decrease_payoff", &RL::Agent::decrease)
            .def("reset_payoff", &RL::Agent::resetPayoff)
            .def("update_policy", &RL::Agent::inferPolicy)
            .def("reset_trajectory", &RL::Agent::resetTrajectory)
            .def("reinforce", static_cast<void (RL::Agent::*)()>(&RL::Agent::reinforceTrajectory),
                 "reinforces the actions from the current trajectory, based on the agent's reward")
            .def("reinforce", static_cast<void (RL::Agent::*)(size_t)>(&RL::Agent::reinforceTrajectory),
                 "reinforces the actions from the current trajectory, based on the agent's reward",
                 py::arg("episode_length"))
            .def("act", static_cast<size_t (RL::Agent::*)(size_t)>(&RL::Agent::selectAction),
                 "samples an action from the agent's policy", py::arg("round"))
            .def("act", static_cast<size_t (RL::Agent::*)(size_t, size_t)>(&RL::Agent::selectAction),
                 "samples an action from the agent's policy", py::arg("round"), py::arg("state"))
            .def("reset_q_values", &RL::Agent::resetQValues)
            .def("__repr__", &RL::Agent::toString);

    py::class_<RL::BatchQLearningAgent, RL::Agent>(mRL, "BatchQLearningAgent")
            .def(py::init<size_t, size_t, size_t, double, double, double>(),
                 "Implementation of the Batch Q-Learning algorithm.",
                 py::arg("nb_states"), py::arg("nb_actions"),
                 py::arg("episode_length"), py::arg("endowment"), py::arg("alpha"), py::arg("temperature"))
            .def_property("alpha", &RL::BatchQLearningAgent::alpha, &RL::BatchQLearningAgent::setAlpha)
            .def_property("temperature", &RL::BatchQLearningAgent::temperature,
                          &RL::BatchQLearningAgent::setTemperature)
            .def("update_policy", &RL::BatchQLearningAgent::inferPolicy)
            .def("reset_trajectory", &RL::BatchQLearningAgent::resetTrajectory)
            .def("reinforce",
                 static_cast<void (RL::BatchQLearningAgent::*)()>(&RL::BatchQLearningAgent::reinforceTrajectory),
                 "reinforces the actions from the current trajectory, based on the agent's reward")
            .def("reinforce",
                 static_cast<void (RL::BatchQLearningAgent::*)(size_t)>(&RL::BatchQLearningAgent::reinforceTrajectory),
                 "reinforces the actions from the current trajectory, based on the agent's reward",
                 py::arg("episode_length"))
            .def("act",
                 static_cast<size_t (RL::BatchQLearningAgent::*)(size_t)>(&RL::BatchQLearningAgent::selectAction),
                 "samples an action from the agent's policy", py::arg("round"))
            .def("act", static_cast<size_t (RL::BatchQLearningAgent::*)(size_t,
                                                                        size_t)>(&RL::BatchQLearningAgent::selectAction),
                 "samples an action from the agent's policy", py::arg("round"), py::arg("state"))
            .def("reset_q_values", &RL::BatchQLearningAgent::resetQValues);

    py::class_<RL::QLearningAgent, RL::Agent>(mRL, "QLearningAgent")
            .def(py::init<size_t, size_t, size_t, double, double, double, double>(),
                 "Implementation of the Q-Learning algorithm.",
                 py::arg("nb_states"), py::arg("nb_actions"),
                 py::arg("episode_length"), py::arg("endowment"),
                 py::arg("alpha"), py::arg("lambda"), py::arg("temperature"))
            .def_property("alpha", &RL::QLearningAgent::alpha, &RL::QLearningAgent::setAlpha)
            .def_property("lambda", &RL::QLearningAgent::lambda, &RL::QLearningAgent::setLambda)
            .def_property("temperature", &RL::QLearningAgent::temperature,
                          &RL::QLearningAgent::setTemperature)
            .def("update_policy", &RL::QLearningAgent::inferPolicy)
            .def("reset_trajectory", &RL::QLearningAgent::resetTrajectory)
            .def("reinforce",
                 static_cast<void (RL::QLearningAgent::*)()>(&RL::QLearningAgent::reinforceTrajectory),
                 "reinforces the actions from the current trajectory, based on the agent's reward")
            .def("reinforce",
                 static_cast<void (RL::QLearningAgent::*)(size_t)>(&RL::QLearningAgent::reinforceTrajectory),
                 "reinforces the actions from the current trajectory, based on the agent's reward",
                 py::arg("episode_length"))
            .def("act",
                 static_cast<size_t (RL::QLearningAgent::*)(size_t)>(&RL::QLearningAgent::selectAction),
                 "samples an action from the agent's policy", py::arg("round"))
            .def("act", static_cast<size_t (RL::QLearningAgent::*)(size_t,
                                                                   size_t)>(&RL::QLearningAgent::selectAction),
                 "samples an action from the agent's policy", py::arg("round"), py::arg("state"))
            .def("reset_q_values", &RL::QLearningAgent::resetQValues);

    py::class_<RL::HistericQLearningAgent, RL::Agent>(mRL, "HistericQLearningAgent")
            .def(py::init<size_t, size_t, size_t, double, double, double, double>(),
                 "Implementation of the Histeric Q-Learning algorithm.",
                 py::arg("nb_states"), py::arg("nb_actions"),
                 py::arg("episode_length"), py::arg("endowment"),
                 py::arg("alpha"), py::arg("beta"), py::arg("temperature"))
            .def_property("alpha", &RL::HistericQLearningAgent::alpha, &RL::HistericQLearningAgent::setAlpha)
            .def_property("beta", &RL::HistericQLearningAgent::beta, &RL::HistericQLearningAgent::setBeta)
            .def_property("temperature", &RL::HistericQLearningAgent::temperature,
                          &RL::HistericQLearningAgent::setTemperature)
            .def("update_policy", &RL::HistericQLearningAgent::inferPolicy)
            .def("reset_trajectory", &RL::HistericQLearningAgent::resetTrajectory)
            .def("reinforce",
                 static_cast<void (RL::HistericQLearningAgent::*)()>(&RL::HistericQLearningAgent::reinforceTrajectory),
                 "reinforces the actions from the current trajectory, based on the agent's reward")
            .def("reinforce",
                 static_cast<void (RL::HistericQLearningAgent::*)(
                         size_t)>(&RL::HistericQLearningAgent::reinforceTrajectory),
                 "reinforces the actions from the current trajectory, based on the agent's reward",
                 py::arg("episode_length"))
            .def("act",
                 static_cast<size_t (RL::HistericQLearningAgent::*)(size_t)>(&RL::HistericQLearningAgent::selectAction),
                 "samples an action from the agent's policy", py::arg("round"))
            .def("act", static_cast<size_t (RL::HistericQLearningAgent::*)(size_t,
                                                                           size_t)>(&RL::HistericQLearningAgent::selectAction),
                 "samples an action from the agent's policy", py::arg("round"), py::arg("state"))
            .def("reset_q_values", &RL::HistericQLearningAgent::resetQValues);

    py::class_<RL::RothErevAgent, RL::Agent>(mRL, "RothErevAgent")
            .def(py::init<size_t, size_t, size_t, double, double, double>(),
                 "Implementation of the Histeric Q-Learning algorithm.",
                 py::arg("nb_states"), py::arg("nb_actions"),
                 py::arg("episode_length"), py::arg("endowment"),
                 py::arg("lambda"), py::arg("temperature"))
            .def_property("lambda", &RL::RothErevAgent::lambda, &RL::RothErevAgent::setLambda)
            .def_property("temperature", &RL::RothErevAgent::temperature,
                          &RL::RothErevAgent::setTemperature)
            .def("update_policy", &RL::RothErevAgent::inferPolicy)
            .def("reset_trajectory", &RL::RothErevAgent::resetTrajectory)
            .def("reinforce",
                 static_cast<void (RL::RothErevAgent::*)()>(&RL::RothErevAgent::reinforceTrajectory),
                 "reinforces the actions from the current trajectory, based on the agent's reward")
            .def("reinforce",
                 static_cast<void (RL::RothErevAgent::*)(size_t)>(&RL::RothErevAgent::reinforceTrajectory),
                 "reinforces the actions from the current trajectory, based on the agent's reward",
                 py::arg("episode_length"))
            .def("act",
                 static_cast<size_t (RL::RothErevAgent::*)(size_t)>(&RL::RothErevAgent::selectAction),
                 "samples an action from the agent's policy", py::arg("round"))
            .def("act", static_cast<size_t (RL::RothErevAgent::*)(size_t,
                                                                  size_t)>(&RL::RothErevAgent::selectAction),
                 "samples an action from the agent's policy", py::arg("round"), py::arg("state"))
            .def("reset_q_values", &RL::RothErevAgent::resetQValues)
            .def("__repr__", &RL::RothErevAgent::toString);

    py::class_<RL::CRDGame<RL::Agent>>(mRL, "CRDGame")
            .def(py::init<>(), "Collective-risk dilemma without uncertainty")
            .def("play", static_cast<std::pair<double, size_t> (RL::CRDGame<RL::Agent>::*)(std::vector<RL::Agent> &,
                                                                                           EGTTools::RL::ActionSpace &,
                                                                                           size_t)>(&RL::CRDGame<RL::Agent>::playGame),
                 "plays the game for a fixed number of rounds",
                 py::arg("players"), py::arg("actions"), py::arg("rounds"))
            .def("reinforce_players", static_cast<bool (RL::CRDGame<RL::Agent>::*)(
                         std::vector<RL::Agent> &)>(&RL::CRDGame<RL::Agent>::reinforcePath),
                 "reinforces the actions of all agents in the group", py::arg("players"))
            .def("update_strategies", static_cast<bool (RL::CRDGame<RL::Agent>::*)(
                         std::vector<RL::Agent> &)>(&RL::CRDGame<RL::Agent>::calcProbabilities),
                 "updates the payoffs of the players in the group", py::arg("players"))
            .def("reset_episode", static_cast<bool (RL::CRDGame<RL::Agent>::*)(
                         std::vector<RL::Agent> &)>(&RL::CRDGame<RL::Agent>::resetEpisode),
                 "resets the trajectories of the players", py::arg("players"))
            .def("group_payoff", static_cast<double (RL::CRDGame<RL::Agent>::*)(
                         std::vector<RL::Agent> &)>(&RL::CRDGame<RL::Agent>::playersPayoff),
                 "returns the total payoff of the group", py::arg("players"))
            .def("group_contributions", static_cast<double (RL::CRDGame<RL::Agent>::*)(
                         std::vector<RL::Agent> &)>(&RL::CRDGame<RL::Agent>::playersContribution),
                 "returns the total contribution of the group", py::arg("players"))
            .def("set_payoffs", static_cast<void (RL::CRDGame<RL::Agent>::*)(
                         std::vector<RL::Agent> &, unsigned int)>(&RL::CRDGame<RL::Agent>::setPayoffs),
                 "sets players payoffs", py::arg("players"), py::arg("payoff"));

    py::class_<RL::CRDGame<RL::PopContainer>>(mRL, "CRDGameGeneric")
            .def(py::init<>(), "Collective-risk dilemma without uncertainty")
            .def("play", static_cast<std::pair<double, size_t> (RL::CRDGame<RL::PopContainer>::*)(RL::PopContainer &,
                                                                                                  EGTTools::RL::ActionSpace &,
                                                                                                  size_t)>(&RL::CRDGame<RL::PopContainer>::playGame),
                 "plays the game for a fixed number of rounds",
                 py::arg("players"), py::arg("actions"), py::arg("rounds"))
            .def("reinforce_players", static_cast<bool (RL::CRDGame<RL::PopContainer>::*)(
                         RL::PopContainer &)>(&RL::CRDGame<RL::PopContainer>::reinforcePath),
                 "reinforces the actions of all agents in the group", py::arg("players"))
            .def("update_strategies", static_cast<bool (RL::CRDGame<RL::PopContainer>::*)(
                         RL::PopContainer &)>(&RL::CRDGame<RL::PopContainer>::calcProbabilities),
                 "updates the payoffs of the players in the group", py::arg("players"))
            .def("reset_episode", static_cast<bool (RL::CRDGame<RL::PopContainer>::*)(
                         RL::PopContainer &)>(&RL::CRDGame<RL::PopContainer>::resetEpisode),
                 "resets the trajectories of the players", py::arg("players"))
            .def("group_payoff", static_cast<double (RL::CRDGame<RL::PopContainer>::*)(
                         RL::PopContainer &)>(&RL::CRDGame<RL::PopContainer>::playersPayoff),
                 "returns the total payoff of the group", py::arg("players"))
            .def("group_contributions", static_cast<double (RL::CRDGame<RL::PopContainer>::*)(
                         RL::PopContainer &)>(&RL::CRDGame<RL::PopContainer>::playersContribution),
                 "returns the total contribution of the group", py::arg("players"))
            .def("set_payoffs", static_cast<void (RL::CRDGame<RL::PopContainer>::*)(
                         RL::PopContainer &, unsigned int)>(&RL::CRDGame<RL::PopContainer>::setPayoffs),
                 "sets players payoffs", py::arg("players"), py::arg("payoff"));

    py::class_<RL::PopContainer>(mRL, "Population")
            .def(py::init<const std::string &, size_t, size_t, size_t, size_t, double, std::vector<double>>(),
                 "Population container. This class allows for the initialization of a vector of players (that can be from"
                 "different child classes) that can be passed to a game class.",
                 py::arg("agent_type"), py::arg("nb_agents"),
                 py::arg("nb_states"), py::arg("nb_actions"),
                 py::arg("episode_length"), py::arg("endowment"), py::arg("*args"),
                 py::return_value_policy::reference_internal)
            .def("size", &RL::PopContainer::size, "returns the size of the contained vector")
            .def("reset", &RL::PopContainer::reset, "reset all individuals in the population")
            .def("__getitem__", [](RL::PopContainer &s, size_t i) -> RL::Agent & {
                if (i >= s.size()) throw py::index_error();
                return s[i];
            }, py::return_value_policy::reference_internal)
            .def("__len__", [](const RL::PopContainer &v) { return v.size(); })
            .def("__repr__", &RL::PopContainer::toString);

    py::class_<RL::CRDSim>(mRL, "CRDSim")
            .def(py::init<size_t, size_t, size_t, size_t, size_t, double, double,
                         double, const EGTTools::RL::ActionSpace &,
                         const std::string &, const std::vector<double> &>(),
                 "Performs a Collective-risk dilemma simulation similar to milinksi 2008 experiment.",
                 py::arg("nb_episodes"), py::arg("nb_games"),
                 py::arg("nb_rounds"), py::arg("nb_actions"),
                 py::arg("group_size"), py::arg("risk"),
                 py::arg("endowment"), py::arg("threshold"),
                 py::arg("available_actions"), py::arg("agent_type"), py::arg("*args"))
            .def("run", static_cast<EGTTools::Matrix2D (RL::CRDSim::*)(size_t, size_t)>(&RL::CRDSim::run),
                 "Runs a CRD simulation for a single group", py::arg("nb_episodes"), py::arg("nb_games"))
            .def("run", static_cast<EGTTools::Matrix2D (RL::CRDSim::*)(size_t, size_t, size_t, double,
                                                                       const std::vector<double> &)>(&RL::CRDSim::run),
                 "Runs a CRD simulation for nb_groups",
                 py::arg("nb_episodes"), py::arg("nb_games"), py::arg("nb_groups"), py::arg("risk"),
                 py::arg("*agent_args"))
            .def_readwrite("population", &RL::CRDSim::population, py::return_value_policy::reference_internal)
            .def("reset_population", &RL::CRDSim::resetPopulation)
            .def_property("nb_games", &RL::CRDSim::nb_games, &RL::CRDSim::set_nb_games)
            .def_property("nb_episodes", &RL::CRDSim::nb_episodes, &RL::CRDSim::set_nb_episodes)
            .def_property("nb_rounds", &RL::CRDSim::nb_rounds, &RL::CRDSim::set_nb_rounds)
            .def_property("nb_actions", &RL::CRDSim::nb_actions, &RL::CRDSim::set_nb_actions)
            .def_property("risk", &RL::CRDSim::risk, &RL::CRDSim::set_risk)
            .def_property("threshold", &RL::CRDSim::threshold, &RL::CRDSim::set_threshold)
            .def_property("available_actions", &RL::CRDSim::available_actions, &RL::CRDSim::set_available_actions,
                          py::return_value_policy::reference_internal)
            .def_property("agent_type", &RL::CRDSim::agent_type, &RL::CRDSim::set_agent_type)
            .def_property_readonly("endowment", &RL::CRDSim::endowment);
}