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
#include <Dyrwin/SED/structure/Group.hpp>
#include <Dyrwin/SED/structure/GarciaGroup.hpp>
#include <Dyrwin/SED/PairwiseMoran.hpp>
#include <Dyrwin/SED/games/AbstractGame.hpp>
#include <Dyrwin/SED/games/NormalFormGame.h>
#include <Dyrwin/SED/games/CrdGame.hpp>
#include <Dyrwin/SED/games/CrdGameTU.hpp>
#include <Dyrwin/SED/behaviors/CrdBehaviors.hpp>
#include <Dyrwin/LruCache.hpp>
#include <Dyrwin/RL/Agent.h>
#include <Dyrwin/RL/BatchQLearningAgent.h>
#include <Dyrwin/RL/BatchQLearningForgetAgent.h>
#include <Dyrwin/RL/QLearningAgent.h>
#include <Dyrwin/RL/SARSAAgent.h>
#include <Dyrwin/RL/RothErevAgent.h>
#include <Dyrwin/RL/HistericQLearningAgent.hpp>
#include <Dyrwin/RL/CRDGame.h>
//#include <Dyrwin/RL/CRDDemocracy.h>
#include <Dyrwin/RL/CRDConditional.h>
#include <Dyrwin/RL/CrdSim.hpp>
#include <Dyrwin/RL/Data.hpp>
#include <Dyrwin/RL/Utils.h>
#include <Dyrwin/RL/DiscountedQLearning.h>
#include <Dyrwin/RL/simulators/CrdIslands.h>

namespace py = pybind11;
using namespace EGTTools;
using PairwiseComparison = EGTTools::SED::PairwiseMoran<EGTTools::Utils::LRUCache<std::string, double>>;
using crdData = EGTTools::RL::DataTypes::CRDData;
using crdDataIslands = EGTTools::RL::DataTypes::CRDDataIslands;
using DataTableCRD = EGTTools::RL::DataTypes::DataTableCRD;

PYBIND11_MODULE(EGTtools, m) {
  m.doc() = R"pbdoc(
        EGTtools: Efficient methods for modeling and studying Social Dynamics and Game Theory.
        This library is written in C++ (with python bindings) and pure Python.

        Note:
        Soon this library will be separated into two modules: EGTtools and LTtools which will contain
        methods for analysing social dynamics with evolutionary game theory (EGT) and Learning Theory (LT).
        The latter will be mostly focused on reinforcement learning (RL). These modules will be
        issues both separately and joined in the Dyrwin library.
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
      }, R"pbdoc(
            Initializes the random seed generator from random_device.
           )pbdoc")
      .def("init", [](unsigned long int seed) {
        auto instance = std::unique_ptr<Random::SeedGenerator, py::nodelete>(&Random::SeedGenerator::getInstance());
        instance->setMainSeed(seed);
        return instance;
      }, R"pbdoc(
            Initializes the random seed generator from seed.
           )pbdoc")
      .def_property_readonly_static("seed_", [](const py::object&) {
                                      return EGTTools::Random::SeedGenerator::getInstance().getMainSeed();
                                    }, R"pbdoc(Returns current seed)pbdoc"
      )
      .def_static("generate", []() {
        return EGTTools::Random::SeedGenerator::getInstance().getSeed();
      },"generates a random seed")
      .def_static("seed", [](unsigned long int seed){
        EGTTools::Random::SeedGenerator::getInstance().setMainSeed(seed);
      });

  py::class_<PDImitation>(m, "PDImitation")
      .def(py::init<unsigned int, unsigned int, float, float, float, const Eigen::Ref<const Matrix2D> &>())
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
      .def(py::init<size_t,
                    size_t,
                    size_t,
                    size_t,
                    double,
                    double,
                    Eigen::Ref<const Vector>,
                    Eigen::Ref<const Matrix2D>>())
      .def(py::init<size_t,
                    size_t,
                    size_t,
                    size_t,
                    double,
                    double,
                    double,
                    Eigen::Ref<const Vector>,
                    Eigen::Ref<const Matrix2D>>())
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
      .def(py::init<size_t,
                    size_t,
                    size_t,
                    size_t,
                    double,
                    const Eigen::Ref<const Vector> &,
                    const Eigen::Ref<const Matrix2D> &>(),
           "Implement the Multi-level Selection process described in [Traulsen & Nowak 2006].",
           py::arg("generations"), py::arg("nb_strategies"), py::arg("group_size"), py::arg("nb_groups"),
           py::arg("w"), py::arg("strategies_freq"), py::arg("payoff_matrix"))
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
           py::return_value_policy::reference_internal,
           "updates the payoff matrix with the values from the input.",
           py::arg("payoff_matrix"))
      .def("evolve", static_cast<EGTTools::Vector (SED::MLS<SED::Group>::*)(double, double,
                                                                            const Eigen::Ref<const EGTTools::VectorXui> &)>(&SED::MLS<
               SED::Group>::evolve),
           "runs the moran process with multi-level selection for a given number of generations or until"
           "it reaches a monomorphic state", py::arg("q"), py::arg("w"), py::arg("init_state"))
      .def("evolve", static_cast<EGTTools::Vector (SED::MLS<SED::Group>::*)(double, double, double,
                                                                            const Eigen::Ref<const EGTTools::VectorXui> &)>(&SED::MLS<
               SED::Group>::evolve),
           "runs the moran process with multi-level selection for a given number of generations or until"
           "it reaches a monomorphic state with migration", py::arg("q"), py::arg("w"), py::arg("lambda"),
           py::arg("init_state"))
      .def("evolve",
           static_cast<EGTTools::Vector (SED::MLS<SED::Group>::*)(double, double, double, double, double,
                                                                  const Eigen::Ref<const EGTTools::VectorXui> &)>(&SED::MLS<
               SED::Group>::evolve),
           "runs the moran process with multi-level selection for a given number of generations or until"
           "it reaches a monomorphic state with migration and direct multi-level selection (Garcia et al.)",
           py::arg("q"), py::arg("w"), py::arg("lambda"), py::arg("kappa"), py::arg("z"),
           py::arg("init_state"))
      .def("fixation_probability",
           static_cast<double (SED::MLS<SED::Group>::*)(size_t, size_t, size_t, double,
                                                        double)>( &SED::MLS<SED::Group>::fixationProbability),
           py::call_guard<py::gil_scoped_release>(),
           "Estimates the fixation probability of the invading strategy over the resident strategy for MLS.",
           py::arg("invader"), py::arg("resident"), py::arg("runs"), py::arg("q"), py::arg("w"))
      .def("fixation_probability",
           static_cast<double (SED::MLS<SED::Group>::*)(size_t, size_t, size_t, double, double,
                                                        double)>( &SED::MLS<SED::Group>::fixationProbability),
           py::call_guard<py::gil_scoped_release>(),
           "Estimates the fixation probability of the invading strategy over the resident strategy "
           "for MLS with migration.", py::arg("invader"), py::arg("resident"), py::arg("runs"), py::arg("q"),
           py::arg("lambda"), py::arg("w"))
      .def("fixation_probability",
           static_cast<double (SED::MLS<SED::Group>::*)(size_t, size_t, size_t, double, double, double, double,
                                                        double)>( &SED::MLS<SED::Group>::fixationProbability),
           py::call_guard<py::gil_scoped_release>(),
           "Estimates the fixation probability of the invading strategy over the resident strategy "
           "for MLS with migration and direct conflict.", py::arg("invader"), py::arg("resident"),
           py::arg("runs"), py::arg("q"),
           py::arg("lambda"), py::arg("w"), py::arg("kappa"), py::arg("z"))
      .def("fixation_probability",
           static_cast<Vector (SED::MLS<SED::Group>::*)(size_t, const Eigen::Ref<const VectorXui> &, size_t,
                                                        double,
                                                        double)>( &SED::MLS<SED::Group>::fixationProbability),
           py::call_guard<py::gil_scoped_release>(),
           "Estimates the fixation probability of the invading strategy over the resident strategy "
           "fir MLS from any initial state", py::arg("invader"), py::arg("init_state"), py::arg("runs"),
           py::arg("q"), py::arg("w"))
      .def("gradient_selection",
           static_cast<Vector (SED::MLS<SED::Group>::*)(size_t, size_t, size_t, double,
                                                        double)>(&SED::MLS<SED::Group>::gradientOfSelection),
           py::call_guard<py::gil_scoped_release>(),
           "Calculates the gradient of selection between two strategies.",
           py::arg("invader"), py::arg("resident"), py::arg("runs"), py::arg("w"), py::arg("a"))
      .def("gradient_selection",
           static_cast<Vector (SED::MLS<SED::Group>::*)(size_t, size_t, const Eigen::Ref<const VectorXui> &,
                                                        size_t, double,
                                                        double)>(&SED::MLS<SED::Group>::gradientOfSelection),
           py::call_guard<py::gil_scoped_release>(),
           "Calculates the gradient of selection for an invading strategy and any initial state.",
           py::arg("invader"), py::arg("strategy_to_reduce"), py::arg("init_state"),
           py::arg("runs"), py::arg("w"), py::arg("q"))
      .def("__repr__", &SED::MLS<SED::Group>::toString);

  py::class_<SED::MLS<SED::GarciaGroup>>(m, "GarciaMLS")
      .def(py::init<size_t, size_t, size_t, size_t>(),
           "Implement the Multi-level Selection process described in [Traulsen & Nowak 2006].",
           py::arg("generations"), py::arg("nb_strategies"), py::arg("group_size"), py::arg("nb_groups"))
      .def_property("generations", &SED::MLS<SED::GarciaGroup>::generations,
                    &SED::MLS<SED::GarciaGroup>::set_generations)
      .def_property_readonly("nb_strategies", &SED::MLS<SED::GarciaGroup>::nb_strategies)
      .def_property("n", &SED::MLS<SED::GarciaGroup>::group_size, &SED::MLS<SED::GarciaGroup>::set_group_size)
      .def_property("m", &SED::MLS<SED::GarciaGroup>::nb_groups, &SED::MLS<SED::GarciaGroup>::set_nb_groups)
      .def_property_readonly("max_pop_size", &SED::MLS<SED::GarciaGroup>::max_pop_size)
      .def("fixation_probability",
           static_cast<double (SED::MLS<SED::GarciaGroup>::*)(size_t, size_t, size_t,
                                                              double, double, double, double,
                                                              double, double,
                                                              const Eigen::Ref<const Matrix2D> &,
                                                              const Eigen::Ref<const Matrix2D> &)>( &SED::MLS<SED::GarciaGroup>::fixationProbability),
           py::keep_alive<1, 11>(), py::keep_alive<1, 12>(),
           py::call_guard<py::gil_scoped_release>(),
           "Estimates the fixation probability of the invading strategy over the resident strategy "
           "for MLS with migration and direct conflict and inter-group interactions.", py::arg("invader"),
           py::arg("resident"),
           py::arg("runs"), py::arg("q"),
           py::arg("lambda"), py::arg("w"), py::arg("alpha"),
           py::arg("kappa"), py::arg("z"), py::arg("payoff_matrix_in"), py::arg("payoff_matrix_out"))
      .def("__repr__", &SED::MLS<SED::GarciaGroup>::toString);

  py::class_<EGTTools::TimingUncertainty<std::mt19937_64>>(m, "TimingUncertainty")
      .def(py::init<double>(), "Timing uncertainty object", py::arg("p"))
      .def(py::init<double, size_t>(), "Timing uncertainty object", py::arg("p"), py::arg("max_rounds"))
      .def("calculateEnd", &EGTTools::TimingUncertainty<std::mt19937_64>::calculateEnd)
      .def("calculateFullEnd", &EGTTools::TimingUncertainty<std::mt19937_64>::calculateFullEnd)
      .def_property("p", &EGTTools::TimingUncertainty<std::mt19937_64>::probability,
                    &EGTTools::TimingUncertainty<std::mt19937_64>::setProbability)
      .def_property("max_rounds", &EGTTools::TimingUncertainty<std::mt19937_64>::max_rounds,
                    &EGTTools::TimingUncertainty<std::mt19937_64>::setMaxRounds);

  py::class_<EGTTools::SED::AbstractGame>(m, "AbstractGame")
      .def("play", &EGTTools::SED::AbstractGame::play)
      .def("calculate_payoffs", &EGTTools::SED::AbstractGame::calculate_payoffs)
      .def("calculate_fitness", &EGTTools::SED::AbstractGame::calculate_fitness)
      .def("to_string", &EGTTools::SED::AbstractGame::toString)
      .def("type", &EGTTools::SED::AbstractGame::type)
      .def("payoffs", &EGTTools::SED::AbstractGame::payoffs)
      .def("payoff", &EGTTools::SED::AbstractGame::payoff)
      .def_property_readonly("nb_strategies", &EGTTools::SED::AbstractGame::nb_strategies)
      .def("save_payoffs", &EGTTools::SED::AbstractGame::save_payoffs);

  m.def("calculate_state",
        static_cast<size_t (*)(const size_t &, const EGTTools::Factors &)>(&EGTTools::SED::calculate_state),
        "calculates an index given a simplex state",
        py::arg("group_size"), py::arg("group_composition"));
  m.def("calculate_state",
        static_cast<size_t (*)(const size_t &,
                               const Eigen::Ref<const EGTTools::VectorXui> &)>(&EGTTools::SED::calculate_state),
        "calculates an index given a simplex state",
        py::arg("group_size"), py::arg("group_composition"));
  m.def("sample_simplex",
        static_cast<EGTTools::VectorXui (*)(size_t, const size_t &, const size_t &)>(&EGTTools::SED::sample_simplex),
        "returns a point in the simplex given an index", py::arg("index"), py::arg("pop_size"),
        py::arg("nb_strategies"));
  m.def("calculate_nb_states",
        &EGTTools::starsBars,
        "calculates the number of states (combinations) of the members of a group in a subgroup.",
        py::arg("group_size"), py::arg("sub_group_size"));

  py::class_<EGTTools::SED::NormalFormGame, EGTTools::SED::AbstractGame>(m, "NormalFormGame")
      .def(py::init<size_t, const Eigen::Ref<const Matrix2D> &>(), "Normal Form Game", py::arg("nb_rounds"),
           py::arg("payoff_matrix"))
      .def("play", &EGTTools::SED::NormalFormGame::play)
      .def("calculate_payoffs", &EGTTools::SED::NormalFormGame::calculate_payoffs,
           "updates the internal payoff and coop_level matrices by calculating the payoff of each strategy "
           "given any possible strategy pair")
      .def("calculate_fitness", &EGTTools::SED::NormalFormGame::calculate_fitness,
           "calculates the fitness of an individual of a given strategy given a population state."
           "It always assumes that the population state does not contain the current individual",
           py::arg("player_strategy"),
           py::arg("pop_size"), py::arg("population_state"))
      .def("to_string", &EGTTools::SED::NormalFormGame::toString)
      .def("type", &EGTTools::SED::NormalFormGame::type)
      .def("payoffs", &EGTTools::SED::NormalFormGame::payoffs)
      .def("payoff", &EGTTools::SED::NormalFormGame::payoff,
           "returns the payoff of a strategy given a strategy pair.", py::arg("strategy"),
           py::arg("strategy pair"))
      .def("expected_payoffs", &EGTTools::SED::NormalFormGame::expected_payoffs, "returns the expected payoffs of each strategy vs another")
      .def_property_readonly("nb_strategies", &EGTTools::SED::NormalFormGame::nb_strategies)
      .def_property_readonly("nb_rounds", &EGTTools::SED::NormalFormGame::nb_rounds)
      .def_property_readonly("nb_states", &EGTTools::SED::NormalFormGame::nb_states)
      .def("save_payoffs", &EGTTools::SED::NormalFormGame::save_payoffs);

  // Now we define a submodule
  auto mCRD = m.def_submodule("CRD");

  py::class_<EGTTools::SED::CRD::CrdGame, EGTTools::SED::AbstractGame>(mCRD, "CRDGame")
      .def(py::init<size_t, size_t, size_t, size_t, double>(), "Collective-risk game", py::arg("endowment"),
           py::arg("threshold"), py::arg("nb_rounds"), py::arg("group_size"), py::arg("risk"))
      .def("play", &EGTTools::SED::CRD::CrdGame::play)
      .def("calculate_payoffs", &EGTTools::SED::CRD::CrdGame::calculate_payoffs,
           "updates the internal payoff matrix by calculating the payoff of each strategy "
           "given any possible group composition")
      .def("calculate_fitness", &EGTTools::SED::CRD::CrdGame::calculate_fitness,
           "calculates the fitness of an individual of a given strategy given a population state."
           "It always assumes that the population state does not contain the current individual",
           py::arg("player_strategy"),
           py::arg("pop_size"), py::arg("population_state"))
      .def("to_string", &EGTTools::SED::CRD::CrdGame::toString)
      .def("type", &EGTTools::SED::CRD::CrdGame::type)
      .def("payoffs", &EGTTools::SED::CRD::CrdGame::payoffs)
      .def("payoff", &EGTTools::SED::CRD::CrdGame::payoff,
           "returns the payoff of a strategy given a group state.", py::arg("strategy"),
           py::arg("group_composition"))
      .def("group_achievements", &EGTTools::SED::CRD::CrdGame::group_achievements)
      .def("calculate_population_group_achievement",
           &EGTTools::SED::CRD::CrdGame::calculate_population_group_achievement,
           "calculates the group achievement given a population state", py::arg("pop_size"),
           py::arg("population_state"))
      .def("calculate_group_achievement",
           &EGTTools::SED::CRD::CrdGame::calculate_group_achievement,
           "calculates the group achievement given the stationary distribution of the population",
           py::arg("pop_size"),
           py::arg("stationary_distribution"))
      .def("calculate_polarization",
           &EGTTools::SED::CRD::CrdGame::calculate_polarization,
           "calculates the polarization given the stationary distribution of the population",
           py::arg("pop_size"),
           py::arg("stationary_distribution"))
      .def("calculate_polarization_success",
           &EGTTools::SED::CRD::CrdGame::calculate_polarization_success,
           "calculates the polarization given the stationary distribution of the population and that "
           "the groups reached the target.",
           py::arg("pop_size"),
           py::arg("stationary_distribution"))
      .def("contribution_behaviors", &EGTTools::SED::CRD::CrdGame::contribution_behaviors)
      .def_property_readonly("nb_strategies", &EGTTools::SED::CRD::CrdGame::nb_strategies)
      .def_property_readonly("target", &EGTTools::SED::CRD::CrdGame::target)
      .def_property_readonly("endowment", &EGTTools::SED::CRD::CrdGame::endowment)
      .def_property_readonly("nb_rounds", &EGTTools::SED::CRD::CrdGame::nb_rounds)
      .def_property_readonly("group_size", &EGTTools::SED::CRD::CrdGame::group_size)
      .def_property_readonly("risk", &EGTTools::SED::CRD::CrdGame::risk)
      .def_property_readonly("nb_states", &EGTTools::SED::CRD::CrdGame::nb_states)
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
      .def("payoff", &EGTTools::SED::CRD::CrdGameTU::payoff,
           "returns the payoff of a strategy given a group state.", py::arg("strategy"),
           py::arg("group_composition"))
      .def("group_achievements", &EGTTools::SED::CRD::CrdGameTU::group_achievements)
      .def("calculate_population_group_achievement",
           &EGTTools::SED::CRD::CrdGameTU::calculate_population_group_achievement,
           "calculates the group achievement given a population state", py::arg("pop_size"),
           py::arg("population_state"))
      .def("calculate_group_achievement",
           &EGTTools::SED::CRD::CrdGameTU::calculate_group_achievement,
           "calculates the group achievement given the stationary distribution of the population",
           py::arg("pop_size"),
           py::arg("stationary_distribution"))
      .def("calculate_polarization",
           &EGTTools::SED::CRD::CrdGameTU::calculate_polarization,
           "calculates the polarization given the stationary distribution of the population",
           py::arg("pop_size"),
           py::arg("stationary_distribution"))
      .def("calculate_polarization_success",
           &EGTTools::SED::CRD::CrdGameTU::calculate_polarization_success,
           "calculates the polarization given the stationary distribution of the population and that "
           "the groups reached the target.",
           py::arg("pop_size"),
           py::arg("stationary_distribution"))
      .def("contribution_behaviors", &EGTTools::SED::CRD::CrdGameTU::contribution_behaviors)
      .def_property_readonly("nb_strategies", &EGTTools::SED::CRD::CrdGameTU::nb_strategies)
      .def_property_readonly("nb_strategies", &EGTTools::SED::CRD::CrdGameTU::nb_strategies)
      .def_property_readonly("target", &EGTTools::SED::CRD::CrdGameTU::target)
      .def_property_readonly("endowment", &EGTTools::SED::CRD::CrdGameTU::endowment)
      .def_property_readonly("nb_rounds", &EGTTools::SED::CRD::CrdGameTU::min_rounds)
      .def_property_readonly("group_size", &EGTTools::SED::CRD::CrdGameTU::group_size)
      .def_property_readonly("risk", &EGTTools::SED::CRD::CrdGameTU::risk)
      .def_property_readonly("nb_states", &EGTTools::SED::CRD::CrdGameTU::nb_states)
      .def("save_payoffs", &EGTTools::SED::CRD::CrdGameTU::save_payoffs);

  py::class_<PairwiseComparison>(m, "PairwiseMoran")
      .def(py::init<size_t, EGTTools::SED::AbstractGame &, size_t>(),
           "Runs a moran process with pairwise comparison and calculates fitness according to game",
           py::arg("pop_size"), py::arg("game"), py::arg("cache_size"), py::keep_alive<1, 3>())
      .def("evolve",
           static_cast<EGTTools::VectorXui (PairwiseComparison::*)(size_t, double, double,
                                                                   const Eigen::Ref<const EGTTools::VectorXui> &)>( &PairwiseComparison::evolve ),
           py::keep_alive<1, 5>(),
           "evolves the strategies for a maximum of nb_generations", py::arg("nb_generations"), py::arg("beta"),
           py::arg("mu"), py::arg("init_state"))
      .def("run",
           static_cast<EGTTools::MatrixXui2D (PairwiseComparison::*)(size_t, double, double,
                                                                     const Eigen::Ref<const EGTTools::VectorXui> &)>(&PairwiseComparison::run),
           "runs the moran process with social imitation and returns a matrix with all the states the system went through",
           py::arg("nb_generations"),
           py::arg("beta"),
           py::arg("mu"),
           py::arg("init_state"))
      .def("run",
           static_cast<EGTTools::MatrixXui2D (PairwiseComparison::*)(size_t, double,
                                                                     const Eigen::Ref<const EGTTools::VectorXui> &)>(&PairwiseComparison::run),
           "runs the moran process with social imitation and returns a matrix with all the states the system went through",
           py::arg("nb_generations"),
           py::arg("beta"),
           py::arg("init_state"))
      .def("fixation_probability", &PairwiseComparison::fixationProbability,
           "Estimates the fixation probability of an strategy in the population.",
           py::arg("mutant"), py::arg("resident"), py::arg("nb_runs"), py::arg("nb_generations"), py::arg("beta"))
      .def("stationary_distribution", &PairwiseComparison::stationaryDistribution,
           py::call_guard<py::gil_scoped_release>(),
           "Estimates the stationary distribution of the population of strategies given the game.",
           py::arg("nb_runs"), py::arg("nb_generations"), py::arg("transitory"), py::arg("beta"), py::arg("mu"))
      .def_property_readonly("nb_strategies", &PairwiseComparison::nb_strategies)
      .def_property_readonly("payoffs", &PairwiseComparison::payoffs)
      .def_property("pop_size", &PairwiseComparison::population_size, &PairwiseComparison::set_population_size)
      .def_property("cache_size", &PairwiseComparison::cache_size, &PairwiseComparison::set_cache_size);

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

  py::list strategies_CRD;
  strategies_CRD.append("FAIR");
  strategies_CRD.append("FREE-RIDER");
  strategies_CRD.append("ALTRUIST");
  strategies_CRD.append("RECIPROCAL");
  strategies_CRD.append("COMPENSATOR");
  mCRD.attr("strategies") = strategies_CRD;

  // Bind the enum class so that it's clear from python which are the indexes of each strategy
  py::enum_<EGTTools::SED::CRD::CRDBehaviors>(mCRD, "CRDBehaviors", py::arithmetic(), "Indexes of CRD behaviors")
      .value("Cooperator", EGTTools::SED::CRD::CRDBehaviors::cooperator)
      .value("Defector", EGTTools::SED::CRD::CRDBehaviors::defector)
      .value("Altruist", EGTTools::SED::CRD::CRDBehaviors::altruist)
      .value("Reciprocal", EGTTools::SED::CRD::CRDBehaviors::reciprocal)
      .value("Compensator", EGTTools::SED::CRD::CRDBehaviors::compensator);

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
      .def_property_readonly("type", &RL::Agent::type)
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

  py::class_<RL::BatchQLearningForgetAgent, RL::Agent>(mRL, "BatchQLearningForgetAgent")
      .def(py::init<size_t, size_t, size_t, double, double, double, double>(),
           "Implementation of the Batch Q-Learning algorithm with forgetting factor.",
           py::arg("nb_states"), py::arg("nb_actions"),
           py::arg("episode_length"), py::arg("endowment"), py::arg("alpha"), py::arg("lambda"), py::arg("temperature"))
      .def_property("alpha", &RL::BatchQLearningForgetAgent::alpha, &RL::BatchQLearningForgetAgent::setAlpha)
      .def_property("lambda", &RL::BatchQLearningForgetAgent::lambda, &RL::BatchQLearningForgetAgent::setLambda)
      .def_property("temperature", &RL::BatchQLearningForgetAgent::temperature,
                    &RL::BatchQLearningForgetAgent::setTemperature)
      .def("update_policy", &RL::BatchQLearningForgetAgent::inferPolicy)
      .def("reset_trajectory", &RL::BatchQLearningForgetAgent::resetTrajectory)
      .def("reinforce",
           static_cast<void (RL::BatchQLearningForgetAgent::*)()>(&RL::BatchQLearningForgetAgent::reinforceTrajectory),
           "reinforces the actions from the current trajectory, based on the agent's reward")
      .def("reinforce",
           static_cast<void (RL::BatchQLearningForgetAgent::*)(size_t)>(&RL::BatchQLearningForgetAgent::reinforceTrajectory),
           "reinforces the actions from the current trajectory, based on the agent's reward",
           py::arg("episode_length"))
      .def("act",
           static_cast<size_t (RL::BatchQLearningForgetAgent::*)(size_t)>(&RL::BatchQLearningForgetAgent::selectAction),
           "samples an action from the agent's policy", py::arg("round"))
      .def("act", static_cast<size_t (RL::BatchQLearningForgetAgent::*)(size_t,
                                                                        size_t)>(&RL::BatchQLearningForgetAgent::selectAction),
           "samples an action from the agent's policy", py::arg("round"), py::arg("state"))
      .def("reset_q_values", &RL::BatchQLearningForgetAgent::resetQValues);

  py::class_<RL::QLearningAgent, RL::Agent>(mRL, "QLearningAgent")
      .def(py::init<size_t, size_t, size_t, double, double, double, double>(),
           "Implementation of the Q-Learning algorithm.",
           py::arg("nb_states"), py::arg("nb_actions"),
           py::arg("episode_length"), py::arg("endowment"),
           py::arg("alpha"), py::arg("lambda"), py::arg("temperature"))
      .def_property("alpha", &RL::QLearningAgent::alpha, &RL::QLearningAgent::setAlpha)
      .def_property("lda", &RL::QLearningAgent::lambda, &RL::QLearningAgent::setLambda)
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

  py::class_<RL::SARSAAgent, RL::Agent>(mRL, "SARSA")
      .def(py::init<size_t, size_t, size_t, double, double, double, double>(),
           "Implementation of the Q-Learning algorithm.",
           py::arg("nb_states"), py::arg("nb_actions"),
           py::arg("episode_length"), py::arg("endowment"),
           py::arg("alpha"), py::arg("lambda"), py::arg("temperature"))
      .def_property("alpha", &RL::SARSAAgent::alpha, &RL::SARSAAgent::setAlpha)
      .def_property("lda", &RL::SARSAAgent::lambda, &RL::SARSAAgent::setLambda)
      .def_property("temperature", &RL::SARSAAgent::temperature,
                    &RL::SARSAAgent::setTemperature)
      .def("update_policy", &RL::SARSAAgent::inferPolicy)
      .def("reset_trajectory", &RL::SARSAAgent::resetTrajectory)
      .def("reinforce",
           static_cast<void (RL::SARSAAgent::*)()>(&RL::SARSAAgent::reinforceTrajectory),
           "reinforces the actions from the current trajectory, based on the agent's reward")
      .def("reinforce",
           static_cast<void (RL::SARSAAgent::*)(size_t)>(&RL::SARSAAgent::reinforceTrajectory),
           "reinforces the actions from the current trajectory, based on the agent's reward",
           py::arg("episode_length"))
      .def("act",
           static_cast<size_t (RL::SARSAAgent::*)(size_t)>(&RL::SARSAAgent::selectAction),
           "samples an action from the agent's policy", py::arg("round"))
      .def("act", static_cast<size_t (RL::SARSAAgent::*)(size_t,
                                                         size_t)>(&RL::SARSAAgent::selectAction),
           "samples an action from the agent's policy", py::arg("round"), py::arg("state"))
      .def("reset_q_values", &RL::SARSAAgent::resetQValues);

  py::class_<RL::DiscountedQLearning, RL::Agent>(mRL, "DiscountedQLearning")
      .def(py::init<size_t, size_t, size_t, double, double, double, double>(),
           "Implementation of the Q-Learning algorithm.",
           py::arg("nb_states"), py::arg("nb_actions"),
           py::arg("episode_length"), py::arg("endowment"),
           py::arg("alpha"), py::arg("lambda"), py::arg("temperature"))
      .def_property("alpha", &RL::DiscountedQLearning::alpha, &RL::DiscountedQLearning::setAlpha)
      .def_property("lda", &RL::DiscountedQLearning::lambda, &RL::DiscountedQLearning::setLambda)
      .def_property("temperature", &RL::DiscountedQLearning::temperature,
                    &RL::DiscountedQLearning::setTemperature)
      .def("update_policy", &RL::DiscountedQLearning::inferPolicy)
      .def("reset_trajectory", &RL::DiscountedQLearning::resetTrajectory)
      .def("reinforce",
           static_cast<void (RL::DiscountedQLearning::*)()>(&RL::DiscountedQLearning::reinforceTrajectory),
           "reinforces the actions from the current trajectory, based on the agent's reward")
      .def("reinforce",
           static_cast<void (RL::DiscountedQLearning::*)(size_t)>(&RL::DiscountedQLearning::reinforceTrajectory),
           "reinforces the actions from the current trajectory, based on the agent's reward",
           py::arg("episode_length"))
      .def("act",
           static_cast<size_t (RL::DiscountedQLearning::*)(size_t)>(&RL::DiscountedQLearning::selectAction),
           "samples an action from the agent's policy", py::arg("round"))
      .def("act", static_cast<size_t (RL::DiscountedQLearning::*)(size_t,
                                                                  size_t)>(&RL::DiscountedQLearning::selectAction),
           "samples an action from the agent's policy", py::arg("round"), py::arg("state"))
      .def("reset_q_values", &RL::DiscountedQLearning::resetQValues);

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
      .def(py::init<size_t, size_t, size_t, double, double, double, double>(),
           "Implementation of the Histeric Q-Learning algorithm.",
           py::arg("nb_states"), py::arg("nb_actions"),
           py::arg("episode_length"), py::arg("endowment"),
           py::arg("lambda"), py::arg("epsilon"), py::arg("temperature"))
      .def_property("lambda", &RL::RothErevAgent::lambda, &RL::RothErevAgent::setLambda)
      .def_property("epsilon", &RL::RothErevAgent::epsilon, &RL::RothErevAgent::setEpsilon)
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

  py::class_<RL::CRDGame<RL::Agent, void, void>>(mRL, "CRDGame")
      .def(py::init<>(), "Collective-risk dilemma without uncertainty")
      .def("play",
           static_cast<std::pair<double, size_t> (RL::CRDGame<RL::Agent, void, void>::*)(std::vector<RL::Agent> &,
                                                                                         EGTTools::RL::ActionSpace &,
                                                                                         size_t)>(&RL::CRDGame<RL::Agent,
                                                                                                               void,
                                                                                                               void>::playGame),
           "plays the game for a fixed number of rounds",
           py::arg("players"),
           py::arg("actions"),
           py::arg("rounds"))
      .def("reinforce_players", static_cast<bool (RL::CRDGame<RL::Agent, void, void>::*)(
               std::vector<RL::Agent> &)>(&RL::CRDGame<RL::Agent, void, void>::reinforcePath),
           "reinforces the actions of all agents in the group", py::arg("players"))
      .def("update_strategies", static_cast<bool (RL::CRDGame<RL::Agent, void, void>::*)(
               std::vector<RL::Agent> &)>(&RL::CRDGame<RL::Agent, void, void>::calcProbabilities),
           "updates the payoffs of the players in the group", py::arg("players"))
      .def("reset_episode", static_cast<bool (RL::CRDGame<RL::Agent, void, void>::*)(
               std::vector<RL::Agent> &)>(&RL::CRDGame<RL::Agent, void, void>::resetEpisode),
           "resets the trajectories of the players", py::arg("players"))
      .def("group_payoff", static_cast<double (RL::CRDGame<RL::Agent, void, void>::*)(
               std::vector<RL::Agent> &)>(&RL::CRDGame<RL::Agent, void, void>::playersPayoff),
           "returns the total payoff of the group", py::arg("players"))
      .def("group_contributions", static_cast<double (RL::CRDGame<RL::Agent, void, void>::*)(
               std::vector<RL::Agent> &)>(&RL::CRDGame<RL::Agent, void, void>::playersContribution),
           "returns the total contribution of the group", py::arg("players"))
      .def("set_payoffs", static_cast<void (RL::CRDGame<RL::Agent, void, void>::*)(
               std::vector<RL::Agent> &, unsigned int)>(&RL::CRDGame<RL::Agent, void, void>::setPayoffs),
           "sets players payoffs", py::arg("players"), py::arg("payoff"));

  py::class_<RL::CRDGame<RL::PopContainer, void, void>>(mRL, "CRDGameGeneric")
      .def(py::init<>(), "Collective-risk dilemma without uncertainty")
      .def("play",
           static_cast<std::pair<double, size_t> (RL::CRDGame<RL::PopContainer, void, void>::*)(RL::PopContainer &,
                                                                                                EGTTools::RL::ActionSpace &,
                                                                                                size_t)>(&RL::CRDGame<RL::PopContainer,
                                                                                                                      void,
                                                                                                                      void>::playGame),
           "plays the game for a fixed number of rounds",
           py::arg("players"),
           py::arg("actions"),
           py::arg("rounds"))
      .def("reinforce_players", static_cast<bool (RL::CRDGame<RL::PopContainer, void, void>::*)(
               RL::PopContainer &)>(&RL::CRDGame<RL::PopContainer, void, void>::reinforcePath),
           "reinforces the actions of all agents in the group", py::arg("players"))
      .def("update_strategies", static_cast<bool (RL::CRDGame<RL::PopContainer, void, void>::*)(
               RL::PopContainer &)>(&RL::CRDGame<RL::PopContainer, void, void>::calcProbabilities),
           "updates the payoffs of the players in the group", py::arg("players"))
      .def("reset_episode", static_cast<bool (RL::CRDGame<RL::PopContainer, void, void>::*)(
               RL::PopContainer &)>(&RL::CRDGame<RL::PopContainer, void, void>::resetEpisode),
           "resets the trajectories of the players", py::arg("players"))
      .def("group_payoff", static_cast<double (RL::CRDGame<RL::PopContainer, void, void>::*)(
               RL::PopContainer &)>(&RL::CRDGame<RL::PopContainer, void, void>::playersPayoff),
           "returns the total payoff of the group", py::arg("players"))
      .def("group_contributions", static_cast<double (RL::CRDGame<RL::PopContainer, void, void>::*)(
               RL::PopContainer &)>(&RL::CRDGame<RL::PopContainer, void, void>::playersContribution),
           "returns the total contribution of the group", py::arg("players"))
      .def("set_payoffs", static_cast<void (RL::CRDGame<RL::PopContainer, void, void>::*)(
               RL::PopContainer &, double)>(&RL::CRDGame<RL::PopContainer, void, void>::setPayoffs),
           "sets players payoffs", py::arg("players"), py::arg("payoff"))
      .def("update_payoffs", static_cast<void (RL::CRDGame<RL::PopContainer, void, void>::*)(
               RL::PopContainer &, double)>(&RL::CRDGame<RL::PopContainer, void, void>::setPayoffs),
           "sets players payoffs", py::arg("players"), py::arg("multiplier"));

  py::class_<RL::PopContainer>(mRL, "Population")
      .def(py::init<const std::string &, size_t, size_t, size_t, size_t, double, std::vector<double>>(),
           "Population container. This class allows for the initialization of a vector of players (that can be from"
           "different child classes) that can be passed to a game class.",
           py::arg("agent_type"),
           py::arg("nb_agents"),
           py::arg("nb_states"),
           py::arg("nb_actions"),
           py::arg("episode_length"),
           py::arg("endowment"),
           py::arg("*args"),
           py::return_value_policy::reference_internal)
      .def("size", &RL::PopContainer::size, "returns the size of the contained vector")
      .def("reset", &RL::PopContainer::reset, "reset all individuals in the population")
      .def("__getitem__", [](RL::PopContainer &s, size_t i) -> RL::Agent & {
        if (i >= s.size()) throw py::index_error();
        return s[i];
      }, py::return_value_policy::reference_internal)
      .def("__len__", [](const RL::PopContainer &v) { return v.size(); })
      .def("__repr__", &RL::PopContainer::toString);

  py::class_<crdData>(mRL, "crdData")
      .def(py::init<size_t, EGTTools::RL::PopContainer &>(), py::keep_alive<1, 3>(), "CRD Data Container")
      .def_readwrite("group_achievement", &crdData::eta, py::return_value_policy::reference_internal)
      .def_readwrite("avg_contributions", &crdData::avg_contribution, py::return_value_policy::reference_internal)
      .def_readwrite("population", &crdData::population, py::return_value_policy::reference_internal);

  py::class_<crdDataIslands>(mRL, "crdDataIslands")
      .def(py::init<Vector &, Vector &, std::vector<EGTTools::RL::PopContainer> &>(), "CRD Data Container")
      .def_readwrite("group_achievement", &crdDataIslands::eta, py::return_value_policy::reference_internal)
      .def_readwrite("avg_contributions", &crdDataIslands::avg_contribution,
                     py::return_value_policy::reference_internal)
      .def_readwrite("groups", &crdDataIslands::groups, py::return_value_policy::reference_internal);

  py::class_<DataTableCRD>(mRL, "DataTableCRD")
      .def(py::init<size_t,
                    size_t,
                    std::vector<std::string> &,
                    std::vector<std::string> &,
                    std::vector<EGTTools::RL::PopContainer> &>(),
           "CRD Data container with Table format",
           py::arg("nb_rows"),
           py::arg("nb_columns"),
           py::arg("headers"),
           py::arg("column_types"),
           py::arg("populations"))
      .def_readwrite("data", &DataTableCRD::data, py::return_value_policy::reference_internal)
      .def_readwrite("headers", &DataTableCRD::header,
                     py::return_value_policy::reference_internal)
      .def_readwrite("column_types", &DataTableCRD::column_types,
                     py::return_value_policy::reference_internal)
      .def_readwrite("populations", &DataTableCRD::populations, py::return_value_policy::reference_internal);

  py::class_<EGTTools::RL::FlattenState>(mRL, "FlattenState")
      .def(py::init<const EGTTools::RL::Factors &>(), py::keep_alive<1, 2>(),
           "Converts back and forth multi-dimensional states to flattened states")
      .def("to_factors",
           static_cast<EGTTools::RL::Factors (RL::FlattenState::*)(size_t)>( &RL::FlattenState::toFactors),
           "converts an state index into a tuple (vector)", py::arg("index"))
      .def("to_index", &RL::FlattenState::toIndex,
           "converts a vector (tuple) multi-dimensional state into a scalar index.", py::arg("state"))
      .def_readwrite("space", &EGTTools::RL::FlattenState::space,
                     "List with as many elements as dimensions of "
                     "the sate-space. Each element indicates the size "
                     "of the dimension.")
      .def_readwrite("factored_space_size", &EGTTools::RL::FlattenState::factor_space,
                     "size of the factored/flattened state-space.")
      .def("__repr__",
           [](const EGTTools::RL::FlattenState &a) {
             return "<EGTtools.RL.FlattenState size '" + std::to_string(a.factor_space) + "'>";
           });

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
      .def("run",
           static_cast<RL::DataTypes::CRDData (RL::CRDSim::*)(size_t,
                                                              size_t,
                                                              size_t,
                                                              double,
                                                              double,
                                                              const std::string &,
                                                              const std::vector<double> &)>(&RL::CRDSim::run),
           "Runs a CRD simulation with only one group and returns CRDData",
           py::arg("group_size"),
           py::arg("nb_generations"),
           py::arg("nb_games"),
           py::arg("target"),
           py::arg("risk"),
           py::arg("agent_type"),
           py::arg("*args"))
      .def("run", static_cast<EGTTools::Matrix2D (RL::CRDSim::*)(size_t, size_t, size_t, double,
                                                                 const std::vector<double> &)>(&RL::CRDSim::run),
           py::call_guard<py::gil_scoped_release>(),
           "Runs a CRD simulation for nb_groups",
           py::arg("nb_episodes"), py::arg("nb_games"), py::arg("nb_groups"), py::arg("risk"),
           py::arg("*agent_args"))
      .def("run", static_cast<crdDataIslands (RL::CRDSim::*)(size_t, size_t, size_t, size_t, double,
                                                             size_t, const std::string &agent_type,
                                                             const std::vector<double> &)>(&RL::CRDSim::run),
           py::call_guard<py::gil_scoped_release>(),
           "Runs a CRD simulation for nb_groups and returns the population in a data container",
           py::arg("nb_groups"), py::arg("group_size"), py::arg("nb_episodes"), py::arg("nb_games"),
           py::arg("risk"),
           py::arg("transient"), py::arg("agent_type"), py::arg("*agent_args"))
      .def("runWellMixed", static_cast<EGTTools::Matrix2D (RL::CRDSim::*)(size_t, size_t,
                                                                          size_t, size_t, double, double,
                                                                          const std::vector<double> &)>(&RL::CRDSim::runWellMixed),
           "Runs a simulation with a well mixed population",
           py::arg("nb_generations"), py::arg("nb_games"), py::arg("nb_groups"), py::arg("group_size"),
           py::arg("threshold"),
           py::arg("risk"),
           py::arg("*agent_args"))
      .def("runWellMixed", static_cast<EGTTools::Matrix2D (RL::CRDSim::*)(size_t,
                                                                          size_t,
                                                                          size_t,
                                                                          size_t,
                                                                          size_t,
                                                                          double,
                                                                          double,
                                                                          size_t,
                                                                          const std::string &,
                                                                          const std::vector<double> &)>(&RL::CRDSim::runWellMixed),
           py::call_guard<py::gil_scoped_release>(),
           "Runs a simulation with a well mixed population",
           py::arg("nb_runs"), py::arg("pop_size"), py::arg("group_size"), py::arg("nb_generations"),
           py::arg("nb_games"), py::arg("threshold"), py::arg("risk"), py::arg("transient"),
           py::arg("agent_type"),
           py::arg("*agent_args"))
      .def("runWellMixed", static_cast<crdData (RL::CRDSim::*)(size_t, size_t,
                                                               size_t, size_t, double, double,
                                                               const std::string &,
                                                               const std::vector<double> &)>(&RL::CRDSim::runWellMixed),
           "Runs a simulation with a well mixed population and returns the groups success and donations"
           "during learning, as well as the final population.",
           py::arg("pop_size"), py::arg("group_size"),
           py::arg("nb_generations"), py::arg("nb_games"), py::arg("threshold"), py::arg("risk"),
           py::arg("agent_type"),
           py::arg("*agent_args"))
      .def("runWellMixedSync", static_cast<crdData (RL::CRDSim::*)(size_t, size_t,
                                                                   size_t, size_t, double, double,
                                                                   const std::string &,
                                                                   const std::vector<double> &)>(&RL::CRDSim::runWellMixedSync),
           "Runs a simulation with a well mixed population where the updated are synchronous"
           "and all players play the game once and returns the groups success and donations"
           "during learning, as well as the final population.",
           py::arg("pop_size"), py::arg("group_size"),
           py::arg("nb_generations"), py::arg("nb_games"), py::arg("threshold"), py::arg("risk"),
           py::arg("agent_type"),
           py::arg("*agent_args"))
      .def("runWellMixedSync", static_cast<EGTTools::Matrix2D (RL::CRDSim::*)(size_t, size_t,
                                                                              size_t, size_t, size_t, double,
                                                                              double, size_t,
                                                                              const std::string &,
                                                                              const std::vector<double> &)>(&RL::CRDSim::runWellMixedSync),
           py::call_guard<py::gil_scoped_release>(),
           "Runs a simulation with a well mixed population with synchronous updates and each agent plays only"
           "once per generation.",
           py::arg("nb_runs"), py::arg("pop_size"), py::arg("group_size"), py::arg("nb_generations"),
           py::arg("nb_games"),
           py::arg("threshold"), py::arg("risk"), py::arg("transient"),
           py::arg("agent_type"),
           py::arg("*agent_args"))
      .def("runTimingUncertainty", &RL::CRDSim::runTimingUncertainty,
           "Runs CRD simulations with unconditional agents and timing uncertainty.",
           py::arg("nb_episodes"), py::arg("nb_games"),
           py::arg("min_rounds"), py::arg("mean_rounds"), py::arg("max_rounds"),
           py::arg("p"), py::arg("risk"), py::arg("*agent_args"), py::arg("crd_type"))
      .def("runWellMixedTU",
           static_cast<crdData (RL::CRDSim::*)(size_t, size_t, size_t, size_t, double, double, size_t,
                                               size_t, size_t, double, const std::string &,
                                               const std::vector<double> &)>(&RL::CRDSim::runWellMixedTU),
           "Runs CRD simulations with unconditional agents and timing uncertainty in a well-mixed population.",
           py::arg("pop_size"), py::arg("group_size"),
           py::arg("nb_generations"), py::arg("nb_games"), py::arg("threshold"),
           py::arg("risk"),
           py::arg("min_rounds"), py::arg("mean_rounds"),
           py::arg("max_rounds"), py::arg("p"),
           py::arg("agent_type"), py::arg("*agent_args"))
      .def("runWellMixedTU",
           static_cast<EGTTools::Matrix2D (RL::CRDSim::*)(size_t,
                                                          size_t,
                                                          size_t,
                                                          size_t,
                                                          size_t,
                                                          double,
                                                          double,
                                                          size_t,
                                                          size_t,
                                                          size_t,
                                                          size_t,
                                                          double,
                                                          const std::string &,
                                                          const std::vector<double> &)>(&RL::CRDSim::runWellMixedTU),
           "Runs CRD simulations with unconditional agents and timing uncertainty in a well-mixed population.",
           py::arg("nb_runs"), py::arg("pop_size"), py::arg("group_size"),
           py::arg("nb_generations"), py::arg("nb_games"),
           py::arg("threshold"), py::arg("risk"),
           py::arg("transient"),
           py::arg("min_rounds"), py::arg("mean_rounds"),
           py::arg("max_rounds"), py::arg("p"),
           py::arg("agent_type"), py::arg("*agent_args"))
      .def("runWellMixedTUSync",
           static_cast<crdData (RL::CRDSim::*)(size_t, size_t, size_t, size_t, double, double, size_t,
                                               size_t, size_t, double, const std::string &,
                                               const std::vector<double> &)>(&RL::CRDSim::runWellMixedTUSync),
           "Runs CRD simulations with unconditional agents and timing uncertainty in a well-mixed population"
           "with synchronous updates and where each player plays only once per generation.",
           py::arg("pop_size"), py::arg("group_size"),
           py::arg("nb_generations"), py::arg("nb_games"), py::arg("threshold"),
           py::arg("risk"),
           py::arg("min_rounds"), py::arg("mean_rounds"),
           py::arg("max_rounds"), py::arg("p"),
           py::arg("agent_type"), py::arg("*agent_args"))
      .def("runWellMixedTUSync",
           static_cast<EGTTools::Matrix2D (RL::CRDSim::*)(size_t,
                                                          size_t,
                                                          size_t,
                                                          size_t,
                                                          size_t,
                                                          double,
                                                          double,
                                                          size_t,
                                                          size_t,
                                                          size_t,
                                                          size_t,
                                                          double,
                                                          const std::string &,
                                                          const std::vector<double> &)>(&RL::CRDSim::runWellMixedTUSync),
           "Runs CRD simulations with unconditional agents and timing uncertainty in a well-mixed population with"
           "synchronous updates and each agent plays only once per generation.",
           py::arg("nb_runs"),
           py::arg("pop_size"),
           py::arg("group_size"),
           py::arg("nb_generations"),
           py::arg("nb_games"),
           py::arg("threshold"),
           py::arg("risk"),
           py::arg("transient"),
           py::arg("min_rounds"),
           py::arg("mean_rounds"),
           py::arg("max_rounds"),
           py::arg("p"),
           py::arg("agent_type"),
           py::arg("*agent_args"))
      .def("runWellMixedThU", static_cast<crdData (RL::CRDSim::*)(size_t, size_t,
                                                                  size_t, size_t, size_t, size_t,
                                                                  double, const std::string &,
                                                                  const std::vector<double> &)>(&RL::CRDSim::runWellMixedThU),
           "Runs a simulation with a well mixed population with threshold uncertainty"
           " returns the groups success and donations"
           "during learning, as well as the final population.",
           py::arg("pop_size"), py::arg("group_size"),
           py::arg("nb_generations"), py::arg("nb_games"), py::arg("threshold"), py::arg("delta"),
           py::arg("risk"),
           py::arg("agent_type"),
           py::arg("*agent_args"))
      .def("runWellMixedThU", static_cast<Matrix2D (RL::CRDSim::*)(size_t, size_t, size_t,
                                                                   size_t, size_t, size_t,
                                                                   size_t,
                                                                   double, size_t,
                                                                   const std::string &,
                                                                   const std::vector<double> &)>(&RL::CRDSim::runWellMixedThU),
           "Runs a simulation with a well mixed population with threshold uncertainty"
           " returns the groups success and donations"
           "during learning, as well as the final population.",
           py::arg("nb_runs"), py::arg("pop_size"), py::arg("group_size"),
           py::arg("nb_generations"), py::arg("nb_games"), py::arg("threshold"), py::arg("delta"),
           py::arg("risk"), py::arg("transient"),
           py::arg("agent_type"),
           py::arg("*agent_args"))
      .def("runWellMixedThUSync", static_cast<crdData (RL::CRDSim::*)(size_t, size_t,
                                                                      size_t, size_t, size_t, size_t,
                                                                      double, const std::string &,
                                                                      const std::vector<double> &)>(&RL::CRDSim::runWellMixedThUSync),
           "Runs a simulation with a well mixed population with threshold uncertainty"
           " returns the groups success and donations"
           "during learning, as well as the final population.",
           py::arg("pop_size"), py::arg("group_size"),
           py::arg("nb_generations"), py::arg("nb_games"), py::arg("threshold"), py::arg("delta"),
           py::arg("risk"),
           py::arg("agent_type"),
           py::arg("*agent_args"))
      .def("runWellMixedThUSync", static_cast<Matrix2D (RL::CRDSim::*)(size_t, size_t, size_t,
                                                                       size_t, size_t, size_t,
                                                                       size_t,
                                                                       double, size_t,
                                                                       const std::string &,
                                                                       const std::vector<double> &)>(&RL::CRDSim::runWellMixedThUSync),
           "Runs a simulation with a well mixed population with threshold uncertainty"
           " returns the groups success and donations"
           "during learning, as well as the final population.",
           py::arg("nb_runs"), py::arg("pop_size"), py::arg("group_size"),
           py::arg("nb_generations"), py::arg("nb_games"), py::arg("threshold"), py::arg("delta"),
           py::arg("risk"), py::arg("transient"),
           py::arg("agent_type"),
           py::arg("*agent_args"))
      .def("runWellMixedTUnThU",
           static_cast<crdData (RL::CRDSim::*)(size_t, size_t, size_t, size_t, size_t, size_t, double, size_t,
                                               size_t, size_t, double, const std::string &,
                                               const std::vector<double> &)>(&RL::CRDSim::runWellMixedTUnThU),
           "Runs CRD simulations with unconditional agents and timing uncertainty and threshold uncertainty"
           " in a well-mixed population.",
           py::arg("pop_size"), py::arg("group_size"),
           py::arg("nb_generations"), py::arg("nb_games"), py::arg("threshold"), py::arg("delta"),
           py::arg("risk"),
           py::arg("min_rounds"), py::arg("mean_rounds"),
           py::arg("max_rounds"), py::arg("p"),
           py::arg("agent_type"), py::arg("*agent_args"))
      .def("runWellMixedTUnThU",
           static_cast<EGTTools::Matrix2D (RL::CRDSim::*)(size_t, size_t, size_t, size_t, size_t, size_t,
                                                          size_t, double, size_t, size_t,
                                                          size_t, size_t, double, const std::string &,
                                                          const std::vector<double> &)>(&RL::CRDSim::runWellMixedTUnThU),
           "Runs CRD simulations with unconditional agents and timing uncertainty and threshold uncertainty"
           " in a well-mixed population.",
           py::arg("nb_runs"), py::arg("pop_size"), py::arg("group_size"),
           py::arg("nb_generations"), py::arg("nb_games"), py::arg("threshold"), py::arg("delta"),
           py::arg("risk"), py::arg("transient"), py::arg("min_rounds"), py::arg("mean_rounds"),
           py::arg("max_rounds"), py::arg("p"),
           py::arg("agent_type"), py::arg("*agent_args"))
      .def("runConditionalTU", &RL::CRDSim::runConditionalTU,
           "Runs CRD simulations with conditional agents and timing uncertainty.",
           py::arg("nb_episodes"), py::arg("nb_games"),
           py::arg("min_rounds"), py::arg("mean_rounds"), py::arg("max_rounds"),
           py::arg("p"), py::arg("risk"), py::arg("*agent_args"), py::arg("crd_type"))
      .def("runConditional",
           static_cast<EGTTools::Matrix2D (RL::CRDSim::*)(size_t, size_t, const std::vector<double> &,
                                                          const std::string &)>(&RL::CRDSim::runConditional),
           "Runs a CRD simulation for a single group with conditional agents", py::arg("nb_episodes"),
           py::arg("nb_games"),
           py::arg("*agent_args"), py::arg("crd_type"))
      .def("runConditional",
           static_cast<crdData (RL::CRDSim::*)(size_t, size_t, size_t, double, const std::string &,
                                               const std::vector<double> &)>(&RL::CRDSim::runConditional),
           "Runs a CRD simulation for a single group with conditional agents that returns a population",
           py::arg("group_size"),
           py::arg("nb_episodes"), py::arg("nb_games"), py::arg("risk"), py::arg("agent_type"),
           py::arg("*agent_args"))
      .def("runConditional",
           static_cast<EGTTools::Matrix2D (RL::CRDSim::*)(size_t, size_t, size_t, double,
                                                          const std::vector<double> &,
                                                          const std::string &)>(&RL::CRDSim::runConditional),
           "Runs a CRD simulation for nb_groups with conditional agents", py::arg("nb_episodes"),
           py::arg("nb_games"),
           py::arg("nb_groups"), py::arg("risk"),
           py::arg("*agent_args"), py::arg("crd_type"))
      .def("runConditionalWellMixed",
           static_cast<crdData (RL::CRDSim::*)(size_t, size_t, size_t, size_t, double, double,
                                               const std::string &,
                                               const std::vector<double> &)>(&RL::CRDSim::runConditionalWellMixed),
           "Runs CRD simulations with unconditional agents in a well-mixed population.",
           py::arg("pop_size"), py::arg("group_size"),
           py::arg("nb_generations"), py::arg("nb_games"), py::arg("threshold"), py::arg("risk"),
           py::arg("agent_type"), py::arg("*agent_args"))
      .def("runConditionalWellMixed",
           static_cast<EGTTools::Matrix2D (RL::CRDSim::*)(size_t,
                                                          size_t,
                                                          size_t,
                                                          size_t,
                                                          size_t,
                                                          double,
                                                          double,
                                                          size_t,
                                                          const std::string &,
                                                          const std::vector<double> &)>(&RL::CRDSim::runConditionalWellMixed),
           "Runs CRD simulations with unconditional agents in a well-mixed population.",
           py::arg("nb_runs"), py::arg("pop_size"), py::arg("group_size"),
           py::arg("nb_generations"), py::arg("nb_games"), py::arg("threshold"), py::arg("risk"),
           py::arg("transient"),
           py::arg("agent_type"), py::arg("*agent_args"))
      .def("runConditionalWellMixedSync",
           static_cast<crdData (RL::CRDSim::*)(size_t, size_t, size_t, size_t, double, double,
                                               const std::string &,
                                               const std::vector<double> &)>(&RL::CRDSim::runConditionalWellMixedSync),
           "Runs CRD simulations with unconditional agents in a well-mixed population.",
           py::arg("pop_size"), py::arg("group_size"),
           py::arg("nb_generations"), py::arg("nb_games"), py::arg("threshold"), py::arg("risk"),
           py::arg("agent_type"), py::arg("*agent_args"))
      .def("runConditionalWellMixedSync",
           static_cast<EGTTools::Matrix2D (RL::CRDSim::*)(size_t,
                                                          size_t,
                                                          size_t,
                                                          size_t,
                                                          size_t,
                                                          double,
                                                          double,
                                                          size_t,
                                                          const std::string &,
                                                          const std::vector<double> &)>(&RL::CRDSim::runConditionalWellMixedSync),
           "Runs CRD simulations with unconditional agents in a well-mixed population.",
           py::arg("nb_runs"), py::arg("pop_size"), py::arg("group_size"),
           py::arg("nb_generations"), py::arg("nb_games"), py::arg("threshold"), py::arg("risk"),
           py::arg("transient"),
           py::arg("agent_type"), py::arg("*agent_args"))
      .def("runConditionalWellMixedTU",
           static_cast<crdData (RL::CRDSim::*)(size_t, size_t, size_t, size_t, double, double, size_t,
                                               size_t, size_t, double, const std::string &,
                                               const std::vector<double> &)>(&RL::CRDSim::runConditionalWellMixedTU),
           "Runs CRD simulations with unconditional agents and timing uncertainty in a well-mixed population.",
           py::arg("pop_size"), py::arg("group_size"),
           py::arg("nb_generations"), py::arg("nb_games"), py::arg("threshold"), py::arg("risk"),
           py::arg("min_rounds"), py::arg("mean_rounds"),
           py::arg("max_rounds"), py::arg("p"),
           py::arg("agent_type"), py::arg("*agent_args"))
      .def("runConditionalWellMixedTU",
           static_cast<EGTTools::Matrix2D (RL::CRDSim::*)(size_t,
                                                          size_t,
                                                          size_t,
                                                          size_t,
                                                          size_t,
                                                          double,
                                                          double,
                                                          size_t,
                                                          size_t,
                                                          size_t,
                                                          size_t,
                                                          double,
                                                          const std::string &,
                                                          const std::vector<double> &)>(&RL::CRDSim::runConditionalWellMixedTU),
           "Runs CRD simulations with unconditional agents and timing uncertainty in a well-mixed population.",
           py::arg("nb_runs"), py::arg("pop_size"), py::arg("group_size"),
           py::arg("nb_generations"), py::arg("nb_games"), py::arg("threshold"), py::arg("risk"),
           py::arg("transient"),
           py::arg("min_rounds"), py::arg("mean_rounds"),
           py::arg("max_rounds"), py::arg("p"),
           py::arg("agent_type"), py::arg("*agent_args"))
      .def("runConditionalWellMixedTUSync",
           static_cast<crdData (RL::CRDSim::*)(size_t, size_t, size_t, size_t, double, double, size_t,
                                               size_t, size_t, double, const std::string &,
                                               const std::vector<double> &)>(&RL::CRDSim::runConditionalWellMixedTUSync),
           "Runs CRD simulations with unconditional agents and timing uncertainty in a well-mixed population.",
           py::arg("pop_size"), py::arg("group_size"),
           py::arg("nb_generations"), py::arg("nb_games"), py::arg("threshold"), py::arg("risk"),
           py::arg("min_rounds"), py::arg("mean_rounds"),
           py::arg("max_rounds"), py::arg("p"),
           py::arg("agent_type"), py::arg("*agent_args"))
      .def("runConditionalWellMixedTUSync",
           static_cast<EGTTools::Matrix2D (RL::CRDSim::*)(size_t,
                                                          size_t,
                                                          size_t,
                                                          size_t,
                                                          size_t,
                                                          double,
                                                          double,
                                                          size_t,
                                                          size_t,
                                                          size_t,
                                                          size_t,
                                                          double,
                                                          const std::string &,
                                                          const std::vector<double> &)>(&RL::CRDSim::runConditionalWellMixedTUSync),
           "Runs CRD simulations with unconditional agents and timing uncertainty in a well-mixed population.",
           py::arg("nb_runs"), py::arg("pop_size"), py::arg("group_size"),
           py::arg("nb_generations"), py::arg("nb_games"), py::arg("threshold"), py::arg("risk"),
           py::arg("transient"),
           py::arg("min_rounds"), py::arg("mean_rounds"),
           py::arg("max_rounds"), py::arg("p"),
           py::arg("agent_type"), py::arg("*agent_args"))
      .def("runConditionalWellMixedThU", static_cast<crdData (RL::CRDSim::*)(size_t, size_t,
                                                                             size_t, size_t, size_t, size_t,
                                                                             double, const std::string &,
                                                                             const std::vector<double> &)>(&RL::CRDSim::runConditionalWellMixedThU),
           "Runs a simulation with a well mixed population with threshold uncertainty"
           " returns the groups success and donations"
           "during learning, as well as the final population.",
           py::arg("pop_size"), py::arg("group_size"),
           py::arg("nb_generations"), py::arg("nb_games"), py::arg("threshold"),
           py::arg("delta"),
           py::arg("risk"),
           py::arg("agent_type"),
           py::arg("*agent_args"))
      .def("runConditionalWellMixedThU", static_cast<Matrix2D (RL::CRDSim::*)(size_t, size_t, size_t,
                                                                              size_t, size_t, size_t,
                                                                              size_t,
                                                                              double, size_t,
                                                                              const std::string &,
                                                                              const std::vector<double> &)>(&RL::CRDSim::runConditionalWellMixedThU),
           "Runs a simulation with a well mixed population with threshold uncertainty"
           " returns the groups success and donations"
           "during learning, as well as the final population.",
           py::arg("nb_runs"), py::arg("pop_size"), py::arg("group_size"),
           py::arg("nb_generations"), py::arg("nb_games"), py::arg("threshold"),
           py::arg("delta"),
           py::arg("risk"), py::arg("transient"),
           py::arg("agent_type"),
           py::arg("*agent_args"))
      .def("runConditionalWellMixedThUSync", static_cast<crdData (RL::CRDSim::*)(size_t,
                                                                                 size_t,
                                                                                 size_t,
                                                                                 size_t,
                                                                                 size_t,
                                                                                 size_t,
                                                                                 double,
                                                                                 const std::string &,
                                                                                 const std::vector<double> &)>(&RL::CRDSim::runConditionalWellMixedThUSync),
           "Runs a simulation with a well mixed population with threshold uncertainty"
           " returns the groups success and donations"
           "during learning, as well as the final population.",
           py::arg("pop_size"), py::arg("group_size"),
           py::arg("nb_generations"), py::arg("nb_games"), py::arg("threshold"), py::arg("delta"),
           py::arg("risk"),
           py::arg("agent_type"),
           py::arg("*agent_args"))
      .def("runConditionalWellMixedThUSync", static_cast<Matrix2D (RL::CRDSim::*)(size_t, size_t, size_t,
                                                                                  size_t, size_t, size_t,
                                                                                  size_t,
                                                                                  double, size_t,
                                                                                  const std::string &,
                                                                                  const std::vector<double> &)>(&RL::CRDSim::runConditionalWellMixedThUSync),
           "Runs a simulation with a well mixed population with threshold uncertainty"
           " returns the groups success and donations"
           "during learning, as well as the final population.",
           py::arg("nb_runs"), py::arg("pop_size"), py::arg("group_size"),
           py::arg("nb_generations"), py::arg("nb_games"), py::arg("threshold"), py::arg("delta"),
           py::arg("risk"), py::arg("transient"),
           py::arg("agent_type"),
           py::arg("*agent_args"))
      .def("runConditionalWellMixedTUnThU",
           static_cast<crdData (RL::CRDSim::*)(size_t, size_t, size_t, size_t, size_t, size_t, double, size_t,
                                               size_t, size_t, double, const std::string &,
                                               const std::vector<double> &)>(&RL::CRDSim::runConditionalWellMixedTUnThU),
           "Runs CRD simulations with unconditional agents and timing uncertainty and threshold uncertainty"
           " in a well-mixed population.",
           py::arg("pop_size"), py::arg("group_size"),
           py::arg("nb_generations"), py::arg("nb_games"), py::arg("threshold"), py::arg("delta"),
           py::arg("risk"),
           py::arg("min_rounds"), py::arg("mean_rounds"),
           py::arg("max_rounds"), py::arg("p"),
           py::arg("agent_type"), py::arg("*agent_args"))
      .def("runConditionalWellMixedTUnThU",
           static_cast<EGTTools::Matrix2D (RL::CRDSim::*)(size_t, size_t, size_t, size_t, size_t, size_t,
                                                          size_t, double, size_t, size_t,
                                                          size_t, size_t, double, const std::string &,
                                                          const std::vector<double> &)>(&RL::CRDSim::runConditionalWellMixedTUnThU),
           "Runs CRD simulations with unconditional agents and timing uncertainty and threshold uncertainty"
           " in a well-mixed population.",
           py::arg("nb_runs"), py::arg("pop_size"), py::arg("group_size"),
           py::arg("nb_generations"), py::arg("nb_games"), py::arg("threshold"), py::arg("delta"),
           py::arg("risk"), py::arg("transient"), py::arg("min_rounds"), py::arg("mean_rounds"),
           py::arg("max_rounds"), py::arg("p"),
           py::arg("agent_type"), py::arg("*agent_args"))
      .def_readwrite("population", &RL::CRDSim::population, py::return_value_policy::reference_internal)
      .def("reset_population", &RL::CRDSim::resetPopulation)
      .def("setGameType", &RL::CRDSim::setGameType, "sets the game to milinski or xico versions of the CRD",
           py::arg("crd_type"))
      .def_property("nb_games", &RL::CRDSim::nb_games, &RL::CRDSim::set_nb_games)
      .def_property("nb_episodes", &RL::CRDSim::nb_episodes, &RL::CRDSim::set_nb_episodes)
      .def_property("nb_rounds", &RL::CRDSim::nb_rounds, &RL::CRDSim::set_nb_rounds)
      .def_property("nb_actions", &RL::CRDSim::nb_actions, &RL::CRDSim::set_nb_actions)
      .def_property("risk", &RL::CRDSim::risk, &RL::CRDSim::set_risk)
      .def_property("endowment", &RL::CRDSim::endowment, &RL::CRDSim::set_endowment)
      .def_property("threshold", &RL::CRDSim::threshold, &RL::CRDSim::set_threshold)
      .def_property("available_actions", &RL::CRDSim::available_actions, &RL::CRDSim::set_available_actions,
                    py::return_value_policy::reference_internal)
      .def_property("agent_type", &RL::CRDSim::agent_type, &RL::CRDSim::set_agent_type);

  // Now we define a submodule
  auto mSims = mRL.def_submodule("simulators", "contains RL simulators");
  auto mSimsCRD = mSims.def_submodule("CRD", "contains RL simulators for the CRD game");
  py::class_<RL::Simulators::CRD::CRDSimIslands>(mSimsCRD, "CRDIslandsSim")
      .def(py::init<>(),
           "Contains Collective-risk dilemma simulations with RL agents that learn in islands "
           "and are later evaluated by randomly playing the game with members of its own or other islands.")
      .def_property("verbose_level",
                    &RL::Simulators::CRD::CRDSimIslands::verbose_level,
                    &RL::Simulators::CRD::CRDSimIslands::set_verbose_level)
      .def("run_group_islands",
           static_cast<DataTableCRD (RL::Simulators::CRD::CRDSimIslands::*)(size_t,
                                                                            size_t,
                                                                            size_t,
                                                                            size_t,
                                                                            size_t,
                                                                            size_t,
                                                                            int,
                                                                            int,
                                                                            double,
                                                                            RL::ActionSpace &,
                                                                            const std::string &,
                                                                            const std::vector<double> &)>(&RL::Simulators::CRD::CRDSimIslands::run_group_islands),
           "Runs a CRD simulation in independent :param nb_groups groups, and "
           "later evaluates them by reshuffling the groups. It uses unconditional agents.",
           py::arg("nb_evaluation_games"),
           py::arg("nb_groups"),
           py::arg("group_size"),
           py::arg("nb_generations"),
           py::arg("nb_games"),
           py::arg("nb_rounds"),
           py::arg("target"),
           py::arg("endowment"),
           py::arg("risk"),
           py::arg("available_actions"),
           py::arg("agent_type"),
           py::arg("*args"))
      .def("run_conditional_group_islands",
           static_cast<DataTableCRD (RL::Simulators::CRD::CRDSimIslands::*)(size_t,
                                                                            size_t,
                                                                            size_t,
                                                                            size_t,
                                                                            size_t,
                                                                            size_t,
                                                                            int,
                                                                            int,
                                                                            double,
                                                                            RL::ActionSpace &,
                                                                            const std::string &,
                                                                            const std::vector<double> &)>(&RL::Simulators::CRD::CRDSimIslands::run_conditional_group_islands),
           "Runs a CRD simulation in independent :param nb_groups groups, and "
           "later evaluates them by reshuffling the groups. Uses conditional agents.",
           py::arg("nb_evaluation_games"),
           py::arg("nb_groups"),
           py::arg("group_size"),
           py::arg("nb_generations"),
           py::arg("nb_games"),
           py::arg("nb_rounds"),
           py::arg("target"),
           py::arg("endowment"),
           py::arg("risk"),
           py::arg("available_actions"),
           py::arg("agent_type"),
           py::arg("*args"))
      .def("run_group_islandsTU",
           static_cast<DataTableCRD (RL::Simulators::CRD::CRDSimIslands::*)(size_t,
                                                                            size_t,
                                                                            size_t,
                                                                            size_t,
                                                                            size_t,
                                                                            size_t,
                                                                            size_t,
                                                                            size_t,
                                                                            double,
                                                                            int,
                                                                            int,
                                                                            double,
                                                                            RL::ActionSpace &,
                                                                            const std::string &,
                                                                            const std::vector<double> &)>(&RL::Simulators::CRD::CRDSimIslands::run_group_islandsTU),
           "Runs a CRD simulation with timing uncertainty in independent :param nb_groups groups, and "
           "later evaluates them by reshuffling the groups. It uses unconditional agents.",
           py::arg("nb_evaluation_games"),
           py::arg("nb_groups"),
           py::arg("group_size"),
           py::arg("nb_generations"),
           py::arg("nb_games"),
           py::arg("min_rounds"),
           py::arg("mean_rounds"),
           py::arg("max_rounds"),
           py::arg("p"),
           py::arg("target"),
           py::arg("endowment"),
           py::arg("risk"),
           py::arg("available_actions"),
           py::arg("agent_type"),
           py::arg("*args"))
      .def("run_conditional_group_islandsTU",
           static_cast<DataTableCRD (RL::Simulators::CRD::CRDSimIslands::*)(size_t,
                                                                            size_t,
                                                                            size_t,
                                                                            size_t,
                                                                            size_t,
                                                                            size_t,
                                                                            size_t,
                                                                            size_t,
                                                                            double,
                                                                            int,
                                                                            int,
                                                                            double,
                                                                            RL::ActionSpace &,
                                                                            const std::string &,
                                                                            const std::vector<double> &)>(&RL::Simulators::CRD::CRDSimIslands::run_conditional_group_islandsTU),
           "Runs a CRD simulation with timing uncertainty in independent :param nb_groups groups, and "
           "later evaluates them by reshuffling the groups. It uses conditional agents.",
           py::arg("nb_evaluation_games"),
           py::arg("nb_groups"),
           py::arg("group_size"),
           py::arg("nb_generations"),
           py::arg("nb_games"),
           py::arg("min_rounds"),
           py::arg("mean_rounds"),
           py::arg("max_rounds"),
           py::arg("p"),
           py::arg("target"),
           py::arg("endowment"),
           py::arg("risk"),
           py::arg("available_actions"),
           py::arg("agent_type"),
           py::arg("*args"))
      .def("run_group_islandsThU",
           static_cast<DataTableCRD (RL::Simulators::CRD::CRDSimIslands::*)(size_t,
                                                                            size_t,
                                                                            size_t,
                                                                            size_t,
                                                                            size_t,
                                                                            size_t,
                                                                            int,
                                                                            double,
                                                                            int,
                                                                            double,
                                                                            RL::ActionSpace &,
                                                                            const std::string &,
                                                                            const std::vector<double> &)>(&RL::Simulators::CRD::CRDSimIslands::run_group_islandsThU),
           "Runs a CRD simulation with Threshold uncertainty in independent :param nb_groups groups, and "
           "later evaluates them by reshuffling the groups. It uses unconditional agents.",
           py::arg("nb_evaluation_games"),
           py::arg("nb_groups"),
           py::arg("group_size"),
           py::arg("nb_generations"),
           py::arg("nb_games"),
           py::arg("nb_rounds"),
           py::arg("target"),
           py::arg("delta"),
           py::arg("endowment"),
           py::arg("risk"),
           py::arg("available_actions"),
           py::arg("agent_type"),
           py::arg("*args"))
      .def("run_conditional_group_islandsThU",
           static_cast<DataTableCRD (RL::Simulators::CRD::CRDSimIslands::*)(size_t,
                                                                            size_t,
                                                                            size_t,
                                                                            size_t,
                                                                            size_t,
                                                                            size_t,
                                                                            int,
                                                                            double,
                                                                            int,
                                                                            double,
                                                                            RL::ActionSpace &,
                                                                            const std::string &,
                                                                            const std::vector<double> &)>(&RL::Simulators::CRD::CRDSimIslands::run_conditional_group_islandsThU),
           "Runs a CRD simulation with Threshold uncertainty in independent :param nb_groups groups, and "
           "later evaluates them by reshuffling the groups. It uses conditional agents.",
           py::arg("nb_evaluation_games"),
           py::arg("nb_groups"),
           py::arg("group_size"),
           py::arg("nb_generations"),
           py::arg("nb_games"),
           py::arg("nb_rounds"),
           py::arg("target"),
           py::arg("delta"),
           py::arg("endowment"),
           py::arg("risk"),
           py::arg("available_actions"),
           py::arg("agent_type"),
           py::arg("*args"))
      .def("run_group_islandsTUThU",
           static_cast<DataTableCRD (RL::Simulators::CRD::CRDSimIslands::*)(size_t,
                                                                            size_t,
                                                                            size_t,
                                                                            size_t,
                                                                            size_t,
                                                                            size_t,
                                                                            size_t,
                                                                            size_t,
                                                                            double,
                                                                            int,
                                                                            double,
                                                                            int,
                                                                            double,
                                                                            RL::ActionSpace &,
                                                                            const std::string &,
                                                                            const std::vector<double> &)>(&RL::Simulators::CRD::CRDSimIslands::run_group_islandsTUThU),
           "Runs a CRD simulation with timing uncertainty and threshold uncertainty "
           "in independent :param nb_groups groups, and "
           "later evaluates them by reshuffling the groups. It uses unconditional agents.",
           py::arg("nb_evaluation_games"),
           py::arg("nb_groups"),
           py::arg("group_size"),
           py::arg("nb_generations"),
           py::arg("nb_games"),
           py::arg("min_rounds"),
           py::arg("mean_rounds"),
           py::arg("max_rounds"),
           py::arg("p"),
           py::arg("target"),
           py::arg("delta"),
           py::arg("endowment"),
           py::arg("risk"),
           py::arg("available_actions"),
           py::arg("agent_type"),
           py::arg("*args"))
      .def("run_conditional_group_islandsTUThU",
           static_cast<DataTableCRD (RL::Simulators::CRD::CRDSimIslands::*)(size_t,
                                                                            size_t,
                                                                            size_t,
                                                                            size_t,
                                                                            size_t,
                                                                            size_t,
                                                                            size_t,
                                                                            size_t,
                                                                            double,
                                                                            int,
                                                                            double,
                                                                            int,
                                                                            double,
                                                                            RL::ActionSpace &,
                                                                            const std::string &,
                                                                            const std::vector<double> &)>(&RL::Simulators::CRD::CRDSimIslands::run_conditional_group_islandsTUThU),
           "Runs a CRD simulation with timing uncertainty and threshold uncertainty "
           "in independent :param nb_groups groups, and "
           "later evaluates them by reshuffling the groups. It uses unconditional agents.",
           py::arg("nb_evaluation_games"),
           py::arg("nb_groups"),
           py::arg("group_size"),
           py::arg("nb_generations"),
           py::arg("nb_games"),
           py::arg("min_rounds"),
           py::arg("mean_rounds"),
           py::arg("max_rounds"),
           py::arg("p"),
           py::arg("target"),
           py::arg("delta"),
           py::arg("endowment"),
           py::arg("risk"),
           py::arg("available_actions"),
           py::arg("agent_type"),
           py::arg("*args"))
      .def("run_population_islands",
           static_cast<DataTableCRD (RL::Simulators::CRD::CRDSimIslands::*)(size_t,
                                                                            size_t,
                                                                            size_t,
                                                                            size_t,
                                                                            size_t,
                                                                            size_t,
                                                                            size_t,
                                                                            int,
                                                                            int,
                                                                            double,
                                                                            RL::ActionSpace &,
                                                                            const std::string &,
                                                                            const std::vector<double> &)>(&RL::Simulators::CRD::CRDSimIslands::run_population_islands),
           "Runs a CRD simulation in independent :param nb_groups well-mixed populations, and "
           "later evaluates them by reshuffling the groups. It uses unconditional agents.",
           py::arg("nb_evaluation_games"),
           py::arg("nb_populations"),
           py::arg("population_size"),
           py::arg("group_size"),
           py::arg("nb_generations"),
           py::arg("nb_games"),
           py::arg("nb_rounds"),
           py::arg("target"),
           py::arg("endowment"),
           py::arg("risk"),
           py::arg("available_actions"),
           py::arg("agent_type"),
           py::arg("*args"))
      .def("run_conditional_population_islands",
           static_cast<DataTableCRD (RL::Simulators::CRD::CRDSimIslands::*)(size_t,
                                                                            size_t,
                                                                            size_t,
                                                                            size_t,
                                                                            size_t,
                                                                            size_t,
                                                                            size_t,
                                                                            int,
                                                                            int,
                                                                            double,
                                                                            RL::ActionSpace &,
                                                                            const std::string &,
                                                                            const std::vector<double> &)>(&RL::Simulators::CRD::CRDSimIslands::run_conditional_population_islands),
           "Runs a CRD simulation in independent :param nb_groups well-mixed populations, and "
           "later evaluates them by reshuffling the groups. It uses conditional agents.",
           py::arg("nb_evaluation_games"),
           py::arg("nb_populations"),
           py::arg("population_size"),
           py::arg("group_size"),
           py::arg("nb_generations"),
           py::arg("nb_games"),
           py::arg("nb_rounds"),
           py::arg("target"),
           py::arg("endowment"),
           py::arg("risk"),
           py::arg("available_actions"),
           py::arg("agent_type"),
           py::arg("*args"))
      .def("run_population_islandsTU",
           static_cast<DataTableCRD (RL::Simulators::CRD::CRDSimIslands::*)(size_t,
                                                                            size_t,
                                                                            size_t,
                                                                            size_t,
                                                                            size_t,
                                                                            size_t,
                                                                            size_t,
                                                                            size_t,
                                                                            size_t,
                                                                            double,
                                                                            int,
                                                                            int,
                                                                            double,
                                                                            RL::ActionSpace &,
                                                                            const std::string &,
                                                                            const std::vector<double> &)>(&RL::Simulators::CRD::CRDSimIslands::run_population_islandsTU),
           "Runs a CRD simulation with timing uncertainty in independent :param nb_groups well-mixed populations, and "
           "later evaluates them by reshuffling the groups. It uses unconditional agents.",
           py::arg("nb_evaluation_games"),
           py::arg("nb_populations"),
           py::arg("population_size"),
           py::arg("group_size"),
           py::arg("nb_generations"),
           py::arg("nb_games"),
           py::arg("min_rounds"),
           py::arg("mean_rounds"),
           py::arg("max_rounds"),
           py::arg("p"),
           py::arg("target"),
           py::arg("endowment"),
           py::arg("risk"),
           py::arg("available_actions"),
           py::arg("agent_type"),
           py::arg("*args"))
      .def("run_conditional_population_islandsTU",
           static_cast<DataTableCRD (RL::Simulators::CRD::CRDSimIslands::*)(size_t,
                                                                            size_t,
                                                                            size_t,
                                                                            size_t,
                                                                            size_t,
                                                                            size_t,
                                                                            size_t,
                                                                            size_t,
                                                                            size_t,
                                                                            double,
                                                                            int,
                                                                            int,
                                                                            double,
                                                                            RL::ActionSpace &,
                                                                            const std::string &,
                                                                            const std::vector<double> &)>(&RL::Simulators::CRD::CRDSimIslands::run_conditional_population_islandsTU),
           "Runs a CRD simulation with timing uncertainty in independent :param nb_groups well-mixed populations, and "
           "later evaluates them by reshuffling the groups. It uses conditional agents.",
           py::arg("nb_evaluation_games"),
           py::arg("nb_populations"),
           py::arg("population_size"),
           py::arg("group_size"),
           py::arg("nb_generations"),
           py::arg("nb_games"),
           py::arg("min_rounds"),
           py::arg("mean_rounds"),
           py::arg("max_rounds"),
           py::arg("p"),
           py::arg("target"),
           py::arg("endowment"),
           py::arg("risk"),
           py::arg("available_actions"),
           py::arg("agent_type"),
           py::arg("*args"))
      .def("run_population_islandsThU",
           static_cast<DataTableCRD (RL::Simulators::CRD::CRDSimIslands::*)(size_t,
                                                                            size_t,
                                                                            size_t,
                                                                            size_t,
                                                                            size_t,
                                                                            size_t,
                                                                            size_t,
                                                                            int,
                                                                            double,
                                                                            int,
                                                                            double,
                                                                            RL::ActionSpace &,
                                                                            const std::string &,
                                                                            const std::vector<double> &)>(&RL::Simulators::CRD::CRDSimIslands::run_population_islandsThU),
           "Runs a CRD simulation with threshold uncertainty "
           "in independent :param nb_groups well-mixed populations, and "
           "later evaluates them by reshuffling the groups. It uses unconditional agents.",
           py::arg("nb_evaluation_games"),
           py::arg("nb_populations"),
           py::arg("population_size"),
           py::arg("group_size"),
           py::arg("nb_generations"),
           py::arg("nb_games"),
           py::arg("nb_rounds"),
           py::arg("target"),
           py::arg("delta"),
           py::arg("endowment"),
           py::arg("risk"),
           py::arg("available_actions"),
           py::arg("agent_type"),
           py::arg("*args"))
      .def("run_conditional_population_islandsThU",
           static_cast<DataTableCRD (RL::Simulators::CRD::CRDSimIslands::*)(size_t,
                                                                            size_t,
                                                                            size_t,
                                                                            size_t,
                                                                            size_t,
                                                                            size_t,
                                                                            size_t,
                                                                            int,
                                                                            double,
                                                                            int,
                                                                            double,
                                                                            RL::ActionSpace &,
                                                                            const std::string &,
                                                                            const std::vector<double> &)>(&RL::Simulators::CRD::CRDSimIslands::run_conditional_population_islandsThU),
           "Runs a CRD simulation with threshold uncertainty "
           "in independent :param nb_groups well-mixed populations, and "
           "later evaluates them by reshuffling the groups. It uses conditional agents.",
           py::arg("nb_evaluation_games"),
           py::arg("nb_populations"),
           py::arg("population_size"),
           py::arg("group_size"),
           py::arg("nb_generations"),
           py::arg("nb_games"),
           py::arg("nb_rounds"),
           py::arg("target"),
           py::arg("delta"),
           py::arg("endowment"),
           py::arg("risk"),
           py::arg("available_actions"),
           py::arg("agent_type"),
           py::arg("*args"))
      .def("run_population_islandsTUThU",
           static_cast<DataTableCRD (RL::Simulators::CRD::CRDSimIslands::*)(size_t,
                                                                            size_t,
                                                                            size_t,
                                                                            size_t,
                                                                            size_t,
                                                                            size_t,
                                                                            size_t,
                                                                            size_t,
                                                                            size_t,
                                                                            double,
                                                                            int,
                                                                            double,
                                                                            int,
                                                                            double,
                                                                            RL::ActionSpace &,
                                                                            const std::string &,
                                                                            const std::vector<double> &)>(&RL::Simulators::CRD::CRDSimIslands::run_population_islandsTUThU),
           "Runs a CRD simulation with timing uncertainty and threshold uncertainty"
           " in independent :param nb_groups well-mixed populations, and "
           "later evaluates them by reshuffling the groups. It uses unconditional agents.",
           py::arg("nb_evaluation_games"),
           py::arg("nb_populations"),
           py::arg("population_size"),
           py::arg("group_size"),
           py::arg("nb_generations"),
           py::arg("nb_games"),
           py::arg("min_rounds"),
           py::arg("mean_rounds"),
           py::arg("max_rounds"),
           py::arg("p"),
           py::arg("target"),
           py::arg("delta"),
           py::arg("endowment"),
           py::arg("risk"),
           py::arg("available_actions"),
           py::arg("agent_type"),
           py::arg("*args"))
      .def("run_conditional_population_islandsTUThU",
           static_cast<DataTableCRD (RL::Simulators::CRD::CRDSimIslands::*)(size_t,
                                                                            size_t,
                                                                            size_t,
                                                                            size_t,
                                                                            size_t,
                                                                            size_t,
                                                                            size_t,
                                                                            size_t,
                                                                            size_t,
                                                                            double,
                                                                            int,
                                                                            double,
                                                                            int,
                                                                            double,
                                                                            RL::ActionSpace &,
                                                                            const std::string &,
                                                                            const std::vector<double> &)>(&RL::Simulators::CRD::CRDSimIslands::run_conditional_population_islandsTUThU),
           "Runs a CRD simulation with timing uncertainty and threshold uncertainty"
           " in independent :param nb_groups well-mixed populations, and "
           "later evaluates them by reshuffling the groups. It uses conditional agents.",
           py::arg("nb_evaluation_games"),
           py::arg("nb_populations"),
           py::arg("population_size"),
           py::arg("group_size"),
           py::arg("nb_generations"),
           py::arg("nb_games"),
           py::arg("min_rounds"),
           py::arg("mean_rounds"),
           py::arg("max_rounds"),
           py::arg("p"),
           py::arg("target"),
           py::arg("delta"),
           py::arg("endowment"),
           py::arg("risk"),
           py::arg("available_actions"),
           py::arg("agent_type"),
           py::arg("*args"));

#ifdef VERSION_INFO
  m.attr("__version__") = VERSION_INFO;
#else
  m.attr("__version__") = "dev0.10.0";
#endif

}