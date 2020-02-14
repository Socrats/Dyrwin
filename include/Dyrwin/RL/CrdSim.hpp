#include <utility>

//
// Created by Elias Fernandez on 2019-05-10.
//

#ifndef DYRWIN_RL_CRDSIM_HPP
#define DYRWIN_RL_CRDSIM_HPP

#include <random>
#include <unordered_set>
#include <Dyrwin/RL/BatchQLearningAgent.h>
#include <Dyrwin/RL/QLearningAgent.h>
#include <Dyrwin/RL/RothErevAgent.h>
#include <Dyrwin/RL/HistericQLearningAgent.hpp>
#include <Dyrwin/RL/CRDGame.h>
#include <Dyrwin/RL/CRDConditional.h>
#include <Dyrwin/RL/PopContainer.hpp>
#include <Dyrwin/RL/Utils.h>
#include <Dyrwin/RL/Data.hpp>
#include <Dyrwin/Sampling.h>
#include <Dyrwin/SeedGenerator.h>
#include <Dyrwin/OpenMPUtils.hpp>

namespace EGTTools::RL {
class CRDSim {
 public:
  /**
   * @brief Implements a simulator for the Collective Risk dilemma.
   *
   * The main objective of this class is to provide a container for the main
   * simulations that need to be performed in the context of the Collective-risk Dilemma
   * with RL agents.
   *
   * @param nb_episodes number of episodes to be run for each group
   * @param nb_games number of games per episode
   * @param nb_rounds number of rounds per game
   * @param nb_actions number of actions that each player can take
   * @param group_size size of a group
   * @param risk probability of cataclysm
   * @param threshold : target of the game
   * @param available_actions : vector of available actions/donations per round
   * @param agent_type algorithm that the agent uses to learn
   * @param args extra arguments for the particular agent_type
   */
  CRDSim(size_t nb_episodes, size_t nb_games,
         size_t nb_rounds,
         size_t nb_actions,
         size_t group_size,
         double risk,
         double endowment,
         double threshold,
         const ActionSpace &available_actions,
         const std::string &agent_type,
         const std::vector<double> &args = {});

  /**
       * @brief Runs a simulation of the CRD for only 1 group.
       *
       * This method runs a simulation of the CRD for @param nb_episodes.
       * Each episode consists of @param nb_games. The policy of the agents
       * is only updated once an episode finishes.
       *
       * @param nb_episodes number of episodes during which the agents will learn
       * @param nb_games number of games per episode
       * @return EGTTools::Matrix2D containing the group achievement and average contribution
       *         for each episode.
       */
  Matrix2D run(size_t nb_episodes, size_t nb_games);

  /**
   * @brief Runs a CRD simulation where the population consists of a single group that always remains the same.
   *
   * Since the group does not change, players always interact among each other and adapt to the
   * specific characteristics of the group.
   *
   * At each generations the group players \param nb_games before updating the policy.
   *
   * @param group_size : size of the group/population
   * @param nb_generations : number of generations through which the agents will adapt
   * @param nb_games : number of games per generation
   * @param threshold : collective target of the game
   * @param risk : probability that all players will loose their remaining endowment
   * @param agent_type : string indicating the algorithm that agents use to adapt
   * @param args : arguments for the agent
   * @return : a data container with the population, the average success and the average contributions per generation.
   */
  DataTypes::CRDData run(size_t group_size,
                         size_t nb_generations,
                         size_t nb_games,
                         double threshold,
                         double risk,
                         const std::string &agent_type,
                         const std::vector<double> &args = {});

  /**
       * @brief Runs a simulation of the CRD for multiple independent groups.
       *
       * This method runs a simulation of the CRD for @param nb_episodes.
       * Each episode consists of @param nb_games. The policy of the agents
       * is only updated once an episode finishes. The simulation is performed
       * for @param nb_groups independent groups, and the results of each episode
       * are averaged over groups.
       *
       * @param nb_episodes number of episodes during which the agents will learn
       * @param nb_games number of games per episode
       * @param nb_groups number of independent groups
       * @return EGTTools::Matrix2D containing the group achievement and average contribution
       *         for each episode.
       */
  Matrix2D
  run(size_t nb_episodes, size_t nb_games, size_t nb_groups, double risk, const std::vector<double> &args = {});

  /**
       * @brief runs the Collective risk dilemma with independent groups of agents.
       *
       * Trains a population of size @param nb_groups * @param group_size of RL agents of
       * type @param agent_type. The agents are subdivided into @param nb_groups groups
       * and during training time they only interact with members of their group.
       *
       * Each agent's strategy is only updated after each episode.
       *
       * Each group is independently run in parallel. Once the simulation finishes, the
       * learning data is transfered to a CRDData container. The population of agents
       * is also inserted in the container. Finally this method returns the data container.
       *
       * @param nb_groups : number of independent groups
       * @param group_size : size of the group
       * @param nb_episodes : number of episodes
       * @param nb_games : number of games per episode
       * @param risk : probability that agents lose their endowment if the target isn't met
       * @param transient : transient time to be discarded on the computation of the average success
       * @param agent_type : learning algorithm used by the agent
       * @param args : vector of arguments for the creation of the agent
       * @return a data container with the results of the simulation
       */
  DataTypes::CRDDataIslands
  run(size_t nb_groups, size_t group_size, size_t nb_episodes, size_t nb_games, double risk,
      size_t transient, const std::string &agent_type, const std::vector<double> &args = {});

  /**
       * @brief Runs simulation with a well mixed population.
       *
       * In the simulations performed here, agents of a population of size Z = _group_size * nb_groups
       * are selected randomly from the population to form a group of size _group_size and play a game.
       * At each generation @param nb_games are played. The simulation is run for @param nb_generations.
       *
       * The @param args is a vector that should contain the arguments specific of the agent type of
       * the population.
       *
       *
       * @param nb_generations : number of generations per simulation
       * @param nb_games : number of games per generation
       * @param nb_groups : will define the population size (Z = nb_groups * _group_size)
       * @param threshold : collective target
       * @param risk : probability of loosing all endowment if the target isn't reached
       * @param args : vector of arguments to instantiate the agent_type of the population
       * @return an Eigen 2D matrix with the average group achievement and average donations at each generation.
       */
  Matrix2D
  runWellMixed(size_t nb_generations, size_t nb_games, size_t nb_groups, size_t group_size, double threshold,
               double risk,
               const std::vector<double> &args = {});

  /**
       * @brief Runs simulation with a well mixed population and returns a data container with the population.
       *
       * In the simulations performed here, agents of a population of size Z = _group_size * nb_groups
       * are selected randomly from the population to form a group of size _group_size and play a game.
       * At each generation @param nb_games are played. The simulation is run for @param nb_generations.
       *
       * The @param args is a vector that should contain the arguments specific of the agent type of
       * the population.
       *
       * \param pop_size must always be bigger than \param group_size!
       *
       * @param pop_size : size of the population
       * @param group_size : group size
       * @param nb_generations : number of generations per simulation
       * @param nb_games : number of games per generation
       * @param threshold : collective target
       * @param risk : probability of loosing all endowment if the target isn't reached
       * @param agent_type : string indicating which agent implementation to use
       * @param args : vector of arguments to instantiate the agent_type of the population
       * @return a data container that includes the group achievement, the average donations and the population
       */
  DataTypes::CRDData
  runWellMixed(size_t pop_size, size_t group_size, size_t nb_generations, size_t nb_games, double threshold,
               double risk, const std::string &agent_type,
               const std::vector<double> &args = {});

  /**
   * @brief runs the CRD in a well-mixed population with synchronised updates.
   *
   * \param pop_size must always be bigger than \param group_size!
   *
   * @param pop_size : size of the population
   * @param group_size : size of the group
   * @param nb_generations : number of generations
   * @param nb_games : number of games
   * @param threshold : collective target of the game
   * @param risk : risk that all players will receive 0 payoff if the target isn't met
   * @param agent_type : string indicating the algorithm, that the agents use to adapt
   * @param args : vector of arguments for the agent
   * @return : a data structure containing the population and the average success and contributions.
   */
  DataTypes::CRDData
  runWellMixedSync(size_t pop_size, size_t group_size, size_t nb_generations, size_t nb_games, double threshold,
                   double risk, const std::string &agent_type,
                   const std::vector<double> &args = {});

  /**
       * @brief Runs several simulations with a well mixed population.
       *
       * In the simulations performed here, agents of a population of size Z = _group_size * nb_groups
       * are selected randomly from the population to form a group of size _group_size and play a game.
       * At each generation @param nb_games are played. The simulation is run for @param nb_generations.
       *
       * The @param args is a vector that should contain the arguments specific of the agent type of
       * the population.
       *
       * The simulation is repeated for @param nb_runs with independent populations.
       *
       * \param pop_size must always be bigger than \param group_size!
       *
       * @param nb_runs : number of independent simulations
       * @param pop_size : size of the population
       * @param group_size : size of the group
       * @param nb_generations : number of generations per simulation
       * @param nb_games : number of games per generation
       * @param threshold : collective target
       * @param risk : probability of loosing all endowment if the target isn't reached
       * @param transient : number of generations to take into account for calculating the average
       * @param agent_type : string indicating whcoh agent implementation to use
       * @param args : vector of arguments to instantiate the agent_type of the population
       * @return Eigen 2D matrix with the average group achievement and avg. donations across independent runs.
       */
  Matrix2D runWellMixed(size_t nb_runs, size_t pop_size, size_t group_size, size_t nb_generations,
                        size_t nb_games, double threshold, double risk, size_t transient,
                        const std::string &agent_type,
                        const std::vector<double> &args = {});

  /**
       * @brief Runs several simulations with a well mixed population with synchronised updates.
       *
       * In the simulations performed here, agents of a population of size Z = _group_size * nb_groups
       * are selected randomly from the population to form a group of size _group_size and play a game.
       * At each generation @param nb_games are played. The simulation is run for @param nb_generations.
       *
       * The @param args is a vector that should contain the arguments specific of the agent type of
       * the population.
       *
       * The simulation is repeated for @param nb_runs with independent populations.
       *
       * \param pop_size must always be bigger than \param group_size!
       *
       * @param nb_runs : number of independent simulations
       * @param pop_size : size of the population
       * @param group_size : size of the group
       * @param nb_generations : number of generations per simulation
       * @param nb_games : number of games per generation
       * @param threshold : collective target
       * @param risk : probability of loosing all endowment if the target isn't reached
       * @param transient : number of generations to take into account for calculating the average
       * @param agent_type : string indicating whcoh agent implementation to use
       * @param args : vector of arguments to instantiate the agent_type of the population
       * @return Eigen 2D matrix with the average group achievement and avg. donations across independent runs.
       */
  Matrix2D
  runWellMixedSync(size_t nb_runs, size_t pop_size, size_t group_size, size_t nb_generations, size_t nb_games,
                   double threshold, double risk, size_t transient,
                   const std::string &agent_type,
                   const std::vector<double> &args = {});

  /**
       * @brief Runs a simulation with timing uncertainty
       *
       * This simulations run the Collective Risk Game with timing uncertainty specified in [Domingos et al. 2019].
       * Here the number of rounds of the game is uncertain. The game will always take at least @param min_rounds
       * and a maximum of @param max_rounds. After @param min_rounds, the game will end with probability @param p.
       *
       * If @param mean_rounds is != 0, @param p will be ignored, instead p = 1 / [(10 - min_rounds) + 1] so that
       * the average of the geometric distribution that defines the final round is @param avg_rounds.
       *
       * @param nb_episodes : nb_episodes number of episodes during which the agents will learn
       * @param nb_games : number of games per episode
       * @param min_rounds : minimum number of rounds
       * @param mean_rounds : average number of rounds
       * @param max_rounds : maximum number of rounds (used so that the rounds of the game is
       *                     never bigger than the number of states of the agent)
       * @param p : probability of the game ending after each round (starting from min_rounds)
       * @param risk : probability of loosing all endowment if the target isn't reached
       * @param args : vector of arguments to instantiate the agent_type of the population
       * @param crd_type : type of reinforcement
       * @return an Eigen 2D matrix with the average group achievement and average donations at each episode.
       */
  Matrix2D
  runTimingUncertainty(size_t nb_episodes, size_t nb_games, size_t min_rounds, size_t mean_rounds,
                       size_t max_rounds, double p,
                       double risk,
                       const std::vector<double> &args = {}, const std::string &crd_type = "milinski");

  /**
       * @brief trains a well-mixed population in the CRD with Timing uncertainty
       *
       * This simulations run the Collective Risk Game with timing uncertainty specified in [Domingos et al. 2019].
       * Here the number of rounds of the game is uncertain. The game will always take at least @param min_rounds
       * and a maximum of @param max_rounds. After @param min_rounds, the game will end with probability @param p.
       *
       * If @param mean_rounds is != 0, @param p will be ignored, instead p = 1 / [(10 - min_rounds) + 1] so that
       * the average of the geometric distribution that defines the final round is @param avg_rounds.
       *
       * In the simulations performed here, agents of a population of size Z = @param pop_size
       * are selected randomly from the population to form a group of size @param group_size and play a game.
       * At each generation @param nb_games are played. The simulation is run for @param nb_generations.
       *
       * The @param args is a vector that should contain the arguments specific of the @param agent_type of
       * the population.
       *
       * The results are transfered to a data container and returned.
       *
       * \param pop_size must always be bigger than \param group_size!
       *
       * @param pop_size : size of the population
       * @param group_size : group size
       * @param nb_generations : total number of generations
       * @param nb_games : number of games per generation
       * @param risk : risk of losing the remaining endowment if the target isn't reached
       * @param agent_type : algorithm type of the agent
       * @param args : arguments for the agent
       * @return a data container
       */
  DataTypes::CRDData
  runWellMixedTU(size_t pop_size, size_t group_size, size_t nb_generations, size_t nb_games, double threshold,
                 double risk, size_t min_rounds, size_t mean_rounds, size_t max_rounds, double p,
                 const std::string &agent_type,
                 const std::vector<double> &args = {});

  DataTypes::CRDData
  runWellMixedTUSync(size_t pop_size, size_t group_size, size_t nb_generations, size_t nb_games,
                     double threshold, double risk, size_t min_rounds, size_t mean_rounds, size_t max_rounds,
                     double p,
                     const std::string &agent_type,
                     const std::vector<double> &args = {});

  /**
   * @brief runs several independent simulations with a well-mixed population in the CRD with Timing uncertainty
   *
   * This simulations run the Collective Risk Game with timing uncertainty specified in [Domingos et al. 2019].
   * Here the number of rounds of the game is uncertain. The game will always take at least @param min_rounds
   * and a maximum of @param max_rounds. After @param min_rounds, the game will end with probability @param p.
   *
   * If @param mean_rounds is != 0, @param p will be ignored, instead p = 1 / [(10 - min_rounds) + 1] so that
   * the average of the geometric distribution that defines the final round is @param avg_rounds.
   *
   * In the simulations performed here, agents of a population of size Z = @param pop_size
   * are selected randomly from the population to form a group of size @param group_size and play a game.
   * At each generation @param nb_games are played. The simulation is run for @param nb_generations.
   *
   * The @param args is a vector that should contain the arguments specific of the @param agent_type of
   * the population.
   *
   * Returns a matrix with the group achievement and average contributions of the players for
   * each independent simulation.
   *
   * @param nb_runs : number of independent simulations
   * @param pop_size : size of the population
   * @param group_size : size of a group
   * @param nb_generations : number of generations of each simulation
   * @param nb_games : number of games per generation
   * @param threshold : threshold of the game
   * @param risk : risk (impact uncertainty) of the game
   * @param transient : the results are average over the last \p transient generations
   * @param min_rounds : minimum number of rounds of each game
   * @param mean_rounds : mean number of rounds per game
   * @param max_rounds : max number of rounds per game
   * @param p : probability of the game ending after the minimum number of rounds
   * @param agent_type : string indicating the algorithm that controls how each agent adapts
   * @param args : arguments for the algorithm
   * @return : matrix containing the averaged group achievement and contributions per simulation
   */
  Matrix2D
  runWellMixedTU(size_t nb_runs, size_t pop_size, size_t group_size, size_t nb_generations, size_t nb_games,
                 double threshold,
                 double risk, size_t transient, size_t min_rounds, size_t mean_rounds, size_t max_rounds,
                 double p,
                 const std::string &agent_type,
                 const std::vector<double> &args = {});

  Matrix2D
  runWellMixedTUSync(size_t nb_runs, size_t pop_size, size_t group_size, size_t nb_generations, size_t nb_games,
                     double threshold,
                     double risk, size_t transient, size_t min_rounds, size_t mean_rounds, size_t max_rounds,
                     double p,
                     const std::string &agent_type,
                     const std::vector<double> &args = {});

  /**
       * @brief Runs a simulation with threshold uncertainty
       *
       * This simulations run the Collective Risk Game with threshold uncertainty.
       * Here the threshold is uncertain. The threshold of the game is a stochastic uniform
       * variable in the range \f$(threshold - delta/2, threshold + delta/2)\f$ .
       *
       * In the simulations performed here, agents of a population of size Z = @param pop_size
       * are selected randomly from the population to form a group of size @param group_size and play a game.
       * At each generation @param nb_games are played. The simulation is run for @param nb_generations.
       *
       * @param pop_size : size of the population
       * @param group_size : group size
       * @param nb_generations : number of training time-steps
       * @param nb_games : number of games per time-step / generation
       * @param threshold : target of the game
       * @param risk : probability of loosing all endowment if the target isn't reached
       * @param delta : variance / uncertainty of the threshold
       * @param agent_type : string indicating the learning algorithm that the agents will use
       * @param args : specific parameters for the learning algorithm
       * @return a data container with the information of the simulation and a pointer to the population.
       */
  DataTypes::CRDData
  runWellMixedThU(size_t pop_size, size_t group_size, size_t nb_generations, size_t nb_games,
                  size_t threshold, size_t delta, double risk,
                  const std::string &agent_type,
                  const std::vector<double> &args = {});

  Matrix2D runWellMixedThU(size_t nb_runs, size_t pop_size, size_t group_size, size_t nb_generations,
                           size_t nb_games, size_t threshold, size_t delta, double risk, size_t transient,
                           const std::string &agent_type,
                           const std::vector<double> &args = {});

  /**
     * @brief Runs a simulation with threshold uncertainty
     *
     * This simulations run the Collective Risk Game with threshold uncertainty.
     * Here the threshold is uncertain. The threshold of the game is a stochastic uniform
     * variable in the range \f$(threshold - delta/2, threshold + delta/2)\f$ .
     *
     * In the simulations performed here, agents of a population of size Z = @param pop_size
     * are selected randomly from the population to form a group of size @param group_size and play a game.
     * At each generation @param nb_games are played. The simulation is run for @param nb_generations.
     *
     * @param pop_size : size of the population
     * @param group_size : group size
     * @param nb_generations : number of training time-steps
     * @param nb_games : number of games per time-step / generation
     * @param threshold : target of the game
     * @param risk : probability of loosing all endowment if the target isn't reached
     * @param delta : variance / uncertainty of the threshold
     * @param agent_type : string indicating the learning algorithm that the agents will use
     * @param args : specific parameters for the learning algorithm
     * @return a data container with the information of the simulation and a pointer to the population.
     */
  DataTypes::CRDData
  runWellMixedThUSync(size_t pop_size, size_t group_size, size_t nb_generations, size_t nb_games,
                      size_t threshold, size_t delta, double risk,
                      const std::string &agent_type,
                      const std::vector<double> &args = {});

  Matrix2D runWellMixedThUSync(size_t nb_runs, size_t pop_size, size_t group_size, size_t nb_generations,
                               size_t nb_games, size_t threshold, size_t delta, double risk, size_t transient,
                               const std::string &agent_type,
                               const std::vector<double> &args = {});

  /**
   * @brief Runs one simulation with both Timing uncertainty and Threshold uncertainty
   *
   * This method permits the exploration of both uncertainties at the same time.
   *
   *
   * The threshold of the game is a stochastic uniform
   * variable in the range \f$(threshold - delta/2, threshold + delta/2)\f$ .
   *
   * @param pop_size : size of the population
   * @param group_size : size of the group that plays the game
   * @param nb_generations : number of generations per simulation
   * @param threshold : threshold of the game
   * @param delta : threshold uncertainty (variation of the threshold)
   * @param risk : risk (impact uncertainty) of the game
   * @param min_rounds : minimum number of rounds in a game
   * @param mean_rounds : mean number of rounds in a game
   * @param max_rounds : maximum number of rounds per game
   * @param p : probability of the game ending after the minimum number of rounds
   * @param agent_type : algorithm used by each agent to adapt in the population
   * @param args : arguments of the algorithm
   * @return a data structure containing the result fo the simulation
   */
  DataTypes::CRDData
  runWellMixedTUnThU(size_t pop_size, size_t group_size, size_t nb_generations, size_t nb_games,
                     size_t threshold, size_t delta, double risk, size_t min_rounds, size_t mean_rounds,
                     size_t max_rounds, double p,
                     const std::string &agent_type,
                     const std::vector<double> &args = {});

  /**
   * @brief Runs several independent simulations with both Timing uncertainty and Threshold uncertainty
   *
   * This method permits the exploration of both uncertainties at the same time.
   *
   * @param nb_runs : number of independent simulations to run
   * @param pop_size : size of the population
   * @param group_size : size of the group that plays the game
   * @param nb_generations : number of generations per simulation
   * @param threshold : threshold of the game
   * @param delta : threshold uncertainty (variation of the threshold)
   * @param risk : risk (impact uncertainty) of the game
   * @param min_rounds : minimum number of rounds in a game
   * @param mean_rounds : mean number of rounds in a game
   * @param max_rounds : maximum number of rounds per game
   * @param p : probability of the game ending after the minimum number of rounds
   * @param agent_type : algorithm used by each agent to adapt in the population
   * @param args : arguments of the algorithm
   * @return a matrix containing the averaged group_achievement and player's contributions over the simulations.
   */
  Matrix2D
  runWellMixedTUnThU(size_t nb_runs, size_t pop_size, size_t group_size, size_t nb_generations, size_t nb_games,
                     size_t threshold, size_t delta, double risk, size_t transient, size_t min_rounds,
                     size_t mean_rounds, size_t max_rounds, double p,
                     const std::string &agent_type,
                     const std::vector<double> &args = {});

  /**
       * @brief Runs a simulation with conditional agents.
       *
       * This method can only be run on actions of the type [0, 1, 2....], i.e., sequential from [0, nb_actions -1].
       *
       * @param nb_episodes : number of times the policy of the agent will be updated based on the propensity matrix.
       * @param nb_games : number of games per episode
       * @param args : vector of arguments to instantiate the agent_type of the population
       * @param crd_type : type of crd game (milinski or xico)
       * @return an Eigen 2D matrix with the average group achievement and average donations at each generation.
       */
  Matrix2D runConditional(size_t nb_episodes, size_t nb_games, const std::vector<double> &args = {},
                          const std::string &crd_type = "milinski");

  /**
   * Runs a CRD simulation with conditional agents.
   *
   * The simulation runs with \p group_size autonomous agents using the RL algorithm
   * specified in \p agent_type, with the arguments specified in \p args, for
   * a total of \p nb_episodes, each of them consisting of \p nb_games.
   * Each game consists of t rounds, specified in the \p nb_rounds of this class.
   *
   * @param group_size
   * @param nb_episodes
   * @param nb_games : number of games per episode
   * @param risk : risk (impact uncertainty) of the game
   * @param agent_type
   * @param args
   * @return
   */
  DataTypes::CRDData
  runConditional(size_t group_size, size_t nb_episodes, size_t nb_games, double risk,
                 const std::string &agent_type, const std::vector<double> &args = {});

  /**
       * @brief Runs a simulation with timing uncertainty with conditional agents.
       *
       * This simulations run the Collective Risk Game with timing uncertainty specified in [Domingos et al. 2019].
       * Here the number of rounds of the game is uncertain. The game will always take at least \p min_rounds
       * and a maximum of \p max_rounds. After \p min_rounds, the game will end with probability \p p.
       *
       * If \p mean_rounds is != 0, \p p will be ignored, instead p = 1 / [(10 - min_rounds) + 1] so that
       * the average of the geometric distribution that defines the final round is \p avg_rounds.
       *
       * @param nb_episodes : nb_episodes number of episodes during which the agents will learn
       * @param nb_games : number of games per episode
       * @param min_rounds : minimum number of rounds
       * @param mean_rounds : average number of rounds
       * @param max_rounds : maximum number of rounds (used so that the rounds of the game is
       *                     never bigger than the number of states of the agent)
       * @param p : probability of the game ending after each round (starting from min_rounds)
       * @param risk : probability of loosing all endowment if the target isn't reached
       * @param args : vector of arguments to instantiate the agent_type of the population
       * @param crd_type : type of reinforcement
       * @return an Eigen 2D matrix with the average group achievement and average donations at each episode.
       */
  Matrix2D
  runConditionalTU(size_t nb_episodes, size_t nb_games, size_t min_rounds, size_t mean_rounds,
                   size_t max_rounds, double p,
                   double risk,
                   const std::vector<double> &args = {}, const std::string &crd_type = "milinski");

  /**
       * @brief Runs several independent simulations with conditional agents.
       *
       * This method can only be run on actions of the type [0, 1, 2....], i.e., sequential from [0, nb_actions -1].
       * A total of @param nb_groups independent simulations will be run. The simulation will return the average
       * group achievement and donations across simulations.
       *
       * This simulation uses agents whose state consists of a tuple \f$(t, d^{t-1}_{-i})\f$,
       * where t is the current round of the game, and \f$d^{t-1}_{-i}\f$ is the sum of donations
       * of the group, without the focal player i, at round t-1. At \f$t=0\f$, \f$d^{t-1}_{-i}\f=0$.
       * In this fashion, agent are able to explore strategy profiles that are conditional to
       * the other members of the group.
       *
       * @param nb_episodes : number of times the policy of the agent will be updated based on the propensity matrix.
       * @param nb_games : number of games per episode
       * @param nb_groups : number of independent simulations
       * @param risk : probability of loosing all endowment if the target isn't reached
       * @param args : vector of arguments to instantiate the agent_type of the population
       * @param crd_type : type of crd game (milinski or xico)
       * @return Eigene 2D matrix with the average group achievement and average donations across independent runs.
       */
  Matrix2D
  runConditional(size_t nb_episodes, size_t nb_games, size_t nb_groups, double risk,
                 const std::vector<double> &args = {}, const std::string &crd_type = "milinski");

  /**
       * @brief Runs simulation with a well mixed population and returns a data container with the population.
       *
       * In the simulations performed here, agents of a population of size Z = _group_size * nb_groups
       * are selected randomly from the population to form a group of size _group_size and play a game.
       * At each generation @param nb_games are played. The simulation is run for @param nb_generations.
       *
       * The @param args is a vector that should contain the arguments specific of the agent type of
       * the population.
       *
       * This simulation uses agents whose state consists of a tuple \f$(t, d^{t-1}_{-i})\f$,
       * where t is the current round of the game, and \f$d^{t-1}_{-i}\f$ is the sum of donations
       * of the group, without the focal player i, at round t-1. At \f$t=0\f$, \f$d^{t-1}_{-i}\f=0$.
       * In this fashion, agent are able to explore strategy profiles that are conditional to
       * the other members of the group.
       *
       * @param pop_size : size of the population
       * @param group_size : group size
       * @param nb_generations : number of generations per simulation
       * @param nb_games : number of games per generation
       * @param threshold : collective target
       * @param risk : probability of loosing all endowment if the target isn't reached
       * @param agent_type : string indicating which agent implementation to use
       * @param args : vector of arguments to instantiate the agent_type of the population
       * @return a data container that includes the group achievement, the average donations and the population
       */
  DataTypes::CRDData
  runConditionalWellMixed(size_t pop_size, size_t group_size, size_t nb_generations, size_t nb_games,
                          double threshold,
                          double risk, const std::string &agent_type,
                          const std::vector<double> &args = {});

  /**
       * @brief Runs several simulations with a well mixed population.
       *
       * In the simulations performed here, agents of a population of size Z = _group_size * nb_groups
       * are selected randomly from the population to form a group of size _group_size and play a game.
       * At each generation @param nb_games are played. The simulation is run for @param nb_generations.
       *
       * The @param args is a vector that should contain the arguments specific of the agent type of
       * the population.
       *
       * The simulation is repeated for @param nb_runs with independent populations.
       *
       * This simulation uses agents whose state consists of a tuple \f$(t, d^{t-1}_{-i})\f$,
       * where t is the current round of the game, and \f$d^{t-1}_{-i}\f$ is the sum of donations
       * of the group, without the focal player i, at round t-1. At \f$t=0\f$, \f$d^{t-1}_{-i}\f=0$.
       * In this fashion, agent are able to explore strategy profiles that are conditional to
       * the other members of the group.
       *
       * @param nb_runs : number of independent simulations
       * @param pop_size : size of the population
       * @param group_size : size of the group
       * @param nb_generations : number of generations per simulation
       * @param nb_games : number of games per generation
       * @param threshold : collective target
       * @param risk : probability of loosing all endowment if the target isn't reached
       * @param transient : number of generations to take into account for calculating the average
       * @param agent_type : string indicating whcoh agent implementation to use
       * @param args : vector of arguments to instantiate the agent_type of the population
       * @return Eigen 2D matrix with the average group achievement and avg. donations across independent runs.
       */
  Matrix2D runConditionalWellMixed(size_t nb_runs, size_t pop_size, size_t group_size, size_t nb_generations,
                                   size_t nb_games, double threshold, double risk, size_t transient,
                                   const std::string &agent_type,
                                   const std::vector<double> &args = {});

  /**
     * @brief Runs simulation with a well mixed population with synchronous updates
     * and returns a data container with the population.
     *
     * In the simulations performed here, agents of a population of size Z = _group_size * nb_groups
     * are selected randomly from the population to form a group of size _group_size and play a game.
     * At each generation @param nb_games are played. The simulation is run for @param nb_generations.
     *
     * The @param args is a vector that should contain the arguments specific of the agent type of
     * the population.
     *
     * This simulation uses agents whose state consists of a tuple \f$(t, d^{t-1}_{-i})\f$,
     * where t is the current round of the game, and \f$d^{t-1}_{-i}\f$ is the sum of donations
     * of the group, without the focal player i, at round t-1. At \f$t=0\f$, \f$d^{t-1}_{-i}\f=0$.
     * In this fashion, agent are able to explore strategy profiles that are conditional to
     * the other members of the group.
     *
     * @param pop_size : size of the population
     * @param group_size : group size
     * @param nb_generations : number of generations per simulation
     * @param nb_games : number of games per generation
     * @param threshold : collective target
     * @param risk : probability of loosing all endowment if the target isn't reached
     * @param agent_type : string indicating which agent implementation to use
     * @param args : vector of arguments to instantiate the agent_type of the population
     * @return a data container that includes the group achievement, the average donations and the population
     */
  DataTypes::CRDData
  runConditionalWellMixedSync(size_t pop_size, size_t group_size, size_t nb_generations, size_t nb_games,
                              double threshold,
                              double risk, const std::string &agent_type,
                              const std::vector<double> &args = {});

  /**
       * @brief Runs several simulations with a well mixed population and synchronous updates.
       *
       * In the simulations performed here, agents of a population of size Z = _group_size * nb_groups
       * are selected randomly from the population to form a group of size _group_size and play a game.
       * At each generation @param nb_games are played. The simulation is run for @param nb_generations.
       *
       * The @param args is a vector that should contain the arguments specific of the agent type of
       * the population.
       *
       * The simulation is repeated for @param nb_runs with independent populations.
       *
       * This simulation uses agents whose state consists of a tuple \f$(t, d^{t-1}_{-i})\f$,
       * where t is the current round of the game, and \f$d^{t-1}_{-i}\f$ is the sum of donations
       * of the group, without the focal player i, at round t-1. At \f$t=0\f$, \f$d^{t-1}_{-i}\f=0$.
       * In this fashion, agent are able to explore strategy profiles that are conditional to
       * the other members of the group.
       *
       * @param nb_runs : number of independent simulations
       * @param pop_size : size of the population
       * @param group_size : size of the group
       * @param nb_generations : number of generations per simulation
       * @param nb_games : number of games per generation
       * @param threshold : collective target
       * @param risk : probability of loosing all endowment if the target isn't reached
       * @param transient : number of generations to take into account for calculating the average
       * @param agent_type : string indicating whcoh agent implementation to use
       * @param args : vector of arguments to instantiate the agent_type of the population
       * @return Eigen 2D matrix with the average group achievement and avg. donations across independent runs.
       */
  Matrix2D runConditionalWellMixedSync(size_t nb_runs, size_t pop_size, size_t group_size, size_t nb_generations,
                                       size_t nb_games, double threshold, double risk, size_t transient,
                                       const std::string &agent_type,
                                       const std::vector<double> &args = {});

  /**
       * @brief trains a well-mixed population in the CRD with Timing uncertainty
       *
       * Here the number of rounds of the game is uncertain. The game will always take at least @param min_rounds
       * and a maximum of @param max_rounds. After @param min_rounds, the game will end with probability @param p.
       *
       * If @param mean_rounds is != 0, @param p will be ignored, instead p = 1 / [(10 - min_rounds) + 1] so that
       * the average of the geometric distribution that defines the final round is @param avg_rounds.
       *
       * In the simulations performed here, agents of a population of size Z = @param pop_size
       * are selected randomly from the population to form a group of size @param group_size and play a game.
       * At each generation @param nb_games are played. The simulation is run for @param nb_generations.
       *
       * The @param args is a vector that should contain the arguments specific of the @param agent_type of
       * the population.
       *
       * The results are transfered to a data container and returned.
       *
       * This simulation uses agents whose state consists of a tuple \f$(t, d^{t-1}_{-i})\f$,
       * where t is the current round of the game, and \f$d^{t-1}_{-i}\f$ is the sum of donations
       * of the group, without the focal player i, at round t-1. At \f$t=0\f$, \f$d^{t-1}_{-i}\f=0$.
       * In this fashion, agent are able to explore strategy profiles that are conditional to
       * the other members of the group.
       *
       * @param pop_size : size of the population
       * @param group_size : group size
       * @param nb_generations : total number of generations
       * @param nb_games : number of games per generation
       * @param threshold : collective target
       * @param risk : risk of losing the remaining endowment if the target isn't reached
       * @param min_rounds : minimum number of rounds
       * @param mean_rounds : mean number of rounds, used for computing the random distribution
       * @param max_rounds : maximum number of rounds
       * @param agent_type : algorithm type of the agent
       * @param args : arguments for the agent
       * @return a data container
       */
  DataTypes::CRDData
  runConditionalWellMixedTU(size_t pop_size, size_t group_size, size_t nb_generations, size_t nb_games,
                            double threshold,
                            double risk, size_t min_rounds, size_t mean_rounds, size_t max_rounds, double p,
                            const std::string &agent_type,
                            const std::vector<double> &args = {});

  /**
       * @brief trains a well-mixed population in the CRD with Timing uncertainty for multiple independent runs.
       *
       * This simulations run the Collective Risk Game with timing uncertainty specified in [Domingos et al. 2019].
       * Here the number of rounds of the game is uncertain. The game will always take at least @param min_rounds
       * and a maximum of @param max_rounds. After @param min_rounds, the game will end with probability @param p.
       *
       * If @param mean_rounds is != 0, @param p will be ignored, instead p = 1 / [(10 - min_rounds) + 1] so that
       * the average of the geometric distribution that defines the final round is @param avg_rounds.
       *
       * In the simulations performed here, agents of a population of size Z = @param pop_size
       * are selected randomly from the population to form a group of size @param group_size and play a game.
       * At each generation @param nb_games are played. The simulation is run for @param nb_generations.
       *
       * This simulation uses agents whose state consists of a tuple \f$(t, d^{t-1}_{-i})\f$,
       * where t is the current round of the game, and \f$d^{t-1}_{-i}\f$ is the sum of donations
       * of the group, without the focal player i, at round t-1. At \f$t=0\f$, \f$d^{t-1}_{-i}\f=0$.
       * In this fashion, agent are able to explore strategy profiles that are conditional to
       * the other members of the group.
       *
       * The @param args is a vector that should contain the arguments specific of the @param agent_type of
       * the population.
       *
       * This method returns a matrix with the average group achievement and contributions over the last
       * \p transient generations of each independent simulation.
       *
       * @param nb_runs : number of independent runs
       * @param pop_size : size of the population
       * @param group_size : group size
       * @param nb_generations : total number of generations
       * @param nb_games : number of games per generation
       * @param threshold : collective target
       * @param risk : risk of losing the remaining endowment if the target isn't reached
       * @param transient : number of generations to take into account for computing the average
       * @param min_rounds : minimum number of rounds
       * @param mean_rounds : mean number of rounds, used for computing the random distribution
       * @param max_rounds : maximum number of rounds
       * @param agent_type : algorithm type of the agent
       * @param args : arguments for the agent
       * @return the average group achievement and donation for each independent run
       */
  Matrix2D
  runConditionalWellMixedTU(size_t nb_runs, size_t pop_size, size_t group_size, size_t nb_generations,
                            size_t nb_games, double threshold,
                            double risk, size_t transient,
                            size_t min_rounds, size_t mean_rounds, size_t max_rounds, double p,
                            const std::string &agent_type,
                            const std::vector<double> &args = {});

  /**
     * @brief trains a well-mixed population in the CRD with Timing uncertainty and synchronous updates.
     *
     * Here the number of rounds of the game is uncertain. The game will always take at least @param min_rounds
     * and a maximum of @param max_rounds. After @param min_rounds, the game will end with probability @param p.
     *
     * If @param mean_rounds is != 0, @param p will be ignored, instead p = 1 / [(10 - min_rounds) + 1] so that
     * the average of the geometric distribution that defines the final round is @param avg_rounds.
     *
     * In the simulations performed here, agents of a population of size Z = @param pop_size
     * are selected randomly from the population to form a group of size @param group_size and play a game.
     * At each generation @param nb_games are played. The simulation is run for @param nb_generations.
     *
     * The @param args is a vector that should contain the arguments specific of the @param agent_type of
     * the population.
     *
     * The results are transfered to a data container and returned.
     *
     * This simulation uses agents whose state consists of a tuple \f$(t, d^{t-1}_{-i})\f$,
     * where t is the current round of the game, and \f$d^{t-1}_{-i}\f$ is the sum of donations
     * of the group, without the focal player i, at round t-1. At \f$t=0\f$, \f$d^{t-1}_{-i}\f=0$.
     * In this fashion, agent are able to explore strategy profiles that are conditional to
     * the other members of the group.
     *
     * @param pop_size : size of the population
     * @param group_size : group size
     * @param nb_generations : total number of generations
     * @param nb_games : number of games per generation
     * @param threshold : collective target
     * @param risk : risk of losing the remaining endowment if the target isn't reached
     * @param min_rounds : minimum number of rounds
     * @param mean_rounds : mean number of rounds, used for computing the random distribution
     * @param max_rounds : maximum number of rounds
     * @param agent_type : algorithm type of the agent
     * @param args : arguments for the agent
     * @return a data container
     */
  DataTypes::CRDData
  runConditionalWellMixedTUSync(size_t pop_size, size_t group_size, size_t nb_generations, size_t nb_games,
                                double threshold,
                                double risk, size_t min_rounds, size_t mean_rounds, size_t max_rounds, double p,
                                const std::string &agent_type,
                                const std::vector<double> &args = {});

  /**
       * @brief trains a well-mixed population in the CRD with Timing uncertainty for multiple independent runs and
       * synchronous updates.
       *
       * This simulations run the Collective Risk Game with timing uncertainty specified in [Domingos et al. 2019].
       * Here the number of rounds of the game is uncertain. The game will always take at least @param min_rounds
       * and a maximum of @param max_rounds. After @param min_rounds, the game will end with probability @param p.
       *
       * If @param mean_rounds is != 0, @param p will be ignored, instead p = 1 / [(10 - min_rounds) + 1] so that
       * the average of the geometric distribution that defines the final round is @param avg_rounds.
       *
       * In the simulations performed here, agents of a population of size Z = @param pop_size
       * are selected randomly from the population to form a group of size @param group_size and play a game.
       * At each generation @param nb_games are played. The simulation is run for @param nb_generations.
       *
       * This simulation uses agents whose state consists of a tuple \f$(t, d^{t-1}_{-i})\f$,
       * where t is the current round of the game, and \f$d^{t-1}_{-i}\f$ is the sum of donations
       * of the group, without the focal player i, at round t-1. At \f$t=0\f$, \f$d^{t-1}_{-i}\f=0$.
       * In this fashion, agent are able to explore strategy profiles that are conditional to
       * the other members of the group.
       *
       * The @param args is a vector that should contain the arguments specific of the @param agent_type of
       * the population.
       *
       * This method returns a matrix with the average group achievement and contributions over the last
       * \p transient generations of each independent simulation.
       *
       * @param nb_runs : number of independent runs
       * @param pop_size : size of the population
       * @param group_size : group size
       * @param nb_generations : total number of generations
       * @param nb_games : number of games per generation
       * @param threshold : collective target
       * @param risk : risk of losing the remaining endowment if the target isn't reached
       * @param transient : number of generations to take into account for computing the average
       * @param min_rounds : minimum number of rounds
       * @param mean_rounds : mean number of rounds, used for computing the random distribution
       * @param max_rounds : maximum number of rounds
       * @param agent_type : algorithm type of the agent
       * @param args : arguments for the agent
       * @return the average group achievement and donation for each independent run
       */
  Matrix2D
  runConditionalWellMixedTUSync(size_t nb_runs, size_t pop_size, size_t group_size, size_t nb_generations,
                                size_t nb_games, double threshold,
                                double risk, size_t transient,
                                size_t min_rounds, size_t mean_rounds, size_t max_rounds, double p,
                                const std::string &agent_type,
                                const std::vector<double> &args = {});

  /**
       * @brief Runs a simulation with threshold uncertainty and conditional agents.
       *
       * This simulations run the Collective Risk Game with threshold uncertainty.
       * Here the threshold is uncertain. The threshold of the game is a stochastic uniform
       * variable in the range \f$(threshold - \delta/2, threshold + \delta/2)\f$ .
       *
       * Parameter \p delta must be divisible by 2! Otherwise the simulation will not produce
       * correct results.
       *
       * This simulation uses agents whose state consists of a tuple \f$(t, d^{t-1}_{-i})\f$,
       * where t is the current round of the game, and \f$d^{t-1}_{-i}\f$ is the sum of donations
       * of the group, without the focal player i, at round t-1. At \f$t=0\f$, \f$d^{t-1}_{-i}\f=0$.
       * In this fashion, agent are able to explore strategy profiles that are conditional to
       * the other members of the group.
       *
       * In the simulations performed here, agents of a population of size Z = @param pop_size
       * are selected randomly from the population to form a group of size @param group_size and play a game.
       * At each generation @param nb_games are played. The simulation is run for @param nb_generations.
       *
       * @param pop_size : size of the population
       * @param group_size : group size
       * @param nb_generations : number of training time-steps
       * @param nb_games : number of games per time-step / generation
       * @param threshold : target of the game
       * @param risk : probability of loosing all endowment if the target isn't reached
       * @param delta : variance / uncertainty of the threshold. Must be divisible by 2!
       * @param agent_type : string indicating the learning algorithm that the agents will use
       * @param args : specific parameters for the learning algorithm
       * @return a data container with the information of the simulation and a pointer to the population.
       */
  DataTypes::CRDData
  runConditionalWellMixedThU(size_t pop_size, size_t group_size, size_t nb_generations, size_t nb_games,
                             size_t threshold, size_t delta, double risk,
                             const std::string &agent_type,
                             const std::vector<double> &args = {});

  /**
       * @brief Runs a simulation with threshold uncertainty and conditional agents.
       *
       * This simulations run the Collective Risk Game with threshold uncertainty.
       * Here the threshold is uncertain. The threshold of the game is a stochastic uniform
       * variable in the range \f$(threshold - delta/2, threshold + delta/2)\f$ .
       *
       * Parameter \p delta must be divisible by 2! Otherwise the simulation will not produce
       * correct results.
       *
       * This simulation uses agents whose state consists of a tuple \f$(t, d^{t-1}_{-i})\f$,
       * where t is the current round of the game, and \f$d^{t-1}_{-i}\f$ is the sum of donations
       * of the group, without the focal player i, at round t-1. At \f$t=0\f$, \f$d^{t-1}_{-i}\f=0$.
       * In this fashion, agent are able to explore strategy profiles that are conditional to
       * the other members of the group.
       *
       * In the simulations performed here, agents of a population of size Z = @param pop_size
       * are selected randomly from the population to form a group of size @param group_size and play a game.
       * At each generation @param nb_games are played. The simulation is run for @param nb_generations.
       *
       * @param nb_runs : number of independent simulations
       * @param pop_size : size of the population
       * @param group_size : group size
       * @param nb_generations : number of training time-steps
       * @param nb_games : number of games per time-step / generation
       * @param threshold : target of the game
       * @param risk : probability of loosing all endowment if the target isn't reached
       * @param delta : variance / uncertainty of the threshold
       * @param agent_type : string indicating the learning algorithm that the agents will use
       * @param args : specific parameters for the learning algorithm
       * @return : a matrix indicating the averaged group achievement and agent contributions
       *           over the last \p transient generations of each simulation
       */
  Matrix2D runConditionalWellMixedThU(size_t nb_runs,
                                      size_t pop_size,
                                      size_t group_size,
                                      size_t nb_generations,
                                      size_t nb_games,
                                      size_t threshold,
                                      size_t delta,
                                      double risk,
                                      size_t transient,
                                      const std::string &agent_type,
                                      const std::vector<double> &args = {});

  /**
     * @brief Runs a simulation with threshold uncertainty and conditional agents with synchronous updates.
     *
     * This simulations run the Collective Risk Game with threshold uncertainty.
     * Here the threshold is uncertain. The threshold of the game is a stochastic uniform
     * variable in the range \f$(threshold - delta/2, threshold + delta/2)\f$ .
     *
     * This simulation uses agents whose state consists of a tuple \f$(t, d^{t-1}_{-i})\f$,
     * where t is the current round of the game, and \f$d^{t-1}_{-i}\f$ is the sum of donations
     * of the group, without the focal player i, at round t-1. At \f$t=0\f$, \f$d^{t-1}_{-i}\f=0$.
     * In this fashion, agent are able to explore strategy profiles that are conditional to
     * the other members of the group.
     *
     * In the simulations performed here, agents of a population of size Z = @param pop_size
     * are selected randomly from the population to form a group of size @param group_size and play a game.
     * At each generation @param nb_games are played. The simulation is run for @param nb_generations.
     *
     * @param pop_size : size of the population
     * @param group_size : group size
     * @param nb_generations : number of training time-steps
     * @param nb_games : number of games per time-step / generation
     * @param threshold : target of the game
     * @param risk : probability of loosing all endowment if the target isn't reached
     * @param delta : variance / uncertainty of the threshold
     * @param agent_type : string indicating the learning algorithm that the agents will use
     * @param args : specific parameters for the learning algorithm
     * @return a data container with the information of the simulation and a pointer to the population.
     */
  DataTypes::CRDData
  runConditionalWellMixedThUSync(size_t pop_size, size_t group_size, size_t nb_generations, size_t nb_games,
                                 size_t threshold, size_t delta, double risk,
                                 const std::string &agent_type,
                                 const std::vector<double> &args = {});

  /**
       * @brief Runs multiple simulations with threshold uncertainty and conditional agents with synchronous updates.
       *
       * This simulations run the Collective Risk Game with threshold uncertainty.
       * Here the threshold is uncertain. The threshold of the game is a stochastic uniform
       * variable in the range \f$(threshold - delta/2, threshold + delta/2)\f$ .
       *
       * This simulation uses agents whose state consists of a tuple \f$(t, d^{t-1}_{-i})\f$,
       * where t is the current round of the game, and \f$d^{t-1}_{-i}\f$ is the sum of donations
       * of the group, without the focal player i, at round t-1. At \f$t=0\f$, \f$d^{t-1}_{-i}\f=0$.
       * In this fashion, agent are able to explore strategy profiles that are conditional to
       * the other members of the group.
       *
       * In the simulations performed here, agents of a population of size Z = @param pop_size
       * are selected randomly from the population to form a group of size @param group_size and play a game.
       * At each generation @param nb_games are played. The simulation is run for @param nb_generations.
       *
       * @param nb_runs : number of independent simulations
       * @param pop_size : size of the population
       * @param group_size : group size
       * @param nb_generations : number of training time-steps
       * @param nb_games : number of games per time-step / generation
       * @param threshold : target of the game
       * @param risk : probability of loosing all endowment if the target isn't reached
       * @param delta : variance / uncertainty of the threshold
       * @param agent_type : string indicating the learning algorithm that the agents will use
       * @param args : specific parameters for the learning algorithm
       * @return : a matrix indicating the averaged group achievement and agent contributions
       *           over the last \p transient generations of each simulation
       */
  Matrix2D runConditionalWellMixedThUSync(size_t nb_runs,
                                          size_t pop_size,
                                          size_t group_size,
                                          size_t nb_generations,
                                          size_t nb_games,
                                          size_t threshold,
                                          size_t delta,
                                          double risk,
                                          size_t transient,
                                          const std::string &agent_type,
                                          const std::vector<double> &args = {});

  /**
   * @brief Runs several independent simulations with both Timing uncertainty and Threshold uncertainty
   *        with conditional agents.
   *
   * This method permits the exploration of both uncertainties at the same time.
   *
   * This simulation uses agents whose state consists of a tuple \f$(t, d^{t-1}_{-i})\f$,
   * where t is the current round of the game, and \f$d^{t-1}_{-i}\f$ is the sum of donations
   * of the group, without the focal player i, at round t-1. At \f$t=0\f$, \f$d^{t-1}_{-i}\f=0$.
   * In this fashion, agent are able to explore strategy profiles that are conditional to
   * the other members of the group.
   *
   * In the simulations performed here, agents of a population of size Z = @param pop_size
   * are selected randomly from the population to form a group of size @param group_size and play a game.
   * At each generation @param nb_games are played. The simulation is run for @param nb_generations.
   *
   * Here the threshold is uncertain. The threshold of the game is a stochastic uniform
   * variable in the range \f$(threshold - delta/2, threshold + delta/2)\f$ .
   *
   * Here the number of rounds of the game is uncertain. The game will always take at least @param min_rounds
   * and a maximum of @param max_rounds. After @param min_rounds, the game will end with probability @param p.
   *
   * If @param mean_rounds is != 0, @param p will be ignored, instead p = 1 / [(10 - min_rounds) + 1] so that
   * the average of the geometric distribution that defines the final round is @param avg_rounds.
   *
   * The results are returned in a data container.
   *
   * @param pop_size : size of the population
   * @param group_size : size of the group that plays the game
   * @param nb_generations : number of generations per simulation
   * @param threshold : threshold of the game
   * @param delta : threshold uncertainty (variation of the threshold)
   * @param risk : risk (impact uncertainty) of the game
   * @param min_rounds : minimum number of rounds in a game
   * @param mean_rounds : mean number of rounds in a game
   * @param max_rounds : maximum number of rounds per game
   * @param p : probability of the game ending after the minimum number of rounds
   * @param agent_type : algorithm used by each agent to adapt in the population
   * @param args : arguments of the algorithm
   * @return a matrix containing the averaged group_achievement and player's contributions over the simulations.
   */
  DataTypes::CRDData
  runConditionalWellMixedTUnThU(size_t pop_size, size_t group_size, size_t nb_generations, size_t nb_games,
                                size_t threshold, size_t delta, double risk, size_t min_rounds, size_t mean_rounds,
                                size_t max_rounds, double p,
                                const std::string &agent_type,
                                const std::vector<double> &args = {});

  /**
   * @brief Runs several independent simulations with both Timing uncertainty and Threshold uncertainty
   *        with conditional agents.
   *
   * This method permits the exploration of both uncertainties at the same time.
   *
   * This simulation uses agents whose state consists of a tuple \f$(t, d^{t-1}_{-i})\f$,
   * where t is the current round of the game, and \f$d^{t-1}_{-i}\f$ is the sum of donations
   * of the group, without the focal player i, at round t-1. At \f$t=0\f$, \f$d^{t-1}_{-i}\f=0$.
   * In this fashion, agent are able to explore strategy profiles that are conditional to
   * the other members of the group.
   *
   * In the simulations performed here, agents of a population of size Z = @param pop_size
   * are selected randomly from the population to form a group of size @param group_size and play a game.
   * At each generation @param nb_games are played. The simulation is run for @param nb_generations.
   *
   * Here the threshold is uncertain. The threshold of the game is a stochastic uniform
   * variable in the range \f$(threshold - delta/2, threshold + delta/2)\f$ .
   *
   * Here the number of rounds of the game is uncertain. The game will always take at least @param min_rounds
   * and a maximum of @param max_rounds. After @param min_rounds, the game will end with probability @param p.
   *
   * If @param mean_rounds is != 0, @param p will be ignored, instead p = 1 / [(10 - min_rounds) + 1] so that
   * the average of the geometric distribution that defines the final round is @param avg_rounds.
   *
   * This method returns a matrix with the average group achievement and contributions over the last
   * \p transient generations of each independent simulation.
   *
   * @param nb_runs : number of independent simulations to run
   * @param pop_size : size of the population
   * @param group_size : size of the group that plays the game
   * @param nb_generations : number of generations per simulation
   * @param threshold : threshold of the game
   * @param delta : threshold uncertainty (variation of the threshold)
   * @param risk : risk (impact uncertainty) of the game
   * @param min_rounds : minimum number of rounds in a game
   * @param mean_rounds : mean number of rounds in a game
   * @param max_rounds : maximum number of rounds per game
   * @param p : probability of the game ending after the minimum number of rounds
   * @param agent_type : algorithm used by each agent to adapt in the population
   * @param args : arguments of the algorithm
   * @return a matrix containing the averaged group_achievement and player's contributions over the simulations.
   */
  Matrix2D
  runConditionalWellMixedTUnThU(size_t nb_runs,
                                size_t pop_size,
                                size_t group_size,
                                size_t nb_generations,
                                size_t nb_games,
                                size_t threshold,
                                size_t delta,
                                double risk,
                                size_t transient,
                                size_t min_rounds,
                                size_t mean_rounds,
                                size_t max_rounds,
                                double p,
                                const std::string &agent_type,
                                const std::vector<double> &args = {});

  /**
       * @brief resets the population stored in the class.
       */
  void resetPopulation();

  /**
       * @brief This method reinforces agents only when the target is reached
       * @tparam G : Game class
       * @param pool : crd group with agents
       * @param success : if the group reached the target
       * @param risk : probability of losing all endowment if the target isn't reached
       * @param pop : population container
       * @param game : game object
       */
  template<class G = CRDGame<PopContainer, void, void>>
  void reinforceOnlyPositive(double &pool, size_t &success, double &risk, PopContainer &pop,
                             G &game);

  /**
       * @brief This method reinforces agents only when the target is reached (for Timing uncertainty games)
       * @tparam G : Game class
       * @param pool : crd group with agents
       * @param success : if the group reached the target
       * @param risk : probability of losing all endowment if the target isn't reached
       * @param pop : population container
       * @param final_round : number of rounds of the last played game.
       * @param game : game object
       */
  template<class G = CRDGame<PopContainer, void, void>>
  void reinforceOnlyPositive(double &pool, size_t &success, double &risk, PopContainer &pop, size_t &final_round,
                             G &game);

  /**
       * @brief This method reinforces agents proportionally to the obtained payoff.
       * @tparam G : Game class
       * @param pool : crd group with agents
       * @param success : if the group reached the target
       * @param risk : probability of losing all endowment if the target isn't reached
       * @param pop : population container
       * @param game : game object
       */
  template<class G = CRDGame<PopContainer, void, void>>
  void reinforceAll(double &pool, size_t &success, double &risk, PopContainer &pop,
                    G &game);

  /**
       * @brief This method reinforces agents proportionally to the obtained payoff for Timing uncertainty games.
       * @tparam G : Game class
       * @param pool : crd group with agents
       * @param success : if the group reached the target
       * @param risk : probability of losing all endowment if the target isn't reached
       * @param pop : population container
       * @param final_round : number of rounds of the last played game.
       * @param game : game object
       */
  template<class G = CRDGame<PopContainer, TimingUncertainty<std::mt19937_64>, std::mt19937_64>>
  void reinforceAll(double &pool, size_t &success, double &risk, PopContainer &pop, size_t &final_round,
                    G &game);

  template<class G = CRDGame<PopContainer, void, void>>
  void reinforceAll(double &pool, size_t &success, double threshold, double &risk, PopContainer &pop,
                    G &game, std::mt19937_64 &generator);

  /**
   * @brief This method reinforces agents proportionally to the obtained payoff for Timing uncertainty games.
   * @tparam G
   * @param pool
   * @param success
   * @param threshold
   * @param risk
   * @param pop
   * @param final_round
   * @param game
   * @param generator
   */
  template<class G = CRDGame<PopContainer, TimingUncertainty<std::mt19937_64>, std::mt19937_64>>
  void reinforceAll(double &pool, size_t &success, double threshold, double &risk, PopContainer &pop,
                    size_t &final_round,
                    G &game, std::mt19937_64 &generator);

  template<class G = CRDGame<PopContainer, void, void>>
  void reinforceOnePlayer(double &pool, size_t &success, double threshold, double &risk,
                          EGTTools::RL::Individual &player, std::mt19937_64 &generator);

  /**
   * @brief This method reinforces agents proportionally to the obtained payoff for Timing uncertainty games.
   * @tparam G
   * @param pool
   * @param success
   * @param threshold
   * @param risk
   * @param pop
   * @param final_round
   * @param game
   * @param generator
   */
  template<class G = CRDGame<PopContainer, TimingUncertainty<std::mt19937_64>, std::mt19937_64>>
  void reinforceOnePlayer(double &pool, size_t &success, double threshold, double &risk,
                          size_t &final_round, EGTTools::RL::Individual &player, std::mt19937_64 &generator);

  /**
       * @brief This method reinforces following Xico's version of the CRD payoffs
       * @tparam G : Game class
       * @param pool : crd group with agents
       * @param success :: if the group reached the target
       * @param risk : probability of losing all endowment if the target isn't reached
       * @param pop : population container
       * @param game : game object
       */
  template<class G = CRDGame<PopContainer, void, void>>
  void reinforceXico(double &pool, size_t &success, double &risk, PopContainer &pop,
                     G &game);

  /**
       * @brief This method reinforces following Xico's version of the CRD payoffs (for Timing uncertainty games).
       * @tparam G : Game class
       * @param pool : crd group with agents
       * @param success :: if the group reached the target
       * @param risk : probability of losing all endowment if the target isn't reached
       * @param pop : population container
       * @param final_round : number of rounds of the last played game.
       * @param game : game object
       */
  template<class G = CRDGame<PopContainer, TimingUncertainty<std::mt19937_64>, std::mt19937_64>>
  void reinforceXico(double &pool, size_t &success, double &risk, PopContainer &pop, size_t &final_round,
                     G &game);

  /**
       * @brief sets the game type to Milinski or Xico
       *
       * If the game type is milinski, the minimum payoff is 0.
       * If it is xico, the minimum payoff is -c*b, i.e., if
       * the group doesn't reach the target, the payoff is
       * - the amount contributed by the player.
       *
       * @param crd_type "milinski" or "xico".
       */
  void setGameType(const std::string &crd_type = "milinski");

  // Getters

  [[nodiscard]] size_t nb_games() const;

  [[nodiscard]] size_t nb_episodes() const;

  [[nodiscard]] size_t nb_rounds() const;

  [[nodiscard]] size_t nb_actions() const;

  [[nodiscard]] double endowment() const;

  [[nodiscard]] double risk() const;

  [[nodiscard]] double threshold() const;

  [[nodiscard]] const ActionSpace &available_actions() const;

  [[nodiscard]] const std::string &agent_type() const;

  // Setters

  void set_nb_games(size_t nb_games);

  void set_nb_episodes(size_t nb_episodes);

  void set_nb_rounds(size_t nb_rounds);

  void set_endowment(double endowment);

  void set_nb_actions(size_t nb_actions);

  void set_risk(double risk);

  void set_threshold(double threshold);

  void set_available_actions(const ActionSpace &available_actions);

  void set_agent_type(const std::string &agent_type);

  CRDGame<PopContainer, void, void> Game;
  PopContainer population;

 private:
  size_t _nb_episodes, _nb_games, _nb_rounds, _nb_actions, _group_size;
  double _risk, _endowment, _threshold;
  ActionSpace _available_actions;
  std::string _agent_type;

  std::uniform_real_distribution<double> _real_rand;

  void (EGTTools::RL::CRDSim::*_reinforce)(double &, size_t &, double &, PopContainer &,
                                           CRDGame<PopContainer, void, void> &) = nullptr;

  // Random generators
  std::mt19937_64 _generator{EGTTools::Random::SeedGenerator::getInstance().getSeed()};
};

} // namespace EGTTools::RL

#endif //DYRWIN_RL_CRDSIM_HPP
