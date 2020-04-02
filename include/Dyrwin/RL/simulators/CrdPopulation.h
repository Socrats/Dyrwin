//
// Created by Elias Fernandez on 08/02/2020.
//

#ifndef DYRWIN_RL_SIMULATORS_CRDPOPULATION_H_
#define DYRWIN_RL_SIMULATORS_CRDPOPULATION_H_

#include <random>
#include <unordered_set>
#include <Dyrwin/RL/CRDGame.h>
#include <Dyrwin/RL/CRDConditionalCount.h>
#include <Dyrwin/RL/PopContainer.hpp>
#include <Dyrwin/RL/Utils.h>
#include <Dyrwin/RL/Data.hpp>
#include <Dyrwin/Sampling.h>
#include <Dyrwin/Distributions.h>
#include <Dyrwin/SeedGenerator.h>
#include <Dyrwin/OpenMPUtils.hpp>

namespace EGTTools::RL::Simulators::CRD {

/**
 * This namespace contains functions, classes & methods used to
 * simulate population-based RL models in the context of the CRD Dilemma.
 *
 * Particularly, here we group simulations where learning and evaluation
 * are performed in different populations, i.e., agents learn/adapt
 * in their own group/population, and are evaluated in another group/population.
 *
 * This means that after an adaptation step/learning, groups/populations are mixed
 * and random groups are selected to play the CRD. The actual group achievement
 * is obtained from this step.
 */
class CRDSimPopulation {
 public:

  CRDSimPopulation();

  /**
   * @brief Runs a CRD game for \param nb_generations over a population of agents.
   *
   * It also calculates the average success rate (group achievement)
   * and average contributions per game and generation. This is returned
   * as matrix. This value can then be averaged over different runs.
   *
   * Another option would not to calculate these values, but rather keep only
   * and average over the last X generations. This could be an issue for debugging...
   *
   * There are then 3 options:
   * a) We do not return anything, and all that matters is the evaluation phase.
   * b) We return only the last averaged group achievement and contribution over the last
   *    X rounds
   * c) We return the whole set of control data for each generation
   *
   * We might also want to increase the control set over learning to identify attractors:
   * - For instance it might be interesting to collect information about the polarization
   *   of the population, such as the percentage of players contributing C > F, C = F,
   *   C < F.
   *
   * I think it might be best to separate these functions, i.e., we minimize the computations
   * here, and we create extra versions that return them.
   *
   * @tparam G : game type
   * @param population_size : size of the population
   * @param group_size : size of a group
   * @param nb_generations : number of generations through which the agents adapt
   * @param nb_games : number of games per generation
   * @param nb_rounds : number of rounds per game
   * @param target : collective target
   * @param risk : risk of collective loss
   * @param available_actions : std::vector indicating which actions the agents may take per round
   * @param population : PopContainer object containing the population
   * @param game : CRD game object
   */
  template<class G>
  void run_crd_population(size_t population_size,
                          size_t group_size,
                          size_t nb_generations,
                          size_t nb_games,
                          size_t nb_rounds,
                          int target,
                          double risk,
                          ActionSpace &available_actions,
                          PopContainer &population,
                          G &game);

  template<class G, class U>
  void run_crd_populationTU(size_t population_size,
                            size_t group_size,
                            size_t nb_generations,
                            size_t nb_games,
                            size_t min_rounds,
                            U timing_uncertainty,
                            int target,
                            double risk,
                            ActionSpace &available_actions,
                            PopContainer &population,
                            G &game);

  template<class G>
  void run_crd_populationThU(size_t population_size,
                             size_t group_size,
                             size_t nb_generations,
                             size_t nb_games,
                             size_t nb_rounds,
                             int target,
                             int delta,
                             double risk,
                             ActionSpace &available_actions,
                             PopContainer &population,
                             G &game);

  template<class G, class U>
  void run_crd_populationTUThU(size_t population_size,
                               size_t group_size,
                               size_t nb_generations,
                               size_t nb_games,
                               size_t min_rounds,
                               U timing_uncertainty,
                               int target,
                               int delta,
                               double risk,
                               ActionSpace &available_actions,
                               PopContainer &population,
                               G &game);

  // Simulation methods

  /**
   * @brief Runs \param nb_populations independent population based CRD simulations.
   *
   * Agents learn/adapt by playing the CRD against their own population, but are evaluated
   * by playing random games against another population.
   *
   * @param nb_populations : number of independent populations to adapt.
   * @param population_size : size of the population.
   * @param group_size : size of the group.
   * @param nb_generations : number of generations in which agents adapt.
   * @param nb_games : number of games per generation.
   * @param nb_rounds : number of rounds per game.
   * @param target : collective target.
   * @param endowment : initial private endowment.
   * @param risk : probability of collective loss if the target isn't met.
   * @param available_actions : possible actions per round.
   * @param agent_type : string indicating the algorithm used by the agent to adapt.
   * @param args : arguments to initialize the agent.
   * @return : a data structure that includes the adapted population and the average group achievement and donations.
   */
  DataTypes::DataTableCRD
  run_population_islands(size_t nb_evaluation_games,
                         size_t nb_populations,
                         size_t population_size,
                         size_t group_size,
                         size_t nb_generations,
                         size_t nb_games,
                         size_t nb_rounds,
                         int target,
                         int endowment,
                         double risk,
                         ActionSpace &available_actions,
                         const std::string &agent_type,
                         const std::vector<double> &args = {});

  DataTypes::DataTableCRD
  run_population_islandsTU(size_t nb_evaluation_games,
                           size_t nb_populations,
                           size_t population_size,
                           size_t group_size,
                           size_t nb_generations,
                           size_t nb_games,
                           size_t min_rounds,
                           size_t mean_rounds,
                           size_t max_rounds,
                           double p,
                           int target,
                           int endowment,
                           double risk,
                           ActionSpace &available_actions,
                           const std::string &agent_type,
                           const std::vector<double> &args = {});

  DataTypes::DataTableCRD
  run_population_islandsThU(size_t nb_evaluation_games,
                            size_t nb_populations,
                            size_t population_size,
                            size_t group_size,
                            size_t nb_generations,
                            size_t nb_games,
                            size_t nb_rounds,
                            int target,
                            int delta,
                            int endowment,
                            double risk,
                            ActionSpace &available_actions,
                            const std::string &agent_type,
                            const std::vector<double> &args = {});

  DataTypes::DataTableCRD
  run_population_islandsTUThU(size_t nb_evaluation_games,
                              size_t nb_populations,
                              size_t population_size,
                              size_t group_size,
                              size_t nb_generations,
                              size_t nb_games,
                              size_t min_rounds,
                              size_t mean_rounds,
                              size_t max_rounds,
                              double p,
                              int target,
                              int delta,
                              int endowment,
                              double risk,
                              ActionSpace &available_actions,
                              const std::string &agent_type,
                              const std::vector<double> &args = {});

  /**
   * @brief This method reinforces agents proportionally to the obtained payoff for Timing uncertainty games.
   * @tparam G : game type
   * @param pool : reference to the value contained in the public pool
   * @param success : reference to the variable that indicates if a group me the target
   * @param target : public target
   * @param risk : probability of disaster if the target isn't met
   * @param pop : population container
   * @param final_round : final round of the game
   * @param game : game object
   * @param generator : reference to a random generator
   */
  template<class G = CRDGame<PopContainer, TimingUncertainty<std::mt19937_64>, std::mt19937_64>>
  void reinforce_population(double &pool,
                            size_t &success,
                            double target,
                            double &risk,
                            PopContainer &pop,
                            size_t &final_round,
                            G &game,
                            std::mt19937_64 &generator);

  /**
 * @brief This method reinforces agents proportionally to the obtained payoff for Timing uncertainty games.
 * @tparam G : game type
 * @param pool : reference to the value contained in the public pool
 * @param success : reference to the variable that indicates if a group me the target
 * @param target : public target
 * @param risk : probability of disaster if the target isn't met
 * @param pop : population container
 * @param final_round : final round of the game
 * @param game : game object
 * @param generator : reference to a random generator
 */
  template<class G = CRDGame<PopContainer, void, void>>
  void reinforce_population(double &pool,
                            size_t &success,
                            double threshold,
                            double &risk,
                            PopContainer &pop,
                            G &game,
                            std::mt19937_64 &generator);

 private:
  std::uniform_real_distribution<double> _real_rand;
};
template<class G>
void CRDSimPopulation::reinforce_population(double &pool,
                                         size_t &success,
                                         double target,
                                         double &risk,
                                         PopContainer &pop,
                                         size_t &final_round,
                                         G &game,
                                         std::mt19937_64 &generator) {
  if (pool >= target)
    success++;
  else if (_real_rand(generator) < risk)
    game.setPayoffs(pop, 0);

  game.reinforcePath(pop, final_round);
}
template<class G>
void CRDSimPopulation::reinforce_population(double &pool,
                                         size_t &success,
                                         double threshold,
                                         double &risk,
                                         EGTTools::RL::PopContainer &pop,
                                         G &game,
                                         std::mt19937_64 &generator) {
  if (pool >= threshold)
    success++;
  else if (_real_rand(generator) < risk)
    game.setPayoffs(pop, 0);

  game.reinforcePath(pop);
}
template<class G>
void CRDSimPopulation::run_crd_population(size_t population_size,
                                       size_t group_size,
                                       size_t nb_generations,
                                       size_t nb_games,
                                       size_t nb_rounds,
                                       int target,
                                       double risk,
                                       ActionSpace &available_actions,
                                       EGTTools::RL::PopContainer &population,
                                       G &game) {
  size_t success = 0;
  std::mt19937_64 generator(EGTTools::Random::SeedGenerator::getInstance().getSeed());

  // Prepare group container
  PopContainer group;
  std::unordered_set<size_t> container;
  container.reserve(group_size);
  for (size_t i = 0; i < group_size; ++i)
    group.push_back(population(i));

  for (size_t generation = 0; generation < nb_generations; ++generation) {
    // First we select random groups and let them play nb_games
    success = 0;
    for (size_t i = 0; i < nb_games; ++i) {
      EGTTools::sampling::sample_without_replacement(population_size, group_size, container, generator);
      int j = 0;
      // Sample group
      for (const auto &elem: container) {
        group(j) = population(elem);
        j++;
      }
      // First we play the game
      auto[pool, final_round] = game.playGame(group, available_actions, nb_rounds);
      reinforce_population(pool, success, target, risk, group, game, generator);
      container.clear();
    }

    game.calcProbabilities(population);
    game.resetEpisode(population);
  }

}
template<class G, class U>
void CRDSimPopulation::run_crd_populationTU(size_t population_size,
                                         size_t group_size,
                                         size_t nb_generations,
                                         size_t nb_games,
                                         size_t min_rounds,
                                         U timing_uncertainty,
                                         int target,
                                         double risk,
                                         ActionSpace &available_actions,
                                         PopContainer &population,
                                         G &game) {
  size_t success = 0;
  std::mt19937_64 generator(EGTTools::Random::SeedGenerator::getInstance().getSeed());

  // Prepare group container
  PopContainer group;
  std::unordered_set<size_t> container;
  container.reserve(group_size);
  for (size_t i = 0; i < group_size; ++i)
    group.push_back(population(i));

  for (size_t generation = 0; generation < nb_generations; ++generation) {
    success = 0;
    for (size_t i = 0; i < nb_games; ++i) {
      EGTTools::sampling::sample_without_replacement(population_size, group_size, container, generator);
      int j = 0;
      // Sample group
      for (const auto &elem: container) {
        group(j) = population(elem);
        j++;
      }
      // First we play several games
      auto[pool, final_round] = game.playGame(group, available_actions, min_rounds, timing_uncertainty, generator);
      reinforce_population(pool, success, target, risk, group, final_round, game, generator);
      container.clear();
    }
    // Then, update the population
    game.calcProbabilities(population);
    game.resetEpisode(population);
  }

  game.calcProbabilities(population);
  game.resetEpisode(population);
}
template<class G>
void CRDSimPopulation::run_crd_populationThU(size_t population_size,
                                          size_t group_size,
                                          size_t nb_generations,
                                          size_t nb_games,
                                          size_t nb_rounds,
                                          int target,
                                          int delta,
                                          double risk,
                                          ActionSpace &available_actions,
                                          PopContainer &population,
                                          G &game) {
  size_t success = 0;
  std::mt19937_64 generator(EGTTools::Random::SeedGenerator::getInstance().getSeed());

  // Define the distribution for the threshold
  std::uniform_int_distribution<int> t_dist(target - delta / 2, target + delta / 2);

  // Prepare group container
  PopContainer group;
  std::unordered_set<size_t> container;
  container.reserve(group_size);
  for (size_t i = 0; i < group_size; ++i)
    group.push_back(population(i));

  for (size_t generation = 0; generation < nb_generations; ++generation) {
    // First we select random groups and let them play nb_games
    success = 0;
    for (size_t i = 0; i < nb_games; ++i) {
      EGTTools::sampling::sample_without_replacement(population_size, group_size, container, generator);
      int j = 0;
      // Sample group
      for (const auto &elem: container) {
        group(j) = population(elem);
        j++;
      }
      // First we play the game
      auto[pool, final_round] = game.playGame(group, available_actions, nb_rounds);
      reinforce_population(pool, success, t_dist(generator), risk, group, game, generator);
      container.clear();
    }

    game.calcProbabilities(population);
    game.resetEpisode(population);
  }

}
template<class G, class U>
void CRDSimPopulation::run_crd_populationTUThU(size_t population_size,
                                            size_t group_size,
                                            size_t nb_generations,
                                            size_t nb_games,
                                            size_t min_rounds,
                                            U timing_uncertainty,
                                            int target,
                                            int delta,
                                            double risk,
                                            ActionSpace &available_actions,
                                            PopContainer &population,
                                            G &game) {
  size_t success = 0;
  std::mt19937_64 generator(EGTTools::Random::SeedGenerator::getInstance().getSeed());

  // Define the distribution for the threshold
  std::uniform_int_distribution<int> t_dist(target - delta / 2, target + delta / 2);

  // Prepare group container
  PopContainer group;
  std::unordered_set<size_t> container;
  container.reserve(group_size);
  for (size_t i = 0; i < group_size; ++i)
    group.push_back(population(i));

  for (size_t generation = 0; generation < nb_generations; ++generation) {
    success = 0;
    for (size_t i = 0; i < nb_games; ++i) {
      EGTTools::sampling::sample_without_replacement(population_size, group_size, container, generator);
      int j = 0;
      // Sample group
      for (const auto &elem: container) {
        group(j) = population(elem);
        j++;
      }
      // First we play several games
      auto[pool, final_round] = game.playGame(group, available_actions, min_rounds, timing_uncertainty, generator);
      reinforce_population(pool, success, t_dist(generator), risk, group, final_round, game, generator);
      container.clear();
    }
    // Then, update the population
    game.calcProbabilities(population);
    game.resetEpisode(population);
  }

  game.calcProbabilities(population);
  game.resetEpisode(population);
}
} // namespace EGTTools::RL::Simulators

#endif //DYRWIN_RL_SIMULATORS_CRDPOPULATION_H_
