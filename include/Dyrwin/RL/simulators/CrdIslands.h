//
// Created by Elias Fernandez on 08/02/2020.
//

#ifndef DYRWIN_RL_SIMULATORS_CRDISLANDS_H_
#define DYRWIN_RL_SIMULATORS_CRDISLANDS_H_

#include <random>
#include <unordered_set>
#include <Dyrwin/RL/CRDGame.h>
#include <Dyrwin/RL/CRDConditionalCount.h>
#include <Dyrwin/RL/PopContainer.hpp>
#include <Dyrwin/RL/Utils.h>
#include <Dyrwin/RL/Data.hpp>
#include <Dyrwin/RL/helpers/PopulationHelper.h>
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
class CRDSimIslands {
 public:

  CRDSimIslands();

  // Helper methods

  /**
   * @brief Runs a CRD game for \param nb_generations and a single group.
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
   * @tparam G : game class
   * @param nb_generations : number of generations on which the group adapts.
   * @param nb_games : number of games per generation.
   * @param nb_rounds : number of rounds per game.
   * @param target : collective target.
   * @param endowment : private good.
   * @param risk : probability of losing the remaining endowment if the target isn't met.
   * @param population : PopContainer object containing the players.
   */
  template<class G>
  void run_crd_single_group(size_t nb_generations,
                            size_t nb_games,
                            size_t nb_rounds,
                            int target,
                            double risk,
                            ActionSpace &available_actions,
                            PopContainer &population,
                            G &game);

  template<class G, class U>
  void run_crd_single_groupTU(size_t nb_generations,
                              size_t nb_games,
                              size_t min_rounds,
                              U timing_uncertainty,
                              int target,
                              double risk,
                              ActionSpace &available_actions,
                              PopContainer &population,
                              G &game);

  template<class G>
  void run_crd_single_groupThU(size_t nb_generations,
                               size_t nb_games,
                               size_t nb_rounds,
                               int target,
                               double delta,
                               double risk,
                               ActionSpace &available_actions,
                               PopContainer &population,
                               G &game);

  template<class G, class U>
  void run_crd_single_groupTUThU(size_t nb_generations,
                                 size_t nb_games,
                                 size_t min_rounds,
                                 U timing_uncertainty,
                                 int target,
                                 double delta,
                                 double risk,
                                 ActionSpace &available_actions,
                                 PopContainer &population,
                                 G &game);

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
                             double delta,
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
                               double delta,
                               double risk,
                               ActionSpace &available_actions,
                               PopContainer &population,
                               G &game);

  /**
   * @brief Evaluates behaviors from several population sin the CRD.
   *
   * Each game is played by a random subset of the populations. The subset
   * is sample uniformly randomly from all the populations (without replacement),
   * i.e., an agent from any population, has the same probability to be in a group.
   *
   * This method assumes that all populations have the same size!
   * \param population_size must never be 0! However, no checks are implemented!
   *
   * @tparam G : game type.
   * @param nb_populations : number of independent populations.
   * @param population_size : size of the population.
   * @param group_size : size of the group.
   * @param nb_games : number of games in which the populations are evaluated.
   * @param nb_rounds : number of rounds per game.
   * @param target : colletive target.
   * @param risk : probability of collective loss.
   * @param available_actions : a vector containing the available actions per round.
   * @param populations : a vector containing the independent populations.
   * @param game : the game object.
   * @param data : a reference to the data container/structure.
   */
  template<class G>
  void evaluate_crd_populations(size_t nb_populations,
                                size_t population_size,
                                size_t group_size,
                                size_t nb_games,
                                size_t nb_rounds,
                                int target,
                                double risk,
                                ActionSpace &available_actions,
                                G &game,
                                DataTypes::DataTableCRD &data);

  /**
   *
   * @tparam G
   * @tparam U
   * @param nb_populations
   * @param population_size
   * @param group_size
   * @param nb_games
   * @param min_rounds
   * @param timing_uncertainty
   * @param target
   * @param risk
   * @param available_actions
   * @param game
   * @param data
   * @return the total number of rounds (since it is uncertain) so that the data can be resized.
   */
  template<class G, class U>
  size_t evaluate_crd_populationsTU(size_t nb_populations,
                                    size_t population_size,
                                    size_t group_size,
                                    size_t nb_games,
                                    size_t min_rounds,
                                    U &timing_uncertainty,
                                    int target,
                                    double risk,
                                    ActionSpace &available_actions,
                                    G &game,
                                    DataTypes::DataTableCRD &data);

  template<class G>
  void evaluate_crd_populationsThU(size_t nb_populations,
                                   size_t population_size,
                                   size_t group_size,
                                   size_t nb_games,
                                   size_t nb_rounds,
                                   int target,
                                   double delta,
                                   double risk,
                                   ActionSpace &available_actions,
                                   G &game,
                                   DataTypes::DataTableCRD &data);

  template<class G, class U>
  size_t evaluate_crd_populationsTUThU(size_t nb_populations,
                                       size_t population_size,
                                       size_t group_size,
                                       size_t nb_games,
                                       size_t min_rounds,
                                       U &timing_uncertainty,
                                       int target,
                                       double delta,
                                       double risk,
                                       ActionSpace &available_actions,
                                       G &game,
                                       DataTypes::DataTableCRD &data);

  // Simulation methods

  /**
   * @brief Runs a CRD simulation, without uncertainty, where players learn in groups and later mix for evaluation.
   *
   * @param nb_groups : number of islands/groups.
   * @param group_size : group size.
   * @param nb_generations : number of generations.
   * @param nb_games : number of games per generation.
   * @param nb_rounds : number of rounds per game.
   * @param target : collective target for the CRD.
   * @param endowment : private endowment for each player.
   * @param risk : probability that the remaining endowment is lost if the target isn't met.
   * @param available_actions : possible actions per round.
   * @param agent_type : algorithm used by the agent to adapt.
   * @param args : arguments to initialize the agent.
   * @return a data structure that includes the adapted population and the average group achievement and donations.
   */
  DataTypes::DataTableCRD
  run_group_islands(size_t nb_evaluation_games,
                    size_t nb_groups,
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
  run_conditional_group_islands(size_t nb_evaluation_games,
                                size_t nb_groups,
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

  /**
   * @brief CRD islands simulations with Timing uncertainty.
   *
   * @param nb_groups
   * @param group_size
   * @param nb_generations
   * @param nb_games
   * @param min_rounds
   * @param mean_rounds
   * @param max_rounds
   * @param p
   * @param target
   * @param endowment
   * @param risk
   * @param available_actions
   * @param agent_type
   * @param args
   * @return
   */
  DataTypes::DataTableCRD
  run_group_islandsTU(size_t nb_evaluation_games,
                      size_t nb_groups,
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
  run_conditional_group_islandsTU(size_t nb_evaluation_games,
                                  size_t nb_groups,
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

  /**
   * @brief CRD Islands simulation with Threshold uncertainty.
   *
   * @param nb_groups
   * @param group_size
   * @param nb_generations
   * @param nb_games
   * @param nb_rounds
   * @param target
   * @param delta
   * @param endowment
   * @param risk
   * @param available_actions
   * @param agent_type
   * @param args
   * @return
   */
  DataTypes::DataTableCRD
  run_group_islandsThU(size_t nb_evaluation_games,
                       size_t nb_groups,
                       size_t group_size,
                       size_t nb_generations,
                       size_t nb_games,
                       size_t nb_rounds,
                       int target,
                       double delta,
                       int endowment,
                       double risk,
                       ActionSpace &available_actions,
                       const std::string &agent_type,
                       const std::vector<double> &args = {});

  DataTypes::DataTableCRD
  run_conditional_group_islandsThU(size_t nb_evaluation_games,
                                   size_t nb_groups,
                                   size_t group_size,
                                   size_t nb_generations,
                                   size_t nb_games,
                                   size_t nb_rounds,
                                   int target,
                                   double delta,
                                   int endowment,
                                   double risk,
                                   ActionSpace &available_actions,
                                   const std::string &agent_type,
                                   const std::vector<double> &args = {});

  /**
   * @brief CRD Islands simulations with Timing and Threshold uncertainties.
   *
   * @param nb_groups
   * @param group_size
   * @param nb_generations
   * @param nb_games
   * @param min_rounds
   * @param mean_rounds
   * @param max_rounds
   * @param p
   * @param target
   * @param delta
   * @param endowment
   * @param risk
   * @param available_actions
   * @param agent_type
   * @param args
   * @return
   */
  DataTypes::DataTableCRD
  run_group_islandsTUThU(size_t nb_evaluation_games,
                         size_t nb_groups,
                         size_t group_size,
                         size_t nb_generations,
                         size_t nb_games,
                         size_t min_rounds,
                         size_t mean_rounds,
                         size_t max_rounds,
                         double p,
                         int target,
                         double delta,
                         int endowment,
                         double risk,
                         ActionSpace &available_actions,
                         const std::string &agent_type,
                         const std::vector<double> &args = {});

  DataTypes::DataTableCRD
  run_conditional_group_islandsTUThU(size_t nb_evaluation_games,
                                     size_t nb_groups,
                                     size_t group_size,
                                     size_t nb_generations,
                                     size_t nb_games,
                                     size_t min_rounds,
                                     size_t mean_rounds,
                                     size_t max_rounds,
                                     double p,
                                     int target,
                                     double delta,
                                     int endowment,
                                     double risk,
                                     ActionSpace &available_actions,
                                     const std::string &agent_type,
                                     const std::vector<double> &args = {});

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
  run_conditional_population_islands(size_t nb_evaluation_games,
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
  run_conditional_population_islandsTU(size_t nb_evaluation_games,
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
                            double delta,
                            int endowment,
                            double risk,
                            ActionSpace &available_actions,
                            const std::string &agent_type,
                            const std::vector<double> &args = {});

  DataTypes::DataTableCRD
  run_conditional_population_islandsThU(size_t nb_evaluation_games,
                                        size_t nb_populations,
                                        size_t population_size,
                                        size_t group_size,
                                        size_t nb_generations,
                                        size_t nb_games,
                                        size_t nb_rounds,
                                        int target,
                                        double delta,
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
                              double delta,
                              int endowment,
                              double risk,
                              ActionSpace &available_actions,
                              const std::string &agent_type,
                              const std::vector<double> &args = {});

  DataTypes::DataTableCRD
  run_conditional_population_islandsTUThU(size_t nb_evaluation_games,
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
                                          double delta,
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
   */
  void reinforce_population(double &pool,
                            size_t &success,
                            double target,
                            double &risk,
                            PopContainer &pop,
                            size_t &final_round);

  /**
 * @brief This method reinforces agents proportionally to the obtained payoff for Timing uncertainty games.
 * @tparam G : game type
 * @param pool : reference to the value contained in the public pool
 * @param success : reference to the variable that indicates if a group me the target
 * @param target : public target
 * @param risk : probability of disaster if the target isn't met
 * @param pop : population container
 * @param final_round : final round of the game
 */
  void reinforce_population(double &pool,
                            size_t &success,
                            double target,
                            double &risk,
                            PopContainer &pop);

  // Getter
  [[nodiscard]] size_t verbose_level() const;
  // Setter
  void set_verbose_level(size_t verbose_level);
  /**
   * Sets the decay factor and the minimum learning rate
   * @param decay : decay factor (to be multiplyed by the learning rate at each generation)
   * @param min_lr : minimum learning rate
   */
  void set_learning_rate_decay(double decay, double min_lr);

 private:
  // We consider 3 possible verbose levels
  // 0: only group success and average contributions are stored
  // 1: all information is stored
  size_t _verbose_level;
  double _decay, _min_learning_rate;
  std::uniform_real_distribution<double> _real_rand;
};
template<class G>
void CRDSimIslands::run_crd_single_group(size_t nb_generations,
                                         size_t nb_games,
                                         size_t nb_rounds,
                                         int target,
                                         double risk,
                                         ActionSpace &available_actions,
                                         EGTTools::RL::PopContainer &population,
                                         G &game) {
  size_t success = 0;
//  std::mt19937_64 generator(EGTTools::Random::SeedGenerator::getInstance().getSeed());

  for (size_t generation = 0; generation < nb_generations; ++generation) {
    success = 0;
    for (size_t i = 0; i < nb_games; ++i) {
      // First we play several games
      auto[pool, final_round] = game.playGame(population, available_actions, nb_rounds);
      reinforce_population(pool, success, target, risk, population);
    }
    // Then, update the population, reset the trajectory of each agent, and forget previous propensities.
    EGTTools::RL::helpers::calcProbabilitiesAndResetEpisode(population);
  }
}
template<class G, class U>
void CRDSimIslands::run_crd_single_groupTU(size_t nb_generations,
                                           size_t nb_games,
                                           size_t min_rounds,
                                           U timing_uncertainty,
                                           int target,
                                           double risk,
                                           EGTTools::RL::ActionSpace &available_actions,
                                           EGTTools::RL::PopContainer &population,
                                           G &game) {
  size_t success = 0;
  std::mt19937_64 generator(EGTTools::Random::SeedGenerator::getInstance().getSeed());

  for (size_t generation = 0; generation < nb_generations; ++generation) {
    success = 0;
    for (size_t i = 0; i < nb_games; ++i) {
      // First we play several games
      auto[pool, final_round] = game.playGame(population, available_actions, min_rounds, timing_uncertainty, generator);
      reinforce_population(pool, success, target, risk, population, final_round);
    }
    // Then, update the population, reset the trajectory of each agent, and forget previous propensities.
    EGTTools::RL::helpers::calcProbabilitiesAndResetEpisode(population);
  }
}

template<class G>
void CRDSimIslands::run_crd_single_groupThU(size_t nb_generations,
                                            size_t nb_games,
                                            size_t nb_rounds,
                                            int target,
                                            double delta,
                                            double risk,
                                            EGTTools::RL::ActionSpace &available_actions,
                                            EGTTools::RL::PopContainer &population,
                                            G &game) {
  size_t success = 0;
  std::mt19937_64 generator(EGTTools::Random::SeedGenerator::getInstance().getSeed());

  // Define the distribution for the threshold
  std::uniform_int_distribution<int> t_dist(static_cast<int>(target * (1 - delta)),
                                            static_cast<int>(target * (1 + delta)));

  for (size_t generation = 0; generation < nb_generations; ++generation) {
    success = 0;
    for (size_t i = 0; i < nb_games; ++i) {
      // First we play several games
      auto[pool, final_round] = game.playGame(population, available_actions, nb_rounds);
      reinforce_population(pool, success, t_dist(generator), risk, population);
    }
    // Then, update the population, reset the trajectory of each agent, and forget previous propensities.
    EGTTools::RL::helpers::calcProbabilitiesAndResetEpisode(population);
  }
}
template<class G, class U>
void CRDSimIslands::run_crd_single_groupTUThU(size_t nb_generations,
                                              size_t nb_games,
                                              size_t min_rounds,
                                              U timing_uncertainty,
                                              int target,
                                              double delta,
                                              double risk,
                                              EGTTools::RL::ActionSpace &available_actions,
                                              EGTTools::RL::PopContainer &population,
                                              G &game) {
  size_t success = 0;
  std::mt19937_64 generator(EGTTools::Random::SeedGenerator::getInstance().getSeed());
  // Define the distribution for the threshold
  std::uniform_int_distribution<int> t_dist(static_cast<int>(target * (1 - delta)),
                                            static_cast<int>(target * (1 + delta)));

  for (size_t generation = 0; generation < nb_generations; ++generation) {
    success = 0;
    for (size_t i = 0; i < nb_games; ++i) {
      // First we play several games
      auto[pool, final_round] = game.playGame(population, available_actions, min_rounds, timing_uncertainty, generator);
      reinforce_population(pool, success, t_dist(generator), risk, population, final_round);
    }
    // Then, update the population, reset the trajectory of each agent, and forget previous propensities.
    EGTTools::RL::helpers::calcProbabilitiesAndResetEpisode(population);
  }
}
template<class G>
void CRDSimIslands::run_crd_population(size_t population_size,
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
      reinforce_population(pool, success, target, risk, group);
      container.clear();
    }

    // Then, update the population, reset the trajectory of each agent, and forget previous propensities.
    EGTTools::RL::helpers::calcProbabilitiesAndResetEpisode(population);
//    EGTTools::RL::helpers::calcProbabilitiesResetEpisodeAndUpdateLearningRate(population, _decay, _min_learning_rate);
  }

}
template<class G, class U>
void CRDSimIslands::run_crd_populationTU(size_t population_size,
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
      reinforce_population(pool, success, target, risk, group, final_round);
      container.clear();
    }
    // Then, update the population, reset the trajectory of each agent, and forget previous propensities.
    EGTTools::RL::helpers::calcProbabilitiesAndResetEpisode(population);
  }
}
template<class G>
void CRDSimIslands::run_crd_populationThU(size_t population_size,
                                          size_t group_size,
                                          size_t nb_generations,
                                          size_t nb_games,
                                          size_t nb_rounds,
                                          int target,
                                          double delta,
                                          double risk,
                                          ActionSpace &available_actions,
                                          PopContainer &population,
                                          G &game) {
  size_t success = 0;
  std::mt19937_64 generator(EGTTools::Random::SeedGenerator::getInstance().getSeed());

  // Define the distribution for the threshold
  std::uniform_int_distribution<int> t_dist(static_cast<int>(target * (1 - delta)),
                                            static_cast<int>(target * (1 + delta)));

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
      reinforce_population(pool, success, t_dist(generator), risk, group);
      container.clear();
    }

    // Then, update the population, reset the trajectory of each agent, and forget previous propensities.
    EGTTools::RL::helpers::calcProbabilitiesAndResetEpisode(population);
  }

}
template<class G, class U>
void CRDSimIslands::run_crd_populationTUThU(size_t population_size,
                                            size_t group_size,
                                            size_t nb_generations,
                                            size_t nb_games,
                                            size_t min_rounds,
                                            U timing_uncertainty,
                                            int target,
                                            double delta,
                                            double risk,
                                            ActionSpace &available_actions,
                                            PopContainer &population,
                                            G &game) {
  size_t success = 0;
  std::mt19937_64 generator(EGTTools::Random::SeedGenerator::getInstance().getSeed());

  // Define the distribution for the threshold
  std::uniform_int_distribution<int> t_dist(static_cast<int>(target * (1 - delta)),
                                            static_cast<int>(target * (1 + delta)));

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
      reinforce_population(pool, success, t_dist(generator), risk, group, final_round);
      container.clear();
    }
    // Then, update the population, reset the trajectory of each agent, and forget previous propensities.
    EGTTools::RL::helpers::calcProbabilitiesAndResetEpisode(population);
  }
}
template<class G>
void CRDSimIslands::evaluate_crd_populations(size_t nb_populations,
                                             size_t population_size,
                                             size_t group_size,
                                             size_t nb_games,
                                             size_t nb_rounds,
                                             int target,
                                             double risk,
                                             EGTTools::RL::ActionSpace &available_actions,
                                             G &game,
                                             EGTTools::RL::DataTypes::DataTableCRD &data) {
  // Some helpful variables
  size_t total_population_size = nb_populations * population_size;
  bool success;
  int public_pool = 0;
  size_t data_index = 0;
  Matrix2D game_data = Matrix2D::Zero(group_size, nb_rounds);
  std::mt19937_64 generator(EGTTools::Random::SeedGenerator::getInstance().getSeed());

  // We will need a container for the random groups
  PopContainer group;
  std::unordered_set<size_t> container;
  container.reserve(group_size);

  // This step is simply to make sure that we have allocated memory for the group
  for (size_t i = 0; i < group_size; ++i)
    group.push_back(data.populations[0](i));

  // Depending on the verbose level we store more or less data
  if (_verbose_level == 0) {
    for (size_t i = 0; i < nb_games; ++i) {
      // At each iteration, we sample a random group, taking players
      // randomly from any population. For that we sample integers from the range
      // [0, nb_populations * population_size)
      EGTTools::sampling::sample_without_replacement(total_population_size, group_size, container, generator);
      int j = 0;
      for (const auto &elem: container) {
        group(j) = data.populations[elem / population_size](elem % population_size);
        j++;
      }
      // First we play the game
      auto[public_pool, final_round] = game.playGame(group, available_actions, nb_rounds);
      // Check if the game was successful
      success = public_pool >= target;
      // Now we update the data table with the info of this game
      // group, success, final_public_account, final_round
      data.data(data_index, 0) = i;
      data.data(data_index, 1) = success;
      data.data(data_index, 2) = public_pool;
      data_index++;
      container.clear();
    }
  } else {
    for (size_t i = 0; i < nb_games; ++i) {
      // At each iteration, we sample a random group, taking players
      // randomly from any population. For that we sample integers from the range
      // [0, nb_populations * population_size)
      EGTTools::sampling::sample_without_replacement(total_population_size, group_size, container, generator);
      int j = 0;
      for (const auto &elem: container) {
        group(j) = data.populations[elem / population_size](elem % population_size);
        j++;
      }
      // First we play the game
      public_pool = game.playGameVerbose(group, available_actions, nb_rounds, game_data);
      // Check if the game was successful
      success = public_pool >= target;
      if ((!success) && _real_rand(generator) < risk)
        game.setPayoffs(group, 0);

      // Now we update the data table with the info of this game
      // For each player and round, we add:
      // group, player, game_index, round, action, group_contributions, contributions_others,
      // total_contribution, payoff, success, target, final_public_account, final_round
      j = 0;
      for (const auto &elem: container) {
        auto total_contribution = game_data.row(j).sum();
        for (size_t r = 0; r < nb_rounds; ++r) {
          auto group_contributions = game_data.col(r).sum();
          data.data(data_index, 0) = i;
          data.data(data_index, 1) = elem;
          data.data(data_index, 2) = j;
          data.data(data_index, 3) = r;
          data.data(data_index, 4) = game_data(j, r);
          data.data(data_index, 5) = group_contributions;
          data.data(data_index, 6) = group_contributions - game_data(j, r);
          data.data(data_index, 7) = total_contribution;
          data.data(data_index, 8) = group(j)->payoff();
          data.data(data_index, 9) = success;
          data.data(data_index, 10) = target;
          data.data(data_index, 11) = public_pool;
          data.data(data_index, 12) = nb_rounds;
          data_index++;
        }
        j++;
      }
      container.clear();
    }
  }
}
template<class G, class U>
size_t CRDSimIslands::evaluate_crd_populationsTU(size_t nb_populations,
                                                 size_t population_size,
                                                 size_t group_size,
                                                 size_t nb_games,
                                                 size_t min_rounds,
                                                 U &timing_uncertainty,
                                                 int target,
                                                 double risk,
                                                 ActionSpace &available_actions,
                                                 G &game,
                                                 DataTypes::DataTableCRD &data) {
  // Some helpful variables
  size_t total_population_size = nb_populations * population_size;
  bool success;
  size_t total_nb_rounds = 0;
  size_t data_index = 0;
  Matrix2D game_data = Matrix2D::Zero(group_size, timing_uncertainty.max_rounds());
  std::mt19937_64 generator(EGTTools::Random::SeedGenerator::getInstance().getSeed());

  // We will need a container for the random groups
  PopContainer group;
  std::unordered_set<size_t> container;
  container.reserve(group_size);

  // This step is simply to make sure that we have allocated memory for the group
  for (size_t i = 0; i < group_size; ++i)
    group.push_back(data.populations[0](i));

  // Depending on the verbose level we store more or less data
  if (_verbose_level == 0) {
    // the number of columns correspond to each game in the case
    total_nb_rounds = nb_games;
    for (size_t i = 0; i < nb_games; ++i) {
      // At each iteration, we sample a random group, taking players
      // randomly from any population. For that we sample integers from the range
      // [0, nb_populations * population_size)
      EGTTools::sampling::sample_without_replacement(total_population_size, group_size, container, generator);
      int j = 0;
      for (const auto &elem: container) {
        group(j) = data.populations[elem / population_size](elem % population_size);
        j++;
      }

      // First we play the game
      auto[public_pool, final_round] = game.playGame(group,
                                                     available_actions,
                                                     min_rounds,
                                                     timing_uncertainty,
                                                     generator);
      // Check if the game was successful
      success = public_pool >= target;
      // Now we update the data table with the info of this game
      // For each player and round, we add:
      // group, success, final_public_account, final_round
      data.data(data_index, 0) = i;
      data.data(data_index, 1) = success;
      data.data(data_index, 2) = public_pool;
      data.data(data_index, 3) = final_round;
      data_index++;
      container.clear();
    }
  } else {
    // The agents will be evaluated through nb_games games
    for (size_t i = 0; i < nb_games; ++i) {
      // At each iteration, we sample a random group, taking players
      // randomly from any population. For that we sample integers from the range
      // [0, nb_populations * population_size)
      EGTTools::sampling::sample_without_replacement(total_population_size, group_size, container, generator);
      int j = 0;
      for (const auto &elem: container) {
        group(j) = data.populations[elem / population_size](elem % population_size);
        j++;
      }

      // First we play the game
      auto[public_pool, final_round] = game.playGameVerbose(group,
                                                            available_actions,
                                                            min_rounds,
                                                            timing_uncertainty,
                                                            generator,
                                                            game_data);
      total_nb_rounds += final_round;
      // Check if the game was successful
      success = public_pool >= target;
      if ((!success) && _real_rand(generator) < risk)
        game.setPayoffs(group, 0);

      // Now we update the data table with the info of this game
      // For each player and round, we add:
      // group, player, game_index, round, action, group_contributions, contributions_others,
      // total_contribution, payoff, success, target, final_public_account, final_round
      j = 0;
      for (const auto &elem: container) {
        auto total_contribution = game_data.row(j).leftCols(final_round).sum();
        for (size_t r = 0; r < final_round; ++r) {
          auto group_contributions = game_data.col(r).sum();
          data.data(data_index, 0) = i;
          data.data(data_index, 1) = elem;
          data.data(data_index, 2) = j;
          data.data(data_index, 3) = r;
          data.data(data_index, 4) = game_data(j, r);
          data.data(data_index, 5) = group_contributions;
          data.data(data_index, 6) = group_contributions - game_data(j, r);
          data.data(data_index, 7) = total_contribution;
          data.data(data_index, 8) = group(j)->payoff();
          data.data(data_index, 9) = success;
          data.data(data_index, 10) = target;
          data.data(data_index, 11) = public_pool;
          data.data(data_index, 12) = final_round;
          data_index++;
        }
        j++;
      }
      container.clear();
    }
  }
  return total_nb_rounds;
}
template<class G>
void CRDSimIslands::evaluate_crd_populationsThU(size_t nb_populations,
                                                size_t population_size,
                                                size_t group_size,
                                                size_t nb_games,
                                                size_t nb_rounds,
                                                int target,
                                                double delta,
                                                double risk,
                                                EGTTools::RL::ActionSpace &available_actions,
                                                G &game,
                                                EGTTools::RL::DataTypes::DataTableCRD &data) {
  // Some helpful variables
  size_t total_population_size = nb_populations * population_size;
  bool success;
  int public_pool = 0;
  size_t data_index = 0;
  Matrix2D game_data = Matrix2D::Zero(group_size, nb_rounds);
  std::mt19937_64 generator(EGTTools::Random::SeedGenerator::getInstance().getSeed());
  // Define the distribution for the threshold
  std::uniform_int_distribution<int> t_dist(static_cast<int>(target * (1 - delta)),
                                            static_cast<int>(target * (1 + delta)));

  // We will need a container for the random groups
  PopContainer group;
  std::unordered_set<size_t> container;
  container.reserve(group_size);

  // This step is simply to make sure that we have allocated memory for the group
  for (size_t i = 0; i < group_size; ++i)
    group.push_back(data.populations[0](i));

  // Depending on the verbose level we store more or less data
  if (_verbose_level == 0) {
    for (size_t i = 0; i < nb_games; ++i) {
      // At each iteration, we sample a random group, taking players
      // randomly from any population. For that we sample integers from the range
      // [0, nb_populations * population_size)
      EGTTools::sampling::sample_without_replacement(total_population_size, group_size, container, generator);
      int j = 0;
      for (const auto &elem: container) {
        group(j) = data.populations[elem / population_size](elem % population_size);
        j++;
      }
      // First we play the game
      auto[public_pool, final_round] = game.playGame(group, available_actions, nb_rounds);
      // Check if the game was successful
      auto final_target = t_dist(generator);
      success = public_pool >= final_target;
      // Now we update the data table with the info of this game
      // For each player and round, we add:
      // group, success, final_public_account, final_target
      data.data(data_index, 0) = i;
      data.data(data_index, 1) = success;
      data.data(data_index, 2) = public_pool;
      data.data(data_index, 3) = final_target;
      data_index++;
      container.clear();
    }
  } else {
    // The agents will be evaluated through nb_games games
    for (size_t i = 0; i < nb_games; ++i) {
      // At each iteration, we sample a random group, taking players
      // randomly from any population. For that we sample integers from the range
      // [0, nb_populations * population_size)
      EGTTools::sampling::sample_without_replacement(total_population_size, group_size, container, generator);
      int j = 0;
      for (const auto &elem: container) {
        group(j) = data.populations[elem / population_size](elem % population_size);
        j++;
      }

      // First we play the game
      public_pool = game.playGameVerbose(group, available_actions, nb_rounds, game_data);
      // Check if the game was successful
      auto final_target = t_dist(generator);
      success = public_pool >= final_target;
      if ((!success) && _real_rand(generator) < risk)
        game.setPayoffs(group, 0);

      // Now we update the data table with the info of this game
      // For each player and round, we add:
      // group, player, game_index, round, action, group_contributions, contributions_others,
      // total_contribution, payoff, success, target, final_public_account, final_round
      j = 0;
      for (const auto &elem: container) {
        auto total_contribution = game_data.row(j).sum();
        for (size_t r = 0; r < nb_rounds; ++r) {
          auto group_contributions = game_data.col(r).sum();
          data.data(data_index, 0) = i;
          data.data(data_index, 1) = elem;
          data.data(data_index, 2) = j;
          data.data(data_index, 3) = r;
          data.data(data_index, 4) = game_data(j, r);
          data.data(data_index, 5) = group_contributions;
          data.data(data_index, 6) = group_contributions - game_data(j, r);
          data.data(data_index, 7) = total_contribution;
          data.data(data_index, 8) = group(j)->payoff();
          data.data(data_index, 9) = success;
          data.data(data_index, 10) = final_target;
          data.data(data_index, 11) = public_pool;
          data.data(data_index, 12) = nb_rounds;
          data_index++;
        }
        j++;
      }
      container.clear();
    }
  }
}
template<class G, class U>
size_t CRDSimIslands::evaluate_crd_populationsTUThU(size_t nb_populations,
                                                    size_t population_size,
                                                    size_t group_size,
                                                    size_t nb_games,
                                                    size_t min_rounds,
                                                    U &timing_uncertainty,
                                                    int target,
                                                    double delta,
                                                    double risk,
                                                    ActionSpace &available_actions,
                                                    G &game,
                                                    DataTypes::DataTableCRD &data) {
  // Some helpful variables
  size_t total_population_size = nb_populations * population_size;
  bool success;
  size_t total_nb_rounds = 0;
  size_t data_index = 0;
  Matrix2D game_data = Matrix2D::Zero(group_size, timing_uncertainty.max_rounds());
  std::mt19937_64 generator(EGTTools::Random::SeedGenerator::getInstance().getSeed());
  // Define the distribution for the threshold
  std::uniform_int_distribution<int> t_dist(static_cast<int>(target * (1 - delta)),
                                            static_cast<int>(target * (1 + delta)));

  // We will need a container for the random groups
  PopContainer group;
  std::unordered_set<size_t> container;
  container.reserve(group_size);

  // This step is simply to make sure that we have allocated memory for the group
  for (size_t i = 0; i < group_size; ++i)
    group.push_back(data.populations[0](i));

  // Depending on the verbose level we store more or less data
  if (_verbose_level == 0) {
    // the number of columns correspond to the number of groups (games) in this case
    total_nb_rounds = nb_games;
    // The agents will be evaluated through nb_games games
    for (size_t i = 0; i < nb_games; ++i) {
      // At each iteration, we sample a random group, taking players
      // randomly from any population. For that we sample integers from the range
      // [0, nb_populations * population_size)
      EGTTools::sampling::sample_without_replacement(total_population_size, group_size, container, generator);
      int j = 0;
      for (const auto &elem: container) {
        group(j) = data.populations[elem / population_size](elem % population_size);
        j++;
      }

      // First we play the game
      auto[public_pool, final_round] = game.playGame(group,
                                                     available_actions,
                                                     min_rounds,
                                                     timing_uncertainty,
                                                     generator);
      // Check if the game was successful
      auto final_target = t_dist(generator);
      success = public_pool >= final_target;
      // Now we update the data table with the info of this game
      // For each player and round, we add:
      // group, success, final_public_account, final_round, final_target
      data.data(data_index, 0) = i;
      data.data(data_index, 1) = success;
      data.data(data_index, 2) = public_pool;
      data.data(data_index, 3) = final_round;
      data.data(data_index, 4) = final_target;
      data_index++;
      container.clear();
    }
  } else {
    // The agents will be evaluated through nb_games games
    for (size_t i = 0; i < nb_games; ++i) {
      // At each iteration, we sample a random group, taking players
      // randomly from any population. For that we sample integers from the range
      // [0, nb_populations * population_size)
      EGTTools::sampling::sample_without_replacement(total_population_size, group_size, container, generator);
      int j = 0;
      for (const auto &elem: container) {
        group(j) = data.populations[elem / population_size](elem % population_size);
        j++;
      }

      // First we play the game
      auto[public_pool, final_round] = game.playGameVerbose(group,
                                                            available_actions,
                                                            min_rounds,
                                                            timing_uncertainty,
                                                            generator,
                                                            game_data);
      total_nb_rounds += final_round;
      // Check if the game was successful
      auto final_target = t_dist(generator);
      success = public_pool >= final_target;
      if ((!success) && _real_rand(generator) < risk)
        game.setPayoffs(group, 0);

      // Now we update the data table with the info of this game
      // For each player and round, we add:
      // group, player, game_index, round, action, group_contributions, contributions_others,
      // total_contribution, payoff, success, target, final_public_account, final_round
      j = 0;
      for (const auto &elem: container) {
        auto total_contribution = game_data.leftCols(final_round).row(j).sum();
        for (size_t r = 0; r < final_round; ++r) {
          auto group_contributions = game_data.col(r).sum();
          data.data(data_index, 0) = i;
          data.data(data_index, 1) = elem;
          data.data(data_index, 2) = j;
          data.data(data_index, 3) = r;
          data.data(data_index, 4) = game_data(j, r);
          data.data(data_index, 5) = group_contributions;
          data.data(data_index, 6) = group_contributions - game_data(j, r);
          data.data(data_index, 7) = total_contribution;
          data.data(data_index, 8) = group(j)->payoff();
          data.data(data_index, 9) = success;
          data.data(data_index, 10) = final_target;
          data.data(data_index, 11) = public_pool;
          data.data(data_index, 12) = final_round;
          data_index++;
        }
        j++;
      }
      container.clear();
    }
  }
  return total_nb_rounds;
}
} // namespace EGTTools::RL::Simulators

#endif //DYRWIN_RL_SIMULATORS_CRDISLANDS_H_
