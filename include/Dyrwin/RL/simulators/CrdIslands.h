//
// Created by Elias Fernandez on 08/02/2020.
//

#ifndef DYRWIN_RL_SIMULATORS_CRDISLANDS_H_
#define DYRWIN_RL_SIMULATORS_CRDISLANDS_H_

#include <random>
#include <unordered_set>
#include <Dyrwin/RL/CRDGame.h>
#include <Dyrwin/RL/CRDConditional.h>
#include <Dyrwin/RL/PopContainer.hpp>
#include <Dyrwin/RL/Utils.h>
#include <Dyrwin/RL/Data.hpp>
#include <Dyrwin/Sampling.h>
#include <Dyrwin/SeedGenerator.h>
#include <Dyrwin/OpenMPUtils.hpp>

namespace EGTTools::RL::CRD::Simulators {

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
  DataTypes::CRDDataIslands
  run_group_islands(size_t nb_groups,
                    size_t group_size,
                    size_t nb_generations,
                    size_t nb_games,
                    size_t nb_rounds,
                    int target,
                    int endowment,
                    double risk,
                    const ActionSpace &available_actions,
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
  DataTypes::CRDDataIslands
  run_group_islandsTU(size_t nb_groups,
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
                      const ActionSpace &available_actions,
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
  DataTypes::CRDDataIslands
  run_group_islandsThU(size_t nb_groups,
                       size_t group_size,
                       size_t nb_generations,
                       size_t nb_games,
                       size_t nb_rounds,
                       int target,
                       int delta,
                       int endowment,
                       double risk,
                       const ActionSpace &available_actions,
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
  DataTypes::CRDDataIslands
  run_group_islandsTUThU(size_t nb_groups,
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
                         const ActionSpace &available_actions,
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
  DataTypes::CRDDataIslands
  run_population_islands(size_t nb_populations,
                         size_t population_size,
                         size_t group_size,
                         size_t nb_generations,
                         size_t nb_games,
                         size_t nb_rounds,
                         int target,
                         int endowment,
                         double risk,
                         const ActionSpace &available_actions,
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
  template<class G = CRDGame<PopContainer, TimingUncertainty<std::mt19937_64>>>
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
  template<class G = CRDGame<PopContainer>>
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
} // namespace EGTTools::RL::Simulators

#endif //DYRWIN_RL_SIMULATORS_CRDISLANDS_H_
