//
// Created by Elias Fernandez on 28/12/2020.
//

#ifndef DYRWIN_INCLUDE_DYRWIN_SED_GAMES_NORMALFORMGAME_H_
#define DYRWIN_INCLUDE_DYRWIN_SED_GAMES_NORMALFORMGAME_H_

#include <cassert>
#include <fstream>
#include <Dyrwin/Distributions.h>
#include <Dyrwin/SED/games/AbstractGame.hpp>
#include <Dyrwin/OpenMPUtils.hpp>

namespace EGTTools::SED {
using PayoffVector = std::vector<double>;
using RandomDist = std::uniform_real_distribution<double>;

class NormalFormGame final : public EGTTools::SED::AbstractGame {
 public:
  /**
   * @brief This class implements a normal form game.
   *
   * The payoffs of the game are defined by a payoff matrix.
   * It is always a 2-player game, but may contain more than 2 possible actions.
   *
   * If @param nb_rounds > 1, than the game is iterated (has more than 1 round).
   *
   * In case the number of rounds is > 1, this class will estimate
   * The expected payoff for each strategy and update it's own internal
   * payoff matrix.
   *
   * @param nb_rounds : number of rounds of the game.
   * @param payoff_matrix : Eigen matrix containing the payoffs.
   */
  NormalFormGame(size_t nb_rounds, const Eigen::Ref<const Matrix2D> &payoff_matrix);

  void play(const EGTTools::SED::StrategyCounts &group_composition,
            PayoffVector &game_payoffs) override;

  /**
 * @brief Gets an action from the strategy defined by player type.
 *
 * This method will call one of the behaviors specified in CrdBehaviors.hpp indexed by
 * @param player_type with the parameters @param prev_donation, threshold, current_round.
 *
 * @param player_type : type of strategy (as an unsigned integer).
 * @param prev_action : previous donation of the group.
 * @param current_round : current round of the game
 * @return action of the strategy
 */
//  static inline size_t get_action(const size_t &player_type, const size_t &prev_action, const size_t &current_round);

  /**
   * @brief updates private payoff matrix and returns it
   *
   * @return payoff matrix of the game
   */
  const GroupPayoffs &calculate_payoffs() override;

  double
  calculate_fitness(const size_t &player_type, const size_t &pop_size,
                    const Eigen::Ref<const VectorXui> &strategies) override;

  /**
   * @brief Calculates the expected level of cooperation given a population state
   * @param pop_size : size of the population
   * @param population_state : state of the population (number of players of each strategy)
   * @return the level of cooperation of that population state
   */
  double calculate_cooperation_level(size_t pop_size, const Eigen::Ref<const VectorXui> &population_state);

  // getters
  [[nodiscard]] size_t nb_strategies() const override;
  [[nodiscard]] size_t nb_rounds() const;
  [[nodiscard]] size_t nb_states() const;
  [[nodiscard]] std::string toString() const override;
  [[nodiscard]] std::string type() const override;
  [[nodiscard]] const GroupPayoffs &payoffs() const override;
  [[nodiscard]] const Matrix2D &expected_payoffs() const;
  [[nodiscard]] double payoff(size_t strategy, const EGTTools::SED::StrategyCounts &group_composition) const override;
  void save_payoffs(std::string file_name) const override;

  // setters

 protected:
  size_t nb_rounds_, nb_strategies_, nb_states_;
  Matrix2D payoffs_, expected_payoffs_, coop_level_;

  /**
   * @brief updates the expected_payoffs_ and coop_level_ matrices for the strategies indicates
   * @param s1 : strategy 1
   * @param s2 : strategy 2
   */
  void _update_cooperation_and_payoffs(size_t s1, size_t s2);
};

}

#endif //DYRWIN_INCLUDE_DYRWIN_SED_GAMES_NORMALFORMGAME_H_
