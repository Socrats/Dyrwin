//
// Created by Elias Fernandez on 2019-04-10.
//

#ifndef DYRWIN_RL_ALGORITHMS_DISCOUNTEDQLEARNING_H
#define DYRWIN_RL_ALGORITHMS_DISCOUNTEDQLEARNING_H

#include <cmath>
#include <iostream>
#include <cassert>
#include <vector>
#include <Dyrwin/Distributions.h>
#include <Dyrwin/SeedGenerator.h>
#include <Dyrwin/RL/algorithms/AbstractRLAlgorithm.hpp>

namespace EGTTools::RL::algorithms {
/**
 * payoff = reward in RL terminology
 */
class QLearning : public AbstractRLAlgorithm {
 public:
  QLearning(size_t nb_states,
            size_t nb_actions,
            size_t episode_length,
            double endowment,
            double alpha,
            double lambda,
            double temperature);

  ~QLearning() override = default;

  friend std::ostream &operator<<(std::ostream &o, QLearning &r) { return r.display(o); }

  // Abstract class methods
  /**
   * @brief Updates the q-value estimations of the current trajectory.
   *
   * This method should be used if the number of states that it takes
   * to reach a terminal state (episode_length) is a constant. E.g.,
   * when the number of rounds is constant in an iterative dilemma.
   *
   * The trajectory vector is reset to 0 after the update.
   */
  void reinforceTrajectory() override;

  /**
   * @brief Updates the q-value estimations of the current trajectory.
   *
   * It uses only the values of the trajectory until episode_length. This method
   * should be used when the episode_length is variable, i.e., when the number of
   * terminal states that it takes to reach the terminal state of the environment
   * differs on different executions. E.g., when the number of rounds in an iterative
   * dilemma is variable.
   *
   * The trajectory vector is reset to 0 after the update.
   *
   * @param episode_length
   */
  void reinforceTrajectory(size_t episode_length);

  /**
   * @brief Resets the values of the trajectory vectors
   */
  void resetTrajectory() override;

  /**
   * @brief Updates the strategy profile (policy) matrix from the estrimated q-values.
   *
   * This method uses the defined temperature to make an update using the Gibbs-boltzman
   * distribution (softmax).
   *
   * @return True if the update issued no exceptions.
   */
  bool inferPolicy() override;

  /**
   * @brief Samples an action from the policy for the give \p state.
   *
   * This method draws a random action from the action set using the probabilities
   * defined in the strategy profile (policy) for the given \p state.
   *
   * @param state : current state of the environment
   * @return a size_t indicating the selected action
   */
  size_t selectAction(size_t state) override;

  /**
   * @brief Samples an action from the policy for the give \p state.
   *
   * This method draws a random action from the action set using the probabilities
   * defined in the strategy profile (policy) for the given \p state.
   *
   * @param state : current state of the environment
   * @param current_round : current round at the game/episode
   * @return a size_t indicating the selected action
   */
  size_t selectAction(size_t current_round, size_t state);

  /**
   * @brief Decreases the payoff (reward) by \p value.
   * @param value : amount to decrease the payoff
   */
  bool decreasePayoff(double value) override;

  /**
   * @brief Increases the payoff (reward) by \p value.
   * @param value : value to increase the payoff
   */
  void increasePayoff(double value) override;

  /**
   * @brief Resets the object.
   *
   * This method initialises the strategy profile (policy) and q-values of the algorithm
   * to a random value. It also resets the trajectories and payoff.
   */
  void reset() override;

  /**
  * @brief Q-values are reset to random values.
  */
  void resetQValues();

  // Getters
  [[nodiscard]] std::string toString() const override;
  /**
   * @brief Returns a string indicating the agent's class name
   * @return Returns a string indicating the agent's class name
   */
  [[nodiscard]] std::string type() const override;
  [[nodiscard]] size_t nb_states() const override;
  [[nodiscard]] size_t nb_actions() const override;
  [[nodiscard]] double endowment() const;
  [[nodiscard]] double payoff() const override;
  [[nodiscard]] double alpha() const;
  [[nodiscard]] double lambda() const;
  [[nodiscard]] double temperature() const;
  [[nodiscard]] const VectorXui &trajectoryStates() const override;
  [[nodiscard]] const VectorXui &trajectoryActions() const override;
  [[nodiscard]] const Matrix2D &policy() const override;
  [[nodiscard]] const Matrix2D &qValues() const;

  // Setters
  void set_nb_states(size_t nb_states) override;
  void set_nb_actions(size_t nb_actions) override;
  void set_payoff(double payoff) override;
  void set_endowment(double endowment);
  void reset_payoff() override;
  void set_policy(const Eigen::Ref<const Matrix2D> &policy) override;
  void set_q_values(const Eigen::Ref<const Matrix2D> &q_values);
  void set_trajectory_round(size_t round, size_t action);
  void setAlpha(double alpha);
  void setLambda(double lambda);
  void setTemperature(double temperature);

 private:
  /**
   * serves as a display proxy for when the class is converted to string
   * @param os : and output stream to push the display information
   */
  std::ostream &display(std::ostream &os) const;

  size_t _nb_states, _nb_actions; // Action-state space definition
  size_t _episode_length; // Maximum episode length (maximum number of states until a terminal states is reached)
//  size_t _batch_length; // length of the batch used to update estimations
  double _endowment, _payoff; // variables to keep rewards & maximum expending capacity

  // Q-Learning dependant variables
  double _alpha; // learning rate
  double _lambda; // forgetting factor (or discount rate)
  double _temperature; // temperature of the boltzmann distribution

  Matrix2D _policy; // matrix where the strategy profile is stored
  Matrix2D _q_values; // matrix where the q-values are stored
  Matrix2D _batch_updates; // Stores the sum of the temporal difference updates
                              // during the batch of size K
  Matrix2D _batch_counts; // K(s, a) counts the number of times that
                          // each state-action pair was visited in the current batch
  VectorXui _trajectory_states, _trajectory_actions; // variable used to store trajectories

  // Variables used to implement the algorithm and for the calculation process
  std::vector<size_t> _buffer;
  std::uniform_real_distribution<double> _real_rand;

  // Random generator
  std::mt19937_64 _generator{EGTTools::Random::SeedGenerator::getInstance().getSeed()};
};
}

#endif //DYRWIN_RL_ALGORITHMS_DISCOUNTEDQLEARNING_H
