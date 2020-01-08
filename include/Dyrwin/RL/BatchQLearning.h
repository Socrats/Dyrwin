//
// Created by Elias Fernandez on 2019-04-10.
//

#ifndef DYRWIN_RL_BATCHQLEARNING_H
#define DYRWIN_RL_BATCHQLEARNING_H

#include <cmath>
#include <Dyrwin/RL/Agent.h>

namespace EGTTools::RL {
class BatchQLearning : public Agent {
 public:
  BatchQLearning(size_t nb_states, size_t nb_actions, size_t episode_length, double endowment, double alpha,
                      double temperature);

  void reinforceTrajectory() override;

  void reinforceTrajectory(size_t episode_length) override;

  bool inferPolicy() override;

  /**
  * Roth-Erev already has a discount factor, so it's not necessary to reinitialize
  * the weights.
  */
  void resetQValues() override;

  /**
   * @briefs Returns a string indicating the agent's class name
   * @return Returns a string indicating the agent's class name
   */
  [[nodiscard]] std::string type() const override;

  // Getters
  [[nodiscard]] double alpha() const;

  [[nodiscard]] double temperature() const;

  // Setters
  void setAlpha(double alpha);

  void setTemperature(double temperature);

 private:
  double _alpha; // learning rate
  double _temperature; // temperature of the boltzman distribution
  Matrix2D _batch_updates; // Stores the sum of the temporal difference updates
  // during the batch of size K
  Matrix2D _batch_counts; // K(s, a) counts the number of times that
  // each state-action pair was visited in the current batch

};
}

#endif //DYRWIN_RL_BATCHQLEARNINGAGENT_H
