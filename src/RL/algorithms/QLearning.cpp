//
// Created by Elias Fernandez on 04/01/2020.
//

#include <Dyrwin/RL/algorithms/QLearning.h>

EGTTools::RL::algorithms::QLearning::QLearning(size_t nb_states,
                                               size_t nb_actions,
                                               size_t episode_length,
                                               double endowment,
                                               double alpha,
                                               double lambda,
                                               double temperature)
    : _nb_states(nb_states),
      _nb_actions(nb_actions),
      _episode_length(episode_length),
      _endowment(endowment),
      _alpha(alpha),
      _lambda(lambda),
      _temperature(temperature) {
  _payoff = _endowment;
  // Initialise q-values uniformly random between (-1, 1).
  _q_values = Matrix2D::Random(_nb_states, _nb_actions);
  // Initialise all actions with equal probability
  _policy = Matrix2D::Constant(_nb_states, _nb_actions, 1.0 / static_cast<double>(_nb_actions));
  // Initialise to zero the state-action counts
  _batch_counts = Matrix2D::Zero(_nb_states, _nb_actions);
  // Initialise trajectory (the actions taken at each round)
  _trajectory_states = VectorXui::Zero(_episode_length);
  _trajectory_actions = VectorXui::Zero(_episode_length);
  // Initialise the helper containers
  _buffer = std::vector<size_t>(_nb_actions);
  _real_rand = std::uniform_real_distribution<double>(0.0, 1.0);
}

void EGTTools::RL::algorithms::QLearning::reinforceTrajectory() {
  // Reinforces the trajectory and adds to the count vector the states of the trajectory
  for (unsigned i = 0; i < _episode_length; ++i) {
    // q-value update - Calculate temporal difference error
    _q_values(_trajectory_states(i), _trajectory_actions(i)) +=
        _alpha * (_payoff - _q_values(_trajectory_states(i), _trajectory_actions(i)));
    // Increase state-action counts
    ++_batch_counts(_trajectory_states(i), _trajectory_actions(i));
  }
}

void EGTTools::RL::algorithms::QLearning::reinforceTrajectory(size_t episode_length) {
  for (unsigned i = 0; i < episode_length; ++i) {
    // q-value update - Calculate temporal difference error
    _q_values(_trajectory_states(i), _trajectory_actions(i)) +=
        _alpha * (_payoff - _q_values(_trajectory_states(i), _trajectory_actions(i)));
    // Increase state-action counts
    ++_batch_counts(_trajectory_states(i), _trajectory_actions(i));
  }
}

void EGTTools::RL::algorithms::QLearning::resetQValues() {
  _q_values.setRandom();
}
