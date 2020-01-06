//
// Created by Elias Fernandez on 2019-04-10.
//

#include <Dyrwin/RL/BatchQLearningAgent.h>

using namespace EGTTools::RL;

BatchQLearningAgent::BatchQLearningAgent(size_t nb_states, size_t nb_actions, size_t episode_length, double endowment,
                                         double alpha, double temperature)
    : Agent(nb_states, nb_actions, episode_length, endowment), _alpha(alpha), _temperature(temperature) {
  // Initialise buffer for the batch updates
  _batch_updates = Matrix2D::Zero(_nb_states, _nb_actions);
  // Initialise to zero the state-action counts
  _batch_counts = Matrix2D::Zero(_nb_states, _nb_actions);
}

void BatchQLearningAgent::reinforceTrajectory() {
  // Now it will calculate the temporal difference errors of the current trajectory
  // And will store them in the TD error buffer (_batch_updates). It will also
  // increase the counts of every visited state (_batch_count).
  // The reinforcement/update of q-values estimation should be
  // calculated at the end of the batch, just before the policy is inferred.
  for (unsigned i = 0; i < _episode_length; ++i) {
    // q-value update - Calculate temporal difference error
    _batch_updates(_trajectory_states(i), _trajectory_actions(i)) += _payoff -
        _q_values(_trajectory_states(i), _trajectory_actions(i));
    // Increase state-action counts
    ++_batch_counts(_trajectory_states(i), _trajectory_actions(i));
  }
}

void BatchQLearningAgent::reinforceTrajectory(size_t episode_length) {
// Now it will calculate the temporal difference errors of the current trajectory
  // And will store them in the TD error buffer (_batch_updates). It will also
  // increase the counts of every visited state (_batch_count).
  // The reinforcement/update of q-values estimation should be
  // calculated at the end of the batch, just before the policy is inferred.
  for (unsigned i = 0; i < episode_length; ++i) {
    // q-value update - Calculate temporal difference error
    _batch_updates(_trajectory_states(i), _trajectory_actions(i)) += _payoff -
        _q_values(_trajectory_states(i), _trajectory_actions(i));
    // Increase state-action counts
    ++_batch_counts(_trajectory_states(i), _trajectory_actions(i));
  }
}

bool BatchQLearningAgent::inferPolicy() {
  // Before the policy is calculated using the Gibbs-Boltzmann distribution,
  // we update the q-values estimation through the stored batch TD-errors.
  _batch_updates = (_batch_counts.array() != 0).select(_batch_updates.cwiseQuotient(_batch_counts),
                                                       _batch_updates);
  _q_values += _alpha * _batch_updates;
  // After we reinitialize both buffers
  _batch_updates.setZero();
  _batch_counts.setZero();
  // Then we infer the policy
  unsigned int j;

  for (unsigned i = 0; i < _nb_states; i++) {
    // We calculate the sum of exponential(s) of q values for each state
    double total = 0.;
    unsigned nb_infs = 0;
    for (j = 0; j < _nb_actions; j++) {
      _policy(i, j) = exp(_q_values(i, j) * _temperature);
      if (std::isinf(_policy(i, j))) _buffer[nb_infs++] = j;
      total += _policy(i, j);
    }
    if (nb_infs) {
      auto dist = std::uniform_int_distribution<unsigned>(0, nb_infs - 1);
      unsigned selection = dist(_mt);
      _policy.row(i).setZero();
      _policy(i, _buffer[selection]) = 1.0;
    } else {
      _policy.row(i).array() /= total;
    }
  }
  return true;
}

/**
       * Roth-Erev already has a discount factor, so it's not necessary to reinitialize
       * the weights.
       */
void BatchQLearningAgent::resetQValues() {
  _q_values.setRandom();
}

std::string BatchQLearningAgent::type() const {
  return "EGTTools::RL::BatchQLearningAgent";
}

// Getters
double BatchQLearningAgent::alpha() const { return _alpha; }

double BatchQLearningAgent::temperature() const { return _temperature; }

// Setters
void BatchQLearningAgent::setAlpha(double alpha) {
  if (alpha <= 0.0 || alpha > 1.0) throw std::invalid_argument("Learning rate parameter must be in (0,1]");
  _alpha = alpha;
}

void BatchQLearningAgent::setTemperature(double temperature) {
  if (temperature < 0.0) throw std::invalid_argument("temperature must be > 0");
  _temperature = temperature;
}
