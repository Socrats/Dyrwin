//
// Created by Elias Fernandez on 2019-04-10.
//

#include <Dyrwin/RL/BatchQLearningForgetAgent.h>

using namespace EGTTools::RL;

BatchQLearningForgetAgent::BatchQLearningForgetAgent(size_t nb_states,
                                                     size_t nb_actions,
                                                     size_t episode_length,
                                                     double endowment,
                                                     double lambda,
                                                     double alpha,
                                                     double temperature)
    : Agent(nb_states,
        nb_actions,
        episode_length,
        endowment),
        _alpha(alpha),
        _lambda(lambda),
        _temperature(temperature) {}

void BatchQLearningForgetAgent::reinforceTrajectory() {
  for (unsigned i = 0; i < _episode_length; ++i) {
    for (unsigned j = 0; j < _nb_actions; ++j) {
      if (j == _trajectory_actions(i)) {
        _q_values(_trajectory_states(i), _trajectory_actions(i)) +=
            _alpha * (_payoff - _q_values(_trajectory_states(i), _trajectory_actions(i)));
      } else {
        // forgetting the other actions
        _q_values(_trajectory_states(i), j) *= _lambda;
      }
    }
  }
}

void BatchQLearningForgetAgent::reinforceTrajectory(size_t episode_length) {
  for (unsigned i = 0; i < episode_length; ++i) {
    for (unsigned j = 0; j < _nb_actions; ++j) {
      if (j == _trajectory_actions(i)) {
        _q_values(_trajectory_states(i), _trajectory_actions(i)) +=
            _alpha * (_payoff - _q_values(_trajectory_states(i), _trajectory_actions(i)));
      } else {
        // forgetting the other actions
        _q_values(_trajectory_states(i), j) *= _lambda;
      }
    }
  }
}

bool BatchQLearningForgetAgent::inferPolicy() {
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
void BatchQLearningForgetAgent::resetQValues() {
  _q_values.setZero();
}

std::string BatchQLearningForgetAgent::type() const {
  return "EGTTools::RL::BatchQLearningAgent";
}

// Getters
double BatchQLearningForgetAgent::alpha() const { return _alpha; }

double BatchQLearningForgetAgent::lambda() const { return _lambda; }

double BatchQLearningForgetAgent::temperature() const { return _temperature; }

// Setters
void BatchQLearningForgetAgent::setAlpha(double alpha) {
  if (alpha <= 0.0 || alpha > 1.0) throw std::invalid_argument("Learning rate parameter must be in (0,1]");
  _alpha = alpha;
}

void BatchQLearningForgetAgent::setLambda(double lambda) {
  if (lambda <= 0.0 || lambda > 1.0) throw std::invalid_argument("Forgetting rate parameter must be in (0,1]");
  _lambda = lambda;
}

void BatchQLearningForgetAgent::setTemperature(double temperature) {
  if (temperature < 0.0) throw std::invalid_argument("temperature must be > 0");
  _temperature = temperature;
}
