//
// Created by Elias Fernandez on 2019-04-10.
//

#include <Dyrwin/RL/SARSAAgent.h>

using namespace EGTTools::RL;

SARSAAgent::SARSAAgent(size_t nb_states, size_t nb_actions, size_t episode_length, double endowment,
                       double alpha, double lambda, double temperature) : Agent(nb_states, nb_actions,
                                                                                episode_length, endowment),
                                                                          _alpha(alpha), _lambda(lambda),
                                                                          _temperature(temperature) {}

void SARSAAgent::reinforceTrajectory() {
  int last = static_cast<int>(_episode_length) - 1;
  _q_values(_trajectory_states(last),
            _trajectory_actions(last)) +=
      _alpha * (_payoff - _q_values(_trajectory_states(last),
                                    _trajectory_actions(last)));
  for (int i = last - 1; i >= 0; i--) {
    _q_values(_trajectory_states(i),
              _trajectory_actions(i)) += _alpha *
        (_payoff + _lambda * _q_values(_trajectory_states(i + 1),
                                       _trajectory_actions(i + 1)) - _q_values(_trajectory_states(i),
                                                                               _trajectory_actions(i)));
  }
}

void SARSAAgent::reinforceTrajectory(size_t episode_length) {
  int last = static_cast<int>(episode_length) - 1;
  _q_values(_trajectory_states(last),
            _trajectory_actions(last)) +=
      _alpha * (_payoff - _q_values(_trajectory_states(last),
                                    _trajectory_actions(last)));
  for (int i = last - 1; i >= 0; i--) {
    _q_values(_trajectory_states(i), _trajectory_actions(i)) += _alpha *
        (_payoff + _lambda * _q_values(_trajectory_states(i + 1),
                                       _trajectory_actions(i + 1)) - _q_values(_trajectory_states(i),
                                                                               _trajectory_actions(i)));
  }
}

bool SARSAAgent::inferPolicy() {
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

std::string SARSAAgent::type() const { return "EGTTools::RL::QLearningAgent"; }

// Getters
double SARSAAgent::alpha() const { return _alpha; }

double SARSAAgent::lambda() const { return _lambda; }

double SARSAAgent::temperature() const { return _temperature; }

// Setters
void SARSAAgent::setAlpha(double alpha) {
  if (alpha <= 0.0 || alpha > 1.0) throw std::invalid_argument("Learning rate parameter must be in (0,1]");
  _alpha = alpha;
}

void SARSAAgent::setLambda(double lambda) {
  if (lambda <= 0.0 || lambda > 1.0)
    throw std::invalid_argument("Forgetting rate parameter must be in (0,1]");
  _lambda = lambda;
}

void SARSAAgent::setTemperature(double temperature) {
  if (temperature < 0.0) throw std::invalid_argument("temperature must be > 0");
  _temperature = temperature;
}
