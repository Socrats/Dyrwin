//
// Created by Elias Fernandez on 2019-04-10.
//

#include <Dyrwin/RL/ESARSA.h>

using namespace EGTTools::RL;

ESARSA::ESARSA(size_t nb_states, size_t nb_actions, size_t episode_length, double endowment,
               double alpha, double lambda, double temperature) : Agent(nb_states, nb_actions,
                                                                        episode_length, endowment),
                                                                  _alpha(alpha), _lambda(lambda),
                                                                  _temperature(temperature) {}

void ESARSA::reinforceTrajectory() {
  _q_values(_trajectory_states(_episode_length - 1), _trajectory_actions(_episode_length - 1)) +=
      _alpha * (_payoff - _q_values(_trajectory_states(_episode_length - 1),
                                    _trajectory_actions(_episode_length - 1)));
  for (size_t i = _episode_length - 1; i > 0; i--) {
    double expected_value = (_policy.row(i).transpose() * _q_values.row(i))(0);
    _q_values(_trajectory_states(i - 1), _trajectory_actions(i - 1)) += _alpha *
        (_lambda * expected_value -
            _q_values(_trajectory_states(i - 1), _trajectory_actions(i - 1)));
  }
}

void ESARSA::reinforceTrajectory(size_t episode_length) {
  _q_values(_trajectory_states(episode_length - 1), _trajectory_actions(episode_length - 1)) +=
      _alpha * (_payoff - _q_values(_trajectory_states(episode_length - 1),
                                    _trajectory_actions(episode_length - 1)));
  for (size_t i = episode_length - 1; i > 0; i--) {
    double expected_value = (_policy.row(i).transpose() * _q_values.row(i))(0);
    _q_values(_trajectory_states(i - 1), _trajectory_actions(i - 1)) += _alpha *
        (_lambda * expected_value -
            _q_values(_trajectory_states(i - 1), _trajectory_actions(i - 1)));
  }
}

bool ESARSA::inferPolicy() {
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

std::string ESARSA::type() const { return "EGTTools::RL::ESARSA"; }

// Getters
double ESARSA::alpha() const { return _alpha; }

double ESARSA::lambda() const { return _lambda; }

double ESARSA::temperature() const { return _temperature; }

// Setters
void ESARSA::setAlpha(double alpha) {
  if (alpha <= 0.0 || alpha > 1.0) throw std::invalid_argument("Learning rate parameter must be in (0,1]");
  _alpha = alpha;
}

void ESARSA::setLambda(double lambda) {
  if (lambda <= 0.0 || lambda > 1.0)
    throw std::invalid_argument("Forgetting rate parameter must be in (0,1]");
  _lambda = lambda;
}

void ESARSA::setTemperature(double temperature) {
  if (temperature < 0.0) throw std::invalid_argument("temperature must be > 0");
  _temperature = temperature;
}
