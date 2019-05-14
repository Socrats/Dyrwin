//
// Created by Elias Fernandez on 2019-05-14.
//

#include <Dyrwin/RL/HistericQLearningAgent.hpp>

EGTTools::RL::HistericQLearningAgent::HistericQLearningAgent(size_t nb_states, size_t nb_actions, size_t episode_length,
                                                             double endowment, double alpha, double beta,
                                                             double temperature) : Agent(nb_states, nb_actions,
                                                                                         episode_length, endowment),
                                                                                   _alpha(alpha), _beta(beta),
                                                                                   _temperature(temperature) {}

void EGTTools::RL::HistericQLearningAgent::reinforceTrajectory() {
    for (unsigned i = 0; i < _episode_length; i++) {
        const auto delta = _payoff - _q_values(_trajectory_states(i), _trajectory_actions(i));
        if (delta >= 0) {
            _q_values(_trajectory_states(i), _trajectory_actions(i)) += _alpha * delta;
        } else {
            _q_values(_trajectory_states(i), _trajectory_actions(i)) += _beta * delta;
        }
        // reset trace
        _trajectory_states(i) = 0;
        _trajectory_actions(i) = 0;
    }
}

void EGTTools::RL::HistericQLearningAgent::reinforceTrajectory(size_t episode_length) {
    for (unsigned i = 0; i < episode_length; i++) {
        const auto delta = _payoff - _q_values(_trajectory_states(i), _trajectory_actions(i));
        if (delta >= 0) {
            _q_values(_trajectory_states(i), _trajectory_actions(i)) += _alpha * delta;
        } else {
            _q_values(_trajectory_states(i), _trajectory_actions(i)) += _beta * delta;
        }
        // reset trace
        _trajectory_states(i) = 0;
        _trajectory_actions(i) = 0;
    }
}

bool EGTTools::RL::HistericQLearningAgent::inferPolicy() {
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

std::string EGTTools::RL::HistericQLearningAgent::type() const { return "EGTTools::RL::HistericQLearningAgent"; }

// Getters
double EGTTools::RL::HistericQLearningAgent::alpha() const { return _alpha; }

double EGTTools::RL::HistericQLearningAgent::beta() const { return _beta; }

double EGTTools::RL::HistericQLearningAgent::temperature() const { return _temperature; }

// Setters
void EGTTools::RL::HistericQLearningAgent::setAlpha(double alpha) {
    if (alpha <= 0.0 || alpha > 1.0) throw std::invalid_argument("Learning rate parameter must be in (0,1]");
    _alpha = alpha;
}

void EGTTools::RL::HistericQLearningAgent::setBeta(double beta) {
    if (beta <= 0.0 || beta > 1.0) throw std::invalid_argument("Learning rate parameter must be in (0,1]");
    _beta = beta;
}

void EGTTools::RL::HistericQLearningAgent::setTemperature(double temperature) {
    if (temperature < 0.0) throw std::invalid_argument("temperature must be > 0");
    _temperature = temperature;
}