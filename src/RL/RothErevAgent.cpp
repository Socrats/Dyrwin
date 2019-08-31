//
// Created by Elias Fernandez on 2019-05-14.
//

#include <Dyrwin/RL/RothErevAgent.h>

EGTTools::RL::RothErevAgent::RothErevAgent(size_t nb_states, size_t nb_actions, size_t episode_length, double endowment,
                                           double lambda, double epsilon,
                                           double temperature) : Agent(nb_states, nb_actions, episode_length,
                                                                       endowment), _lambda(1 - lambda),
                                                                 _temperature(temperature) {
    _epsilon_others = epsilon / (_nb_actions - 1);
    _epsilon = 1 - epsilon;
}

void EGTTools::RL::RothErevAgent::reinforceTrajectory() {
    _q_values.array() *= _lambda;
    for (size_t i = 0; i < _episode_length; ++i) {
        _q_values(_trajectory_states(i), _trajectory_actions(i)) += _payoff * _epsilon;
        for (size_t j = 0; j < _nb_actions; ++j) {
            if (j == _trajectory_actions(i)) continue;
            _q_values(_trajectory_states(i), j) += _payoff * _epsilon_others;
        }
        _trajectory_states(i) = 0;
        _trajectory_actions(i) = 0;
    }
}


void EGTTools::RL::RothErevAgent::reinforceTrajectory(size_t episode_length) {
    _q_values.array() *= _lambda;
    for (size_t i = 0; i < episode_length; ++i) {
        _q_values(_trajectory_states(i), _trajectory_actions(i)) += _payoff * _epsilon;
        for (size_t j = 0; j < _nb_actions; ++j) {
            if (j == _trajectory_actions(i)) continue;
            _q_values(_trajectory_states(i), j) += _payoff * _epsilon_others;
        }
        _trajectory_states(i) = 0;
        _trajectory_actions(i) = 0;
    }
}

bool EGTTools::RL::RothErevAgent::inferPolicy() {
    unsigned int j;

    for (unsigned i = 0; i < _nb_states; i++) {
        // We calculate the sum of exponential(s) of q values for each state
        double total = 0.;
        unsigned nb_infs = 0;
        for (j = 0; j < _nb_actions; j++) {
            _policy(i, j) = std::exp(_q_values(i, j) * _temperature);
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
void EGTTools::RL::RothErevAgent::resetQValues() {

}

std::string EGTTools::RL::RothErevAgent::type() const { return "EGTTools::RL::RothErevAgent"; }

// Getters
double EGTTools::RL::RothErevAgent::lambda() const { return 1 - _lambda; }

double EGTTools::RL::RothErevAgent::epsilon() const { return 1 - _epsilon; }

double EGTTools::RL::RothErevAgent::temperature() const { return _temperature; }

// Setters
void EGTTools::RL::RothErevAgent::setLambda(double lambda) {
    if (lambda <= 0.0 || lambda > 1.0)
        throw std::invalid_argument("Forgetting rate parameter must be in (0,1]");
    _lambda = lambda;
}

void EGTTools::RL::RothErevAgent::setEpsilon(double epsilon) {
    if (epsilon <= 0.0 || epsilon > 1.0)
        throw std::invalid_argument("Local experimentation parameter must be in (0,1]");
    _epsilon_others = epsilon / (_nb_actions - 1);
    _epsilon = 1 - epsilon;
}

void EGTTools::RL::RothErevAgent::setTemperature(double temperature) {
    if (temperature < 0.0) throw std::invalid_argument("temperature must be > 0");
    _temperature = temperature;
}