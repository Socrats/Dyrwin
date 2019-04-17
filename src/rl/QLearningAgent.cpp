//
// Created by Elias Fernandez on 2019-04-10.
//

#include "../../include/Dyrwin/rl/QLearningAgent.h"

using namespace EGTTools::RL;

void QLearningAgent::reinforceTrajectory() {
    size_t i;
    for (i = 0; i < (_nb_rounds - 1); i++) {
        _q_values(i, _trajectory(i)) +=
                _alpha * (_lambda * _q_values.row(i + 1).maxCoeff() - _q_values(i, _trajectory(i)));
    }
    i = _nb_rounds - 1;
    _q_values(i, _trajectory(i)) += _alpha * (_payoff - _q_values(i, _trajectory(i)));
    resetTrajectory();
}

bool QLearningAgent::inferPolicy() {
    unsigned int j;

    for (unsigned i = 0; i < _nb_rounds; i++) {
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
