//
// Created by Elias Fernandez on 2019-04-10.
//

#include "BatchQLearningAgent.h"

using namespace egt_tools;

void BatchQLearningAgent::reinforceTrajectory() {
    for (unsigned i = 0; i < _nb_rounds; i++) {
        _q_values(i, _trajectory(i)) += (_payoff - _q_values(i, _trajectory(i))) * _alpha;
    }
    resetTrajectory();
}

bool BatchQLearningAgent::inferPolicy() {
    unsigned int j;

    for (unsigned i = 0; i < _nb_rounds; i++) {
        double total = 0.;
        for (j = 0; j < _nb_actions; j++) {
            _policy(i, j) = exp(_q_values(i, j) * _temperature);
            total += _q_values(i, j);
        }
        double check = 0.;
        for (j = 0; j < _nb_actions; j++) {
            _policy(i, j) = _policy(i, j) / total;
            if (j == (_nb_actions - 1)) {
                _policy(i, j) = 1.0 - check;
            }
            check += _policy(i, j);
        }
        assert(check <= 1.0);
    }
    return true;
}
