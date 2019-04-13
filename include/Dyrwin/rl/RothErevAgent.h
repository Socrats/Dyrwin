//
// Created by Elias Fernandez on 2019-04-10.
//

#ifndef DYRWIN_ROTHEREVAGENT_H
#define DYRWIN_ROTHEREVAGENT_H

#include <math.h>
#include "Agent.h"

namespace egt_tools {
    class RothErevAgent : Agent {
/**
 * Implements the Roth-Erev reinforcement learning with discount factor algorithm
 */
    public:
        RothErevAgent(unsigned actions, unsigned rounds, unsigned endowment, double lambda) :
                Agent(actions, rounds, endowment), _lambda(lambda) {};

        RothErevAgent(const RothErevAgent &other) : Agent(other.nb_rounds(), other.nb_actions(), other.endowment()),
                                                    _lambda(other.lambda()) {};

        void reinforceTrajectory() override {
            unsigned int i, j;
            for (i = 0; i < _nb_rounds; i++) {
                // Update chosen action at the given state
                _q_values(i, _trajectory(i)) += (_lambda * _q_values(i, _trajectory(i))) + _payoff;
                // Update other actions
                for (j = 0; j < _nb_actions; j++) {
                    if (j == (unsigned) _trajectory(i)) continue;
                    _q_values(i, j) = _lambda * _q_values(i, j);
                }

            }
        }

        void resetQValues() override {
        /**
        * Roth-Erev already has a discount factor, so it's not necessary to reinitialize
        * the weights.
        */
        }

        double lambda() const { return _lambda; }

    private:
        double _lambda; // Discount factor
    };
}

#endif //DYRWIN_ROTHEREVAGENT_H
