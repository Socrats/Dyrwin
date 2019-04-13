//
// Created by Elias Fernandez on 2019-04-10.
//

#ifndef DYRWIN_BATCHQLEARNINGAGENT_H
#define DYRWIN_BATCHQLEARNINGAGENT_H

#include <math.h>
#include "Agent.h"

namespace egt_tools {
    class BatchQLearningAgent : public Agent {
    public:
        BatchQLearningAgent(unsigned nb_rounds, unsigned nb_actions, unsigned endowment, double alpha, double temperature)
                : Agent(
                nb_rounds, nb_actions, endowment), _alpha(alpha), _temperature(temperature) {};

        BatchQLearningAgent(const BatchQLearningAgent &other) :
                Agent(other.nb_rounds(), other.nb_actions(), other.endowment()), _alpha(other.alpha()),
                _temperature(other.temperature()) {}

        void reinforceTrajectory() override;

        bool inferPolicy() override;

        double alpha() const { return _alpha; }

        double temperature() const { return _temperature; }

    private:
        double _alpha; // learning rate
        double _temperature; // temperature of the boltzman distribution

    };
}

#endif //DYRWIN_BATCHQLEARNINGAGENT_H
