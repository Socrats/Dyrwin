//
// Created by Elias Fernandez on 2019-04-10.
//

#ifndef DYRWIN_RL_BATCHQLEARNINGAGENT_H
#define DYRWIN_RL_BATCHQLEARNINGAGENT_H

#include <cmath>
#include <Dyrwin/RL/Agent.h>

namespace EGTTools::RL {
    class BatchQLearningAgent : public Agent {
    public:
        BatchQLearningAgent(unsigned nb_rounds, unsigned nb_actions, unsigned endowment, double alpha,
                            double temperature)
                : Agent(
                nb_rounds, nb_actions, endowment), _alpha(alpha), _temperature(temperature) {};

        BatchQLearningAgent(const BatchQLearningAgent &other) :
                Agent(other.nb_rounds(), other.nb_actions(), other.endowment()), _alpha(other.alpha()),
                _temperature(other.temperature()) {}

        void reinforceTrajectory() override;

        bool inferPolicy() override;

        /**
        * Roth-Erev already has a discount factor, so it's not necessary to reinitialize
        * the weights.
        */
        void resetQValues() override {

        }

        // Getters
        double alpha() const { return _alpha; }

        double temperature() const { return _temperature; }

        // Setters
        void setAlpha(const double alpha) {
            if (alpha <= 0.0 || alpha > 1.0) throw std::invalid_argument("Learning rate parameter must be in (0,1]");
            _alpha = alpha;
        }

        void setTemperature(const double temperature) {
            if (temperature < 0.0) throw std::invalid_argument("temperature must be > 0");
            _temperature = temperature;
        }

    private:
        double _alpha; // learning rate
        double _temperature; // temperature of the boltzman distribution

    };
}

#endif //DYRWIN_RL_BATCHQLEARNINGAGENT_H
