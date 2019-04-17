//
// Created by Elias Fernandez on 2019-04-10.
//

#ifndef DYRWIN_QLEARNINGAGENT_H
#define DYRWIN_QLEARNINGAGENT_H

#include <math.h>
#include "Agent.h"

namespace EGTTools::RL {
    class QLearningAgent : public Agent {
    public:
        QLearningAgent(unsigned nb_rounds, unsigned nb_actions, unsigned endowment, double alpha,
                            double lambda, double temperature)
                : Agent(
                nb_rounds, nb_actions, endowment), _alpha(alpha), _lambda(lambda), _temperature(temperature) {};

        QLearningAgent(const QLearningAgent &other) :
                Agent(other.nb_rounds(), other.nb_actions(), other.endowment()), _alpha(other.alpha()),
                _lambda(other.lambda()), _temperature(other.temperature()) {}

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

        double lambda() const { return _lambda; }

        double temperature() const { return _temperature; }

        // Setters
        void setAlpha(const double alpha) {
            if (alpha <= 0.0 || alpha > 1.0) throw std::invalid_argument("Learning rate parameter must be in (0,1]");
            _alpha = alpha;
        }

        void setLambda(const double lambda) {
            if (lambda <= 0.0 || lambda > 1.0)
                throw std::invalid_argument("Forgetting rate parameter must be in (0,1]");
            _lambda = lambda;
        }

        void setTemperature(const double temperature) {
            if (temperature < 0.0) throw std::invalid_argument("temperature must be > 0");
            _temperature = temperature;
        }

    private:
        double _alpha; // learning rate
        double _lambda; // discount factor
        double _temperature; // temperature of the boltzman distribution

    };
}

#endif //DYRWIN_QLEARNINGAGENT_H
