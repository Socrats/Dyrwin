//
// Created by Elias Fernandez on 2019-04-10.
//

#ifndef DYRWIN_ROTHEREVAGENT_H
#define DYRWIN_ROTHEREVAGENT_H

#include <math.h>
#include "Agent.h"

namespace EGTTools::RL {
    class RothErevAgent : public Agent {
/**
 * Implements the Roth-Erev reinforcement learning with discount factor algorithm
 */
    public:
        RothErevAgent(unsigned nb_rounds, unsigned nb_actions, unsigned endowment, double lambda, double temperature) :
                Agent(nb_rounds, nb_actions, endowment), _lambda(lambda), _temperature(temperature) {};

        RothErevAgent(const RothErevAgent &other) : Agent(other.nb_rounds(), other.nb_actions(), other.endowment()),
                                                    _lambda(other.lambda()), _temperature(other.temperature()) {};

        /**
         * @brief Reinforces a given strategy accoring to the accumulated payoffs.
         *
         * Reinforces the propensity matrix given a trajectory (a set of actions taken in
         * a window of states) and the accumulated payoff over that window.
         *
         * This function updates the propensity in a batch.
         *
         * lambda is the forgetting rate
         */
        void reinforceTrajectory() override {
            _q_values.array() *= _lambda;
            for (size_t i = 0; i < _nb_rounds; ++i) {
                _q_values(i, _trajectory(i)) += _payoff;
            }
        }

        /**
         * @brief Infers the policy from the propensity matrix
         *
         * Uses a softmax (boltzman distribution) with a temperature parameter
         * to infer the policy.
         *
         * @return
         */
        bool inferPolicy() override {
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
                }
                else {
                    _policy.row(i).array() /= total;
                }
            }
            return true;
        }

        /**
        * Roth-Erev already has a discount factor, so it's not necessary to reinitialize
        * the weights.
        */
        void resetQValues() override {

        }

        // Getters
        double lambda() const { return _lambda; }

        double temperature() const { return _temperature; }

        // Setters
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
        double _lambda; // Forgetting rate
        double _temperature; // Temperature of the boltzman distribution
    };
}

#endif //DYRWIN_ROTHEREVAGENT_H
