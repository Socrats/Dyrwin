//
// Created by Elias Fernandez on 2019-04-10.
//

#ifndef DYRWIN_RL_ROTHEREVAGENT_H
#define DYRWIN_RL_ROTHEREVAGENT_H

#include <cmath>
#include <Dyrwin/RL/Agent.h>

namespace EGTTools::RL {
    class RothErevAgent : public Agent {
/**
 * Implements the Roth-Erev reinforcement learning with discount factor algorithm
 */
    public:
        RothErevAgent(size_t nb_states, size_t nb_actions, size_t episode_length, double endowment, double lambda,
                      double temperature) : Agent(nb_states, nb_actions, episode_length, endowment), _lambda(lambda),
                                            _temperature(temperature) {};

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
            for (size_t i = 0; i < _episode_length; ++i) {
                _q_values(_trajectory_states(i), _trajectory_actions(i)) += _payoff;
                _trajectory_states(i) = 0;
                _trajectory_actions(i) = 0;
            }
        }

        /**
         * @brief Reinforces a given strategy accoring to the accumulated payoffs.
         *
         * Reinforces the propensity matrix given a trajectory (a set of actions taken in
         * a window of states) and the accumulated payoff over that window.
         *
         * This function updates the propensity in a batch.
         *
         * lambda is the forgetting rate
         *
         * @param episode_length : length of the episode to reinforce
         */
        void reinforceTrajectory(size_t episode_length) override {
            _q_values.array() *= _lambda;
            for (size_t i = 0; i < episode_length; ++i) {
                _q_values(_trajectory_states(i), _trajectory_actions(i)) += _payoff;
                _trajectory_states(i) = 0;
                _trajectory_actions(i) = 0;
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
        void resetQValues() override {

        }

        virtual std::string type() const override { return "EGTTools::RL::RothErevAgent"; }

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

#endif //DYRWIN_RL_ROTHEREVAGENT_H
