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
                      double temperature);

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
        void reinforceTrajectory() override;

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
        void reinforceTrajectory(size_t episode_length) override;

        /**
         * @brief Infers the policy from the propensity matrix
         *
         * Uses a softmax (boltzman distribution) with a temperature parameter
         * to infer the policy.
         *
         * @return
         */
        bool inferPolicy() override;

        /**
        * Roth-Erev already has a discount factor, so it's not necessary to reinitialize
        * the weights.
        */
        void resetQValues() override;

        void reset() override ;

        std::string type() const override;

        // Getters
        double lambda() const;

        double temperature() const;

        // Setters
        void setLambda(double lambda);

        void setTemperature(double temperature);

    private:
        double _lambda; // Forgetting rate
        double _temperature; // Temperature of the boltzman distribution
    };
}

#endif //DYRWIN_RL_ROTHEREVAGENT_H
