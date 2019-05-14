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
        BatchQLearningAgent(size_t nb_states, size_t nb_actions, size_t episode_length, double endowment, double alpha,
                            double temperature);

        void reinforceTrajectory() override;

        void reinforceTrajectory(size_t episode_length) override;

        bool inferPolicy() override;

        /**
        * Roth-Erev already has a discount factor, so it's not necessary to reinitialize
        * the weights.
        */
        void resetQValues() override;

        /**
         * @briefs Returns a string indicating the agent's class name
         * @return Returns a string indicating the agent's class name
         */
        std::string type() const override;

        // Getters
        double alpha() const;

        double temperature() const;

        // Setters
        void setAlpha(double alpha);

        void setTemperature(double temperature);

    private:
        double _alpha; // learning rate
        double _temperature; // temperature of the boltzman distribution

    };
}

#endif //DYRWIN_RL_BATCHQLEARNINGAGENT_H
