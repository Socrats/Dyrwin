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
        [[nodiscard]] std::string type() const override;

        // Getters
        [[nodiscard]] double alpha() const override;

        [[nodiscard]] double temperature() const;

        // Setters
        void setAlpha(double alpha) override;

        void decreaseAlpha(double decrease_rate) override ;

        void setTemperature(double temperature);

        void increaseTemperature(double increase_rate) override ;

    private:
        double _alpha; // learning rate
        double _temperature; // temperature of the boltzman distribution

    };
}

#endif //DYRWIN_RL_BATCHQLEARNINGAGENT_H
