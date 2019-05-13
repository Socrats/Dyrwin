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
                            double temperature)
                : Agent(nb_states, nb_actions, episode_length, endowment), _alpha(alpha), _temperature(temperature) {};

        void reinforceTrajectory() override;

        void reinforceTrajectory(size_t episode_length) override;

        bool inferPolicy() override;

        /**
        * Roth-Erev already has a discount factor, so it's not necessary to reinitialize
        * the weights.
        */
        void resetQValues() override {

        }

        virtual std::string type() const override { return "EGTTools::RL::BatchQLearningAgent"; }

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
