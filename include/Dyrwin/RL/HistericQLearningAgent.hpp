//
// Created by Elias Fernandez on 2019-04-17.
//

#ifndef DYRWIN_RL_HISTERICQLEARNINGAGENT_HPP
#define DYRWIN_RL_HISTERICQLEARNINGAGENT_HPP

#include <Dyrwin/RL/Agent.h>

namespace EGTTools::RL {
    class HistericQLearningAgent : public Agent {
    public:
        HistericQLearningAgent(size_t nb_states, size_t nb_actions, size_t episode_length, double endowment,
                               double alpha, double beta, double temperature);


        void reinforceTrajectory() override;

        void reinforceTrajectory(size_t episode_length) override;

        bool inferPolicy() override;

        std::string type() const override;

        // Getters
        double alpha() const;

        double beta() const;

        double temperature() const;

        // Setters
        void setAlpha(double alpha);

        void setBeta(double beta);

        void setTemperature(double temperature);

    private:
        double _alpha, _beta, _temperature;
    };
}

#endif //DYRWIN_RL_HISTERICQLEARNINGAGENT_HPP
