//
// Created by Elias Fernandez on 2019-04-10.
//

#ifndef DYRWIN_RL_DISCOUNTEDQLEARNING_H
#define DYRWIN_RL_DISCOUNTEDQLEARNING_H

#include <cmath>
#include <Dyrwin/RL/Agent.h>

namespace EGTTools::RL {
    class DiscountedQLearning : public Agent {
    public:
        DiscountedQLearning(size_t nb_states, size_t nb_actions, size_t episode_length,
                            double endowment, double alpha, double lambda,
                            double temperature);

        void reinforceTrajectory() override;

        void reinforceTrajectory(size_t episode_length) override;

        bool inferPolicy() override;

        /**
        * @brief Q value is resetted to random values
        */
        void resetQValues() override;

        /**
         * @brief Returns a string indicating the agent's class name
         * @return Returns a string indicating the agent's class name
         */
        [[nodiscard]] std::string type() const override;

        // Getters
        [[nodiscard]] double alpha() const;
        [[nodiscard]] double lambda() const;
        [[nodiscard]] double temperature() const;

        // Setters
        void setAlpha(double alpha);
        void setLambda(double lambda);
        void setTemperature(double temperature);

    private:
        double _alpha; // learning rate
        double _lambda; // forgetting factor (or discount rate)
        double _temperature; // temperature of the boltzman distribution
    };
}

#endif //DYRWIN_RL_DISCOUNTEDQLEARNING_H
