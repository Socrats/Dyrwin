//
// Created by Elias Fernandez on 2019-03-11.
//

#ifndef DYRWIN_RL_AGENT_H
#define DYRWIN_RL_AGENT_H

#include <string>
#include <cmath>
#include <iostream>
#include <cassert>
#include <vector>
#include <Dyrwin/Distributions.h>
#include <Dyrwin/Types.h>
#include <Dyrwin/SeedGenerator.h>


namespace EGTTools::RL {
    class Agent {
/**
     * Implements a version of Roth-Erev learning where each sequence of state-actions
     * that lead to achieving the target is reinforced with 1.0.
     */
    public:
        Agent(size_t nb_states, size_t nb_actions, size_t episode_length, double endowment);

        virtual ~Agent() = default;

        friend std::ostream &operator<<(std::ostream &o, Agent &r) { return r.display(o); }

        std::string toString() const;

        virtual std::string type() const { return "EGTTools::RL::Agent"; }

        bool decrease(double amount);

        virtual bool inferPolicy(); // infer probabilities from the count vector.

        void resetTrajectory();

        virtual void reinforceTrajectory();

        virtual void reinforceTrajectory(size_t episode_length);

        virtual size_t selectAction(size_t round);

        virtual size_t selectAction(size_t round, size_t state);

        virtual void resetQValues();

        virtual void reset();

        void add2payoff(double value);

        // Getters
        size_t nb_states() const;

        size_t nb_actions() const;

        size_t episode_length() const;

        double endowment() const;

        double payoff() const;

        const VectorXui &trajectoryStates() const;

        const VectorXui &trajectoryActions() const;

        const Matrix2D &policy() const;

        const Matrix2D &qValues() const;

        // Setters
        void set_nb_states(size_t nb_states);

        void set_nb_actions(size_t nb_actions);

        void set_episode_length(size_t episode_length);

        void set_endowment(double endowment);

        void set_payoff(double payoff);

        void resetPayoff();

        void set_policy(const Eigen::Ref<const Matrix2D> &policy);

        void set_q_values(const Eigen::Ref<const Matrix2D> &q_values);

        void set_trajectory_round(size_t round, size_t action);


    protected:
        virtual std::ostream &display(std::ostream &os) const;

        size_t _nb_states, _nb_actions, _episode_length;
        double _endowment, _payoff;

        Matrix2D _policy;
        Matrix2D _q_values;
        VectorXui _trajectory_states, _trajectory_actions;
        std::vector<size_t> _buffer;
        std::uniform_real_distribution<double> _real_rand;

        // Random generators
        std::mt19937_64 _mt{EGTTools::Random::SeedGenerator::getInstance().getSeed()};
    };
}


#endif //DYRWIN_RL_AGENT_H
