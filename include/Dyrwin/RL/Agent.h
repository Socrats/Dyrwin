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
        Agent(unsigned int nb_rounds, unsigned int nb_actions, double endowment) : _nb_rounds(nb_rounds),
                                                                                   _nb_actions(nb_actions),
                                                                                   _endowment(endowment),
                                                                                   _payoff(endowment) {

            _q_values = Matrix2D::Random(_nb_rounds, _nb_actions);
            // Initialise all actions with equal probability
            _policy = Matrix2D::Constant(_nb_rounds, _nb_actions, 1.0 / static_cast<double>(_nb_actions));
            // Initialise trajectory (the actions taken at each round)
            _trajectory = VectorXui::Zero(_nb_rounds);
            _buffer = std::vector<size_t>(_nb_actions);

        };

        Agent(const Agent &other);

        virtual ~Agent() = default;

        Agent &operator=(const Agent &other);

        friend std::ostream &operator<<(std::ostream &o, Agent &r) { return r.display(o); }

        bool decrease(unsigned amount);

        virtual bool inferPolicy(); // infer probabilities from the count vector.

        void resetTrajectory();

        virtual void reinforceTrajectory();

        virtual size_t selectAction(size_t round);

        virtual void resetQValues();

        // Getters
        unsigned int nb_rounds() const { return _nb_rounds; }

        unsigned int nb_actions() const { return _nb_actions; }

        double endowment() const { return _endowment; }

        double payoff() const { return _payoff; }

        Matrix2D policy() const { return _policy; }

        Matrix2D qValues() const { return _q_values; }

        // Setters
        void set_nb_rounds(unsigned nb_rounds) { _nb_rounds = nb_rounds; }

        void set_nb_actions(unsigned nb_actions) { _nb_actions = nb_actions; }

        void set_endowment(double endowment) { _endowment = endowment; }

        void set_payoff(double payoff) { _payoff = payoff; }

        void resetPayoff() { _payoff = _endowment; }

        void set_policy(Matrix2D policy) { _policy = std::move(policy); }

        void set_q_values(Matrix2D q_values) { _q_values = std::move(q_values); }


    protected:
        virtual std::ostream &display(std::ostream &os) const;

        unsigned int _nb_rounds, _nb_actions;
        double _endowment, _payoff;

        Matrix2D _policy;
        Matrix2D _q_values;
        VectorXui _trajectory;
        std::vector<size_t> _buffer;

        // Random generators
        std::mt19937_64 _mt{EGTTools::Random::SeedGenerator::getInstance().getSeed()};
    };
}


#endif //DYRWIN_RL_AGENT_H
