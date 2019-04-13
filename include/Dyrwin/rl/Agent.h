//
// Created by Elias Fernandez on 2019-03-11.
//

#ifndef DYRWIN_AGENT_H
#define DYRWIN_AGENT_H

#include <string>
#include <cmath>
#include <iostream>
#include <cassert>
#include <vector>
#include <Eigen/Dense>
#include "../SeedGenerator.h"
#include "../Distributions.h"
#include "RLUtils.h"


namespace egt_tools {
    class Agent {
/**
     * Implements a version of Roth-Erev learning where each sequence of state-actions
     * that lead to achieving the target is reinforced with 1.0.
     */
    public:
        Agent(unsigned nb_rounds, unsigned nb_actions, double endowment) : _nb_rounds(nb_rounds),
                                                                             _nb_actions(nb_actions),
                                                                             _endowment(endowment), _payoff(endowment) {

            _q_values = MatrixXd::Zero(_nb_rounds, _nb_actions);
            // Initialise all actions with equal probability
            _policy = MatrixXd::Constant(_nb_rounds, _nb_actions, 1.0 / static_cast<double>(_nb_actions));
            // Initialise trajectory (the actions taken at each round)
            _trajectory = Eigen::VectorXi::Zero(_nb_rounds);

        };

        Agent(const Agent &other);

        Agent &operator=(const Agent &other);

        friend std::ostream &operator<<(std::ostream &o, Agent &r) { return r.display(o); }

        bool decrease(unsigned amount);

        virtual bool inferPolicy(); // infer probabilities from the count vector.

        void resetTrajectory();

        virtual void reinforceTrajectory();

        virtual unsigned selectAction(unsigned round);

        virtual void resetQValues();

        // Getters
        unsigned nb_rounds() const { return _nb_rounds; }

        unsigned nb_actions() const { return _nb_actions; }

        double endowment() const { return _endowment; }

        double payoff() const { return _payoff; }

        MatrixXd policy() const { return _policy; }

        MatrixXd qValues() const { return _q_values; }

        // Setters
        void set_nb_rounds(unsigned nb_rounds) { _nb_rounds = nb_rounds; }

        void set_nb_actions(unsigned nb_actions) { _nb_actions = nb_actions; }

        void set_endowment(double endowment) { _endowment = endowment; }

        void set_payoff(double payoff) { _payoff = payoff; }

        void resetPayoff() { _payoff = _endowment; }

        void set_policy(MatrixXd policy) { _policy = std::move(policy); }

        void set_q_values(MatrixXd q_values) { _q_values = std::move(q_values); }


    protected:
        virtual std::ostream &display(std::ostream &os) const;

        unsigned _nb_rounds, _nb_actions;
        double _endowment, _payoff;

        MatrixXd _policy;
        MatrixXd _q_values;
        Eigen::VectorXi _trajectory;

        // Random generators
        std::mt19937_64 _mt{SeedGenerator::getInstance().getSeed()};
    };
}


#endif //DYRWIN_AGENT_H
