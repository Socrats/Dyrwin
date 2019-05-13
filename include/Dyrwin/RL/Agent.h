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
        Agent(size_t nb_states, size_t nb_actions, size_t episode_length, double endowment) : _nb_states(nb_states),
                                                                                              _nb_actions(nb_actions),
                                                                                              _episode_length(
                                                                                                      episode_length),
                                                                                              _endowment(endowment),
                                                                                              _payoff(endowment) {

            _q_values = Matrix2D::Random(nb_states, _nb_actions);
            // Initialise all actions with equal probability
            _policy = Matrix2D::Constant(nb_states, _nb_actions, 1.0 / static_cast<double>(_nb_actions));
            // Initialise trajectory (the actions taken at each round)
            _trajectory_states = VectorXui::Zero(_episode_length);
            _trajectory_actions = VectorXui::Zero(_episode_length);
            _buffer = std::vector<size_t>(_nb_actions);

        };

        virtual ~Agent() = default;

        friend std::ostream &operator<<(std::ostream &o, Agent &r) { return r.display(o); }

        std::string toString() const {
            std::stringstream ss;
            Eigen::IOFormat CleanFmt(4, 0, ", ", "\n", "[", "]");

            ss << "[(" << _payoff << ")";
            ss << _policy.format(CleanFmt);
            ss << "]" << std::endl;
            return ss.str();
        }

        virtual std::string type() const { return "EGTTools::RL::Agent"; }

        bool decrease(unsigned amount);

        virtual bool inferPolicy(); // infer probabilities from the count vector.

        void resetTrajectory();

        virtual void reinforceTrajectory();

        virtual void reinforceTrajectory(size_t episode_length);

        virtual size_t selectAction(size_t round);

        virtual size_t selectAction(size_t round, size_t state);

        virtual void resetQValues();

        virtual void reset();

        // Getters
        size_t nb_states() const { return _nb_states; }

        size_t nb_actions() const { return _nb_actions; }

        size_t episode_length() const { return _episode_length; }

        double endowment() const { return _endowment; }

        double payoff() const { return _payoff; }

        const VectorXui &trajectoryStates() const { return _trajectory_states; };

        const VectorXui &trajectoryActions() const { return _trajectory_actions; };

        const Matrix2D &policy() const { return _policy; }

        const Matrix2D &qValues() const { return _q_values; }

        // Setters
        void set_nb_states(size_t nb_states) { _nb_states = nb_states; }

        void set_nb_actions(size_t nb_actions) { _nb_actions = nb_actions; }

        void set_episode_length(size_t episode_length) { _episode_length = episode_length; }

        void set_endowment(double endowment) { _endowment = endowment; }

        void set_payoff(double payoff) { _payoff = payoff; }

        void resetPayoff() { _payoff = _endowment; }

        void set_policy(const Eigen::Ref<const Matrix2D> &policy) {
            if (static_cast<size_t>(policy.rows()) != _nb_states) throw std::invalid_argument(
                        "Policy must have as many rows as states (" + std::to_string(_nb_states) + ")");
            if (static_cast<size_t>(policy.cols()) != _nb_actions) throw std::invalid_argument(
                        "Policy must have as many columns as actions (" + std::to_string(_nb_actions) + ")");
            _policy.array() = policy;
        }

        void set_q_values(const Eigen::Ref<const Matrix2D> &q_values) {
            if (static_cast<size_t>(q_values.rows()) != _nb_states) throw std::invalid_argument(
                        "Q-values must have as many rows as states (" + std::to_string(_nb_states) + ")");
            if (static_cast<size_t>(q_values.cols()) != _nb_actions) throw std::invalid_argument(
                        "Q-values must have as many columns as actions (" + std::to_string(_nb_actions) + ")");
            _q_values.array() = q_values;
        }


    protected:
        virtual std::ostream &display(std::ostream &os) const;

        size_t _nb_states, _nb_actions, _episode_length;
        double _endowment, _payoff;

        Matrix2D _policy;
        Matrix2D _q_values;
        VectorXui _trajectory_states, _trajectory_actions;
        std::vector<size_t> _buffer;

        // Random generators
        std::mt19937_64 _mt{EGTTools::Random::SeedGenerator::getInstance().getSeed()};
    };
}


#endif //DYRWIN_RL_AGENT_H
