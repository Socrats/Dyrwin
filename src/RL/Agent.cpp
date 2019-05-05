//
// Created by Elias Fernandez on 2019-03-11.
//

#include <Dyrwin/RL/Agent.h>

using namespace EGTTools::RL;


/**
 * Overloads = operator. Copies an agent into another already instantiated.
 * @param other
 * @return True if the agents are equal
 */
Agent &Agent::operator=(const Agent &other) {
    _nb_states = other.nb_states();
    _nb_actions = other.nb_actions();
    _episode_length = other.episode_length();
    _endowment = other.endowment();
    _payoff = other.payoff();
    _policy = other.policy();
    _q_values = other.qValues();

    return *this;
}

/**
 * Decreases the payoff by @param amount if _payoff >= amount and returns true.
 * Otherwise returns false.
 * @param amount
 * @return bool
 */
bool Agent::decrease(unsigned amount) {
    if ((double(_payoff) - amount) >= 0) {
        _payoff -= amount;
        return true;
    }
    return false;
}

/**
 * Displays the current payoff and policy of the agent.
 * @param os std::ostream object reference
 * @return std::ostream
 */
std::ostream &Agent::display(std::ostream &os) const {
    Eigen::IOFormat CleanFmt(4, 0, ", ", "\n", "[", "]");

    os << "[(" << _payoff << ")";
    os << _policy.format(CleanFmt);
    os << "]" << std::endl;
    return os;
}

/**
 * Infers the agent's policy from the Q-values.
 * Returns false if there is an error.
 * @return bool
 */
bool Agent::inferPolicy() {
    for (unsigned i = 0; i < _nb_states; i++) {
        double total = 0.;
        for (unsigned j = 0; j < _nb_actions; j++) {
            total += _q_values(i, j);
        }
        if (total == 0.) {
            for (unsigned j = 0; j < _nb_actions; j++) {
                _policy(i, j) = 1. / (double) _nb_actions;
            }
        } else {
            double check = 0;
            for (unsigned j = 0; j < (_nb_actions - 1); j++) {
                _policy(i, j) = _q_values(i, j) / total;
                check += _policy(i, j);
            }
            _policy(i, _nb_actions - 1) = 1.0 - check;
            assert(check <= 1.0);
        }
    }
    return true;
}

/**
 * @brief Resets the trajectory vector.
 */
void Agent::resetTrajectory() {
    for (unsigned i = 0; i < _episode_length; ++i) {
        _trajectory_states(i) = 0;
        _trajectory_actions(i) = 0;
    }
}

/**
 * @brief Reinforce the Q-values of the last trajectory.
 * resets the trajectory afterwards
 */
void Agent::reinforceTrajectory() {
    for (unsigned i = 0; i < _episode_length; ++i) {
        _q_values(_trajectory_states(i), _trajectory_actions(i)) += 1.0;
        _trajectory_states(i) = 0;
        _trajectory_actions(i) = 0;
    }
}

/**
 * @brief Reinforce the Q-values of the last trajectory.
 * resets the trajectory afterwards
 */
void Agent::reinforceTrajectory(size_t episode_length) {
    for (unsigned i = 0; i < episode_length; ++i) {
        _q_values(_trajectory_states(i), _trajectory_actions(i)) += 1.0;
        _trajectory_states(i) = 0;
        _trajectory_actions(i) = 0;
    }
}

/**
 * Samples an action from a discrete probability distribution defined in _policy for a given round.
 * @param round
 * @return selected action
 */
size_t Agent::selectAction(size_t round) {
    assert(round >= 0 && round < _episode_length);
    _trajectory_states(round) = round;
    _trajectory_actions(round) = EGTTools::choice(_nb_actions, _policy.row(round), _mt);
    return _trajectory_actions(round);
}

/**
 * Samples an action from a discrete probability distribution defined in _policy for a given round.
 * @param round
 * @return selected action
 */
size_t Agent::selectAction(size_t round, size_t state) {
    assert(round >= 0 && round < _episode_length);
    _trajectory_states(round) = state;
    _trajectory_actions(round) = EGTTools::choice(_nb_actions, _policy.row(state), _mt);
    return _trajectory_actions(round);
}

void Agent::resetQValues() {
    _q_values.setZero();
}
