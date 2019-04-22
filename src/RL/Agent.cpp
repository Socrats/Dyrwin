//
// Created by Elias Fernandez on 2019-03-11.
//

#include <Dyrwin/RL/Agent.h>

using namespace EGTTools::RL;


/**
 * Defines the copy constructor.
 * @param other
 */
Agent::Agent(const Agent &other) {
    _nb_rounds = other.nb_rounds();
    _nb_actions = other.nb_actions();
    _endowment = other.endowment();
    _payoff = other.payoff();
    _policy = other.policy().replicate(1, 1);
    _q_values = other.qValues().replicate(1, 1);
    _trajectory = VectorXui::Zero(_nb_rounds);
}

/**
 * Overloads = operator. Copies an agent into another already instantiated.
 * @param other
 * @return True if the agents are equal
 */
Agent &Agent::operator=(const Agent &other) {
    _nb_rounds = other.nb_rounds();
    _nb_actions = other.nb_actions();
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
    for (unsigned i = 0; i < _nb_rounds; i++) {
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
 * Resets the trajectory vector.
 * @return
 */
void Agent::resetTrajectory() {
    _trajectory.setZero();
}

/**
 * @brief Reinforce the Q-values of the last trajectory.
 * resets the trajectory afterwards
 * @return
 */
void Agent::reinforceTrajectory() {
    for (unsigned i = 0; i < _nb_rounds; i++) {
        _q_values(i, _trajectory(i)) += 1.0;
    }
    resetTrajectory();
}

/**
 * Samples an action from a discrete probability distribution defined in _policy for a given round.
 * @param round
 * @return selected action
 */
size_t Agent::selectAction(size_t round) {
    assert(round >= 0 && round < _nb_rounds);
//    double p = probabilityDistribution(_mt);
//    double container;
//    for ( size_t i = 0; i < _nb_actions; ++i) {
//        container = _policy(round, i);
//        if (container > p) {
//            _trajectory(round) = i;
//            break;
//        }
//        p -= container;
//    }
    _trajectory(round) = EGTTools::choice(_nb_actions, _policy.row(round), _mt);
    return _trajectory(round);
}

void Agent::resetQValues() {
    _q_values.setZero();
}
