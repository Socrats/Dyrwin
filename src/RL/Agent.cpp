//
// Created by Elias Fernandez on 2019-03-11.
//

#include <Dyrwin/RL/Agent.h>

using namespace EGTTools::RL;


Agent::Agent(size_t nb_states, size_t nb_actions, size_t episode_length, double endowment) : _nb_states(nb_states),
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
    _real_rand = std::uniform_real_distribution<double>(0.0, 1.0);

}

std::string Agent::toString() const {
    std::stringstream ss;
    Eigen::IOFormat CleanFmt(4, 0, ", ", "\n", "[", "]");

    ss << "[(" << _payoff << ")";
    ss << _policy.format(CleanFmt);
    ss << "]" << std::endl;
    return ss.str();
}


/**
 * Decreases the payoff by @param amount if _payoff >= amount and returns true.
 * Otherwise returns false.
 * @param amount
 * @return bool
 */
bool Agent::decrease(double amount) {
    if ((_payoff - amount) >= 0) {
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
    }
}

/**
 * @brief Reinforce the Q-values of the last trajectory.
 * resets the trajectory afterwards
 */
void Agent::reinforceTrajectory(size_t episode_length) {
    for (unsigned i = 0; i < episode_length; ++i) {
        _q_values(_trajectory_states(i), _trajectory_actions(i)) += 1.0;
    }
}

/**
 * Samples an action from a discrete probability distribution defined in _policy for a given round.
 * @param round
 * @return selected action
 */
size_t Agent::selectAction(size_t round) {
    assert(round < _episode_length);
    _trajectory_states(round) = round;
    _trajectory_actions(round) = EGTTools::choice(_nb_actions, _policy.row(round), _real_rand, _mt);
    return _trajectory_actions(round);
}

/**
 * Samples an action from a discrete probability distribution defined in _policy for a given round.
 * @param round
 * @return selected action
 */
size_t Agent::selectAction(size_t round, size_t state) {
    assert(round < _episode_length);
    _trajectory_states(round) = state;
    _trajectory_actions(round) = EGTTools::choice(_nb_actions, _policy.row(state), _real_rand, _mt);
    return _trajectory_actions(round);
}

void Agent::resetQValues() {
    _q_values.setZero();
}

void Agent::reset() {
    _q_values.setRandom();
    // Initialise all actions with equal probability
    _policy.setConstant(1.0 / static_cast<double>(_nb_actions));
    // Initialise trajectory (the actions taken at each round)
    _trajectory_states.setZero();
    _trajectory_actions.setZero();
    _payoff = _endowment;
}

void Agent::add2payoff(double value) {
    _payoff += value;
}

// Getters
size_t Agent::nb_states() const { return _nb_states; }

size_t Agent::nb_actions() const { return _nb_actions; }

size_t Agent::episode_length() const { return _episode_length; }

double Agent::endowment() const { return _endowment; }

double Agent::payoff() const { return _payoff; }

const EGTTools::VectorXui & Agent::trajectoryStates() const { return _trajectory_states; }

const EGTTools::VectorXui & Agent::trajectoryActions() const { return _trajectory_actions; }

const EGTTools::Matrix2D & Agent::policy() const { return _policy; }

const EGTTools::Matrix2D & Agent::qValues() const { return _q_values; }

// Setters
void Agent::set_nb_states(size_t nb_states) {
    _nb_states = nb_states;
    _q_values.conservativeResizeLike(Matrix2D::Random(nb_states, _nb_actions));
    _policy.conservativeResizeLike(
            Matrix2D::Constant(nb_states, _nb_actions, 1.0 / static_cast<double>(_nb_actions)));
}

void Agent::set_nb_actions(size_t nb_actions) { _nb_actions = nb_actions; }

void Agent::set_episode_length(size_t episode_length) { _episode_length = episode_length; }

void Agent::set_endowment(double endowment) { _endowment = endowment; }

void Agent::set_payoff(double payoff) { _payoff = payoff; }

void Agent::resetPayoff() { _payoff = _endowment; }

void Agent::set_policy(const Eigen::Ref<const Matrix2D> &policy) {
    if (static_cast<size_t>(policy.rows()) != _nb_states)
        throw std::invalid_argument(
                "Policy must have as many rows as states (" + std::to_string(_nb_states) + ")");
    if (static_cast<size_t>(policy.cols()) != _nb_actions)
        throw std::invalid_argument(
                "Policy must have as many columns as actions (" + std::to_string(_nb_actions) + ")");
    _policy.array() = policy;
}

void Agent::set_q_values(const Eigen::Ref<const Matrix2D> &q_values) {
    if (static_cast<size_t>(q_values.rows()) != _nb_states)
        throw std::invalid_argument(
                "Q-values must have as many rows as states (" + std::to_string(_nb_states) + ")");
    if (static_cast<size_t>(q_values.cols()) != _nb_actions)
        throw std::invalid_argument(
                "Q-values must have as many columns as actions (" + std::to_string(_nb_actions) + ")");
    _q_values.array() = q_values;
    inferPolicy();
}

void Agent::set_trajectory_round(size_t round, size_t action) {
    _trajectory_actions(round) = action;
}
