//
// Created by Elias Fernandez on 04/01/2020.
//

#include <Dyrwin/RL/algorithms/QLearning.h>

EGTTools::RL::algorithms::QLearning::QLearning(size_t nb_states,
                                               size_t nb_actions,
                                               size_t episode_length,
                                               double endowment,
                                               double alpha,
                                               double lambda,
                                               double temperature)
    : _nb_states(nb_states),
      _nb_actions(nb_actions),
      _episode_length(episode_length),
      _endowment(endowment),
      _alpha(alpha),
      _lambda(lambda),
      _temperature(temperature) {
  _payoff = _endowment;
  // Initialise q-values uniformly random between (-1, 1).
  _q_values = Matrix2D::Random(_nb_states, _nb_actions);
  // Initialise all actions with equal probability
  _policy = Matrix2D::Constant(_nb_states, _nb_actions, 1.0 / static_cast<double>(_nb_actions));
  // Initialise buffer for the batch updates
  _batch_updates = Matrix2D::Zero(_nb_states, _nb_actions);
  // Initialise to zero the state-action counts
  _batch_counts = Matrix2D::Zero(_nb_states, _nb_actions);
  // Initialise trajectory (the actions taken at each round)
  _trajectory_states = VectorXui::Zero(_episode_length);
  _trajectory_actions = VectorXui::Zero(_episode_length);
  // Initialise the helper containers
  _buffer = std::vector<size_t>(_nb_actions);
  _real_rand = std::uniform_real_distribution<double>(0.0, 1.0);
}
void EGTTools::RL::algorithms::QLearning::reinforceTrajectory() {
  // TODO: change the name of this method to "calculateTD".
  // Now it will calculate the temporal difference errors of the current trajectory
  // And will store them in the TD error buffer (_batch_updates). It will also
  // increase the counts of every visited state (_batch_count).
  // The reinforcement/update of q-values estimation should be
  // calculated at the end of the batch, just before the policy is inferred.
  for (unsigned i = 0; i < _episode_length; ++i) {
    // q-value update - Calculate temporal difference error
    _batch_updates(_trajectory_states(i), _trajectory_actions(i)) += _payoff -
        _q_values(_trajectory_states(i), _trajectory_actions(i));
    // Increase state-action counts
    ++_batch_counts(_trajectory_states(i), _trajectory_actions(i));
  }
}
void EGTTools::RL::algorithms::QLearning::reinforceTrajectory(size_t episode_length) {
  // TODO: change the name of this method to "calculateTD".
  // Now it will calculate the temporal difference errors of the current trajectory
  // And will store them in the TD error buffer (_batch_updates). It will also
  // increase the counts of every visited state (_batch_count).
  // The reinforcement/update of q-values estimation should be
  // calculated at the end of the batch, just before the policy is inferred.
  for (unsigned i = 0; i < episode_length; ++i) {
    // q-value update - Calculate temporal difference error
    _batch_updates(_trajectory_states(i), _trajectory_actions(i)) += _payoff -
        _q_values(_trajectory_states(i), _trajectory_actions(i));
    // Increase state-action counts
    ++_batch_counts(_trajectory_states(i), _trajectory_actions(i));
  }
}
void EGTTools::RL::algorithms::QLearning::resetTrajectory() {
  _trajectory_states.setZero();
  _trajectory_actions.setZero();
}
bool EGTTools::RL::algorithms::QLearning::inferPolicy() {
  // Before the policy is calculated using the Gibbs-Boltzmann distribution,
  // we update the q-values estimation through the stored batch TD-errors.
  _batch_updates = (_batch_counts.array() != 0).select(_batch_updates.cwiseQuotient(_batch_counts),
                                                       _batch_updates);
  _q_values += _alpha * _batch_updates;
  // After we reinitialize both buffers
  _batch_updates.setZero();
  _batch_counts.setZero();
  // Then we infer the policy
  unsigned int j;

  for (unsigned i = 0; i < _nb_states; i++) {
    // We calculate the sum of exponential(s) of q values for each state
    double total = 0.;
    unsigned nb_infs = 0;
    for (j = 0; j < _nb_actions; j++) {
      _policy(i, j) = exp(_q_values(i, j) * _temperature);
      if (std::isinf(_policy(i, j))) _buffer[nb_infs++] = j;
      total += _policy(i, j);
    }
    if (nb_infs) {
      auto dist = std::uniform_int_distribution<unsigned>(0, nb_infs - 1);
      unsigned selection = dist(_generator);
      _policy.row(i).setZero();
      _policy(i, _buffer[selection]) = 1.0;
    } else {
      _policy.row(i).array() /= total;
    }
  }
  return true;
}
size_t EGTTools::RL::algorithms::QLearning::selectAction(size_t state) {
  assert(state < _episode_length);
  _trajectory_states(state) = state;
  _trajectory_actions(state) = EGTTools::choice(_nb_actions, _policy.row(state), _real_rand, _generator);
  return _trajectory_actions(state);
}
size_t EGTTools::RL::algorithms::QLearning::selectAction(size_t current_round, size_t state) {
  assert(current_round < _episode_length);
  _trajectory_states(current_round) = state;
  _trajectory_actions(current_round) = EGTTools::choice(_nb_actions, _policy.row(state), _real_rand, _generator);
  return _trajectory_actions(current_round);
}
bool EGTTools::RL::algorithms::QLearning::decreasePayoff(double value) {
  if ((_payoff - value) >= 0) {
    _payoff -= value;
    return true;
  }
  return false;
}
void EGTTools::RL::algorithms::QLearning::increasePayoff(double value) {
  _payoff += value;
}
void EGTTools::RL::algorithms::QLearning::reset() {
  _q_values.setRandom();
  // Initialise all actions with equal probability
  _policy.setConstant(1.0 / static_cast<double>(_nb_actions));
  // Initialise trajectory (the actions taken at each round)
  _trajectory_states.setZero();
  _trajectory_actions.setZero();
  _payoff = _endowment;
}
void EGTTools::RL::algorithms::QLearning::resetQValues() {
  _q_values.setZero();
}
std::string EGTTools::RL::algorithms::QLearning::toString() const {
  std::stringstream ss;
  Eigen::IOFormat CleanFmt(4, 0, ", ", "\n", "[", "]");

  ss << "[(" << _payoff << ")";
  ss << _policy.format(CleanFmt);
  ss << "]" << std::endl;
  return ss.str();
}
std::string EGTTools::RL::algorithms::QLearning::type() const {
  return "EGTTools::RL::algorithms::QLearning";
}
size_t EGTTools::RL::algorithms::QLearning::nb_states() const {
  return _nb_states;
}
size_t EGTTools::RL::algorithms::QLearning::nb_actions() const {
  return _nb_actions;
}
double EGTTools::RL::algorithms::QLearning::endowment() const {
  return _endowment;
}
double EGTTools::RL::algorithms::QLearning::payoff() const {
  return _payoff;
}
double EGTTools::RL::algorithms::QLearning::alpha() const {
  return _alpha;
}
double EGTTools::RL::algorithms::QLearning::lambda() const {
  return _lambda;
}
double EGTTools::RL::algorithms::QLearning::temperature() const {
  return _temperature;
}
const EGTTools::VectorXui &EGTTools::RL::algorithms::QLearning::trajectoryStates() const {
  return _trajectory_states;
}
const EGTTools::VectorXui &EGTTools::RL::algorithms::QLearning::trajectoryActions() const {
  return _trajectory_actions;
}
const EGTTools::Matrix2D &EGTTools::RL::algorithms::QLearning::policy() const {
  return _policy;
}
const EGTTools::Matrix2D &EGTTools::RL::algorithms::QLearning::qValues() const {
  return _q_values;
}
void EGTTools::RL::algorithms::QLearning::set_nb_states(size_t nb_states) {
  _nb_states = nb_states;
  _q_values.conservativeResizeLike(Matrix2D::Zero(nb_states, _nb_actions));
  _policy.conservativeResizeLike(
      Matrix2D::Constant(nb_states, _nb_actions, 1.0 / static_cast<double>(_nb_actions)));
}
void EGTTools::RL::algorithms::QLearning::set_nb_actions(size_t nb_actions) {
  _nb_actions = nb_actions;
  _q_values.conservativeResizeLike(Matrix2D::Zero(_nb_states, _nb_actions));
  _policy.conservativeResizeLike(
      Matrix2D::Constant(_nb_states, _nb_actions, 1.0 / static_cast<double>(_nb_actions)));
}
void EGTTools::RL::algorithms::QLearning::set_endowment(double endowment) {
  _endowment = endowment;
}
void EGTTools::RL::algorithms::QLearning::set_payoff(double payoff) {
  _payoff = payoff;
}
void EGTTools::RL::algorithms::QLearning::reset_payoff() {
  _payoff = _endowment;
}
void EGTTools::RL::algorithms::QLearning::set_policy(const Eigen::Ref<const Matrix2D> &policy) {
  if (static_cast<size_t>(policy.rows()) != _nb_states)
    throw std::invalid_argument(
        "Policy must have as many rows as states (" + std::to_string(_nb_states) + ")");
  if (static_cast<size_t>(policy.cols()) != _nb_actions)
    throw std::invalid_argument(
        "Policy must have as many columns as actions (" + std::to_string(_nb_actions) + ")");
  _policy.array() = policy;
}
void EGTTools::RL::algorithms::QLearning::set_q_values(const Eigen::Ref<const Matrix2D> &q_values) {
  if (static_cast<size_t>(q_values.rows()) != _nb_states)
    throw std::invalid_argument(
        "Policy must have as many rows as states (" + std::to_string(_nb_states) + ")");
  if (static_cast<size_t>(q_values.cols()) != _nb_actions)
    throw std::invalid_argument(
        "Policy must have as many columns as actions (" + std::to_string(_nb_actions) + ")");
  _q_values.array() = q_values;
}
void EGTTools::RL::algorithms::QLearning::set_trajectory_round(size_t round, size_t action) {
  _trajectory_actions(round) = action;
}
void EGTTools::RL::algorithms::QLearning::setAlpha(double alpha) {
  if (alpha <= 0.0 || alpha > 1.0) throw std::invalid_argument("Learning rate must be in (0,1]");
  _alpha = alpha;
}
void EGTTools::RL::algorithms::QLearning::setLambda(double lambda) {
  if (lambda <= 0.0 || lambda > 1.0) throw std::invalid_argument("Discount factor must be in (0,1]");
  _lambda = lambda;
}
void EGTTools::RL::algorithms::QLearning::setTemperature(double temperature) {
  if (temperature < 0.0) throw std::invalid_argument("temperature must be > 0");
  _temperature = temperature;
}
std::ostream &EGTTools::RL::algorithms::QLearning::display(std::ostream &os) const {
  Eigen::IOFormat CleanFmt(4, 0, ", ", "\n", "[", "]");

  os << "[(" << _payoff << ")";
  os << _policy.format(CleanFmt);
  os << "]" << std::endl;
  return os;
}
