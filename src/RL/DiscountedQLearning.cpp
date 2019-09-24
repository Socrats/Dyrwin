//
// Created by Elias Fernandez on 2019-09-18.
//

#include <Dyrwin/RL/DiscountedQLearning.h>

using namespace EGTTools::RL;

DiscountedQLearning::DiscountedQLearning(size_t nb_states, size_t nb_actions, size_t episode_length,
                                         double endowment,
                                         double alpha,
                                         double lambda,
                                         double temperature)
        : Agent(nb_states, nb_actions, episode_length, endowment),
          _alpha(alpha), _lambda(lambda), _temperature(temperature) {}

void DiscountedQLearning::reinforceTrajectory() {
    // First we apply the discount
    _q_values *= _lambda;
    // Then we apply the batch update over the trajectory actions
    for (unsigned i = 0; i < _episode_length; ++i) {
        _q_values(_trajectory_states(i), _trajectory_actions(i)) +=
                _alpha * (_payoff - _q_values(_trajectory_states(i), _trajectory_actions(i)));
    }
}

void DiscountedQLearning::reinforceTrajectory(size_t episode_length) {
    // First we apply the discount
    _q_values *= _lambda;
    // Then we apply the batch update over the trajectory actions and states
    for (unsigned i = 0; i < episode_length; ++i) {
        _q_values(_trajectory_states(i), _trajectory_actions(i)) +=
                _alpha * (_payoff - _q_values(_trajectory_states(i), _trajectory_actions(i)));
    }
}

bool DiscountedQLearning::inferPolicy() {
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
            unsigned selection = dist(_mt);
            _policy.row(i).setZero();
            _policy(i, _buffer[selection]) = 1.0;
        } else {
            _policy.row(i).array() /= total;
        }
    }
    return true;
}

void DiscountedQLearning::resetQValues() {
    _q_values.setRandom();
}

std::string DiscountedQLearning::type() const {
    return "EGTTools::RL::DiscountedQLearning";
}

// Getters
double DiscountedQLearning::alpha() const { return _alpha; }

double DiscountedQLearning::lambda() const { return 1 - _lambda; }

double DiscountedQLearning::temperature() const { return _temperature; }

// Setters
void DiscountedQLearning::setAlpha(double alpha) {
    if (alpha <= 0.0 || alpha > 1.0) throw std::invalid_argument("Learning rate parameter must be in (0,1]");
    _alpha = alpha;
}

void DiscountedQLearning::setLambda(double lambda) {
    if (lambda <= 0.0 || lambda > 1.0) throw std::invalid_argument("Forgetting factor must be in (0,1]");
    _lambda = 1 - lambda;
}

void DiscountedQLearning::setTemperature(double temperature) {
    if (temperature < 0.0) throw std::invalid_argument("temperature must be > 0");
    _temperature = temperature;
}
