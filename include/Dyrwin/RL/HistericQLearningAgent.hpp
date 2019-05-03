//
// Created by Elias Fernandez on 2019-04-17.
//

#ifndef DYRWIN_RL_HISTERICQLEARNINGAGENT_HPP
#define DYRWIN_RL_HISTERICQLEARNINGAGENT_HPP

#include <Dyrwin/RL/Agent.h>

namespace EGTTools::RL {
    class HistericQLearningAgent : public Agent {
    public:
        HistericQLearningAgent(unsigned int nb_rounds, unsigned int nb_actions, double endowment, double alpha,
                               double beta, double temperature) : Agent(nb_rounds, nb_actions, endowment),
                                                                  _alpha(alpha), _beta(beta),
                                                                  _temperature(temperature) {}

        HistericQLearningAgent(const HistericQLearningAgent &other) :
                Agent(other.nb_rounds(), other.nb_actions(), other.endowment()), _alpha(other.alpha()),
                _beta(other.beta()), _temperature(other.temperature()) {}


        void reinforceTrajectory() override {
            for (unsigned i = 0; i < _nb_rounds; i++) {
                const auto delta = _payoff - _q_values(i, _trajectory(i));
                if (delta >= 0) {
                    _q_values(i, _trajectory(i)) += _alpha * delta;
                } else {
                    _q_values(i, _trajectory(i)) += _beta * delta;
                }
            }
            resetTrajectory();
        }

        bool inferPolicy() override {
            unsigned int j;

            for (unsigned i = 0; i < _nb_rounds; i++) {
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

        // Getters
        double alpha() const { return _alpha; }

        double beta() const { return _beta; }

        double temperature() const { return _temperature; }

        // Setters
        void setAlpha(const double alpha) {
            if (alpha <= 0.0 || alpha > 1.0) throw std::invalid_argument("Learning rate parameter must be in (0,1]");
            _alpha = alpha;
        }

        void setBeta(const double beta) {
            if (beta <= 0.0 || beta > 1.0) throw std::invalid_argument("Learning rate parameter must be in (0,1]");
            _beta = beta;
        }

        void setTemperature(const double temperature) {
            if (temperature < 0.0) throw std::invalid_argument("temperature must be > 0");
            _temperature = temperature;
        }

    private:
        double _alpha, _beta, _temperature;
    };
}

#endif //DYRWIN_RL_HISTERICQLEARNINGAGENT_HPP
