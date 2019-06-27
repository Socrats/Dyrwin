//
// Created by Elias Fernandez on 2019-04-17.
//

#ifndef DYRWIN_RL_TIMINGUNCERTAINTY_HPP
#define DYRWIN_RL_TIMINGUNCERTAINTY_HPP

#include <random>

namespace EGTTools {
    template<typename R, typename G = std::geometric_distribution<size_t>>
    class TimingUncertainty {
    public:
        TimingUncertainty(double p) : _p(p), _max_rounds(0) {};

        TimingUncertainty(double p, size_t max_rounds) : _p(p), _max_rounds(max_rounds) {};

        size_t calculateEnd( size_t min_rounds, R& generator ) {
            G d(_p);
            size_t rounds = min_rounds + d(generator);
            return (rounds > _max_rounds) ? _max_rounds : rounds;
        }

        size_t calculateFullEnd( size_t min_rounds, R& generator ) {
            G d(_p);
            return min_rounds + d(generator);
        }

        // Getters
        double probability() const { return _p; }
        size_t max_rounds() const { return _max_rounds; }

        // Setters
        void setProbability( double p ) {
            if (p <= 0.0 || p > 1.0) throw std::invalid_argument("Probability must be in (0,1]");
            _p = p;
        }

        void setMaxRounds( size_t max_rounds ) {
            if (max_rounds <= 0.0) throw std::invalid_argument("Max rounds must be > 0");
            _max_rounds = max_rounds;
        }

    private:
        double _p;
        size_t _max_rounds;

    };
}

#endif //DYRWIN_RL_TIMINGUNCERTAINTY_HPP
