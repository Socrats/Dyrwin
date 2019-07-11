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
        explicit TimingUncertainty(double p, size_t max_rounds=0) : p_(p), max_rounds_(max_rounds) {
            dist_ = G(p);
        };

        size_t calculateEnd( size_t min_rounds, R& generator ) {
            size_t rounds = min_rounds + dist_(generator);
            return (rounds > max_rounds_) ? max_rounds_ : rounds;
        }

        size_t calculateFullEnd( size_t min_rounds, R& generator ) {
            return min_rounds + dist_(generator);
        }

        // Getters
        double probability() const { return p_; }
        size_t max_rounds() const { return max_rounds_; }

        // Setters
        void setProbability( double p ) {
            if (p <= 0.0 || p > 1.0) throw std::invalid_argument("Probability must be in (0,1]");
            p_ = p;
            std::geometric_distribution<size_t> test(0.1);
            test.param(std::geometric_distribution<size_t>::param_type(p_));
        }

        void setMaxRounds( size_t max_rounds ) {
            if (max_rounds <= 0.0) throw std::invalid_argument("Max rounds must be > 0");
            max_rounds_ = max_rounds;
        }

    private:
        double p_;
        size_t max_rounds_;
        G dist_;

    };
}

#endif //DYRWIN_RL_TIMINGUNCERTAINTY_HPP
