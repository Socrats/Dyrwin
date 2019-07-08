//
// Created by Elias Fernandez on 2019-04-25.
//

#ifndef DYRWIN_SED_UTILS_HPP
#define DYRWIN_SED_UTILS_HPP

#include <cmath>
#include <limits>
#include <vector>
#include <Dyrwin/Types.h>
#include <Dyrwin/Distributions.h>

namespace EGTTools::SED {
    using GroupPayoffs = EGTTools::Matrix2D;
    using StrategyCounts = std::vector<size_t>;

    /**
     * @brief returns the imitation probability calculated according to the fermi function.
     *
     * @param beta intensity of selection
     * @param a fitness of player A
     * @param b fitness fo player B
     * @return probability of imitation
     */
    double fermi(double beta, double a, double b);

    /**
     * @brief contest success function that compares 2 payoffs according to a payoff importance z
     *
     * This function must never be called with z = 0. This would produce a zero division error.
     * And the behaviour might be undefined.
     *
     * @param z : importance of the payoff
     * @param a : expected payoff a
     * @param b : expected payoff b
     * @return probability of a winning over b
     */
    double contest_success(double z, double a, double b);

    /**
     * @brief contest success function that compares 2 payoffs according to a payoff importance z
     *
     * This function should be used when z = 0
     *
     * @param a : expected payoff a
     * @param b : expected payoff b
     * @return probability of a winning over b
     */
    double contest_success(double a, double b);

    /**
    * @brief This function converts a vector containing counts into an index.
    *
    * This method was copies from @ebargiac
    *
    * @param data The vector to convert.
    * @param history The sum of the values contained in data.
    *
    * @return The unique index in [0, starsBars(history, data.size() - 1)) representing data.
    */
    size_t calculate_state(const size_t &group_size, const size_t &nb_states, const EGTTools::Factors &current_group);

    /**
     * @brief Transforms and state index into a vector.
     *
     * @param i : state index
     * @param pop_size : size of the population
     * @param nb_strategies : number of strategies
     * @param state : container for the sampled state
     */
    void sample_simplex(size_t i, const size_t &pop_size, const size_t &nb_strategies, VectorXui &state);

    template<typename G>
    void sample_simplex(size_t nb_strategies, Vector &state, std::uniform_real_distribution<double> prob_dist,
                        G &generator) {
        for (size_t i = 0; i < nb_strategies; ++i) {
            state(i) = - std::log(prob_dist(generator));
        }
        state.array() /= state.sum();
        assert(state.sum() == 1.0);
    }

    /**
     * @brief Defines the numeric limit of floating points
     */
    constexpr double_t doubleEpsilon = std::numeric_limits<double>::digits10;
}

#endif //DYRWIN_SED_UTILS_HPP
