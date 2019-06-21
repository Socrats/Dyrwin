//
// Created by Elias Fernandez on 2019-04-25.
//

#ifndef DYRWIN_SED_UTILS_HPP
#define DYRWIN_SED_UTILS_HPP

#include <cmath>
#include <limits>
#include <vector>
#include <Dyrwin/Types.h>

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
     * @brief Defines the numeric limit of floating points
     */
    constexpr double_t doubleEpsilon = std::numeric_limits<double>::digits10;
}

#endif //DYRWIN_SED_UTILS_HPP
