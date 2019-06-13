//
// Created by Elias Fernandez on 2019-04-25.
//

#ifndef DYRWIN_SED_UTILS_HPP
#define DYRWIN_SED_UTILS_HPP

#include <cmath>
#include <limits>

namespace EGTTools::SED {
    /**
     * @brief returns the imitation probability calculated according to the fermi function.
     *
     * @param beta intensity of selection
     * @param a fitness of player A
     * @param b fitness fo player B
     * @return probability of imitation
     */
    double fermi(double beta, double a, double b);
    constexpr double_t doubleEpsilon = std::numeric_limits< double >::digits10;
}

#endif //DYRWIN_SED_UTILS_HPP
