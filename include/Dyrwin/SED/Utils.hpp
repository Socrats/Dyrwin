//
// Created by Elias Fernandez on 2019-04-25.
//

#ifndef DYRWIN_SED_UTILS_HPP
#define DYRWIN_SED_UTILS_HPP

#include <cmath>
#include <limits>

namespace EGTTools::SED {
    double fermi(double beta, double a, double b) {
        return 1 / (1 + std::exp(beta * (a - b)));
    }
    constexpr double_t doubleEpsilon = std::numeric_limits< double >::digits10;
}

#endif //DYRWIN_SED_UTILS_HPP
