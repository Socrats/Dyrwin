//
// Created by Elias Fernandez on 2019-04-25.
//

#ifndef DYRWIN_UTILS_HPP
#define DYRWIN_UTILS_HPP

#include <math.h>
#include <limits>

namespace EGTTools::SED {
    double fermi(double beta, double a, double b) {
        return 1 / (1 + exp(beta * (a - b)));
    }
    constexpr double_t doubleEpsilon = std::numeric_limits< double >::digits10;
}

#endif //DYRWIN_UTILS_HPP
