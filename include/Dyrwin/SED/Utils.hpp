//
// Created by Elias Fernandez on 2019-04-25.
//

#ifndef DYRWIN_UTILS_HPP
#define DYRWIN_UTILS_HPP

#include <math.h>

namespace EGTTools::SED {
    double fermi(double beta, double a, double b) {
        return 1 / (1 + exp(beta * (a - b)));
    }
}

#endif //DYRWIN_UTILS_HPP
