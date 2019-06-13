//
// Created by Elias Fernandez on 2019-06-12.
//

#include <Dyrwin/SED/Utils.hpp>

double EGTTools::SED::fermi(double beta, double a, double b) {
    return 1 / (1 + std::exp(beta * (a - b)));
}