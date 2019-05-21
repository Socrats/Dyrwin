//
// Created by Elias Fernandez on 2019-05-21.
//

#ifndef DYRWIN_OPENMPUTILS_HPP
#define DYRWIN_OPENMPUTILS_HPP

#include <Dyrwin/Types.h>

#pragma omp declare reduction (+: EGTTools::Vector: omp_out=omp_out+omp_in)\
     initializer(omp_priv=EGTTools::Vector::Zero(omp_orig.size()))

#endif //DYRWIN_OPENMPUTILS_HPP