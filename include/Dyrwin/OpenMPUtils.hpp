//
// Created by Elias Fernandez on 2019-05-21.
//

#ifndef DYRWIN_OPENMPUTILS_HPP
#define DYRWIN_OPENMPUTILS_HPP

#include <omp.h>
#include <Dyrwin/Types.h>

#pragma omp declare reduction (+: EGTTools::Vector: omp_out=omp_out+omp_in)\
     initializer(omp_priv=EGTTools::Vector::Zero(omp_orig.size()))

#pragma omp declare reduction (+: EGTTools::VectorXui: omp_out=omp_out+omp_in)\
     initializer(omp_priv=EGTTools::VectorXui::Zero(omp_orig.size()))

#pragma omp declare reduction (+: EGTTools::Vector3d: omp_out=omp_out+omp_in)\
     initializer(omp_priv=EGTTools::Vector3d::Zero())

#endif //DYRWIN_OPENMPUTILS_HPP
