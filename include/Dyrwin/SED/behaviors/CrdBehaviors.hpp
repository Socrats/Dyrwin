//
// Created by Elias Fernandez on 2019-05-15.
//

#ifndef DYRWIN_SED_BEHAVIOR_CRDBEHAVIORS_HPP
#define DYRWIN_SED_BEHAVIOR_CRDBEHAVIORS_HPP

#include <cstdlib>
#include <random>
#include <Dyrwin/SeedGenerator.h>

/**
 * @brief This header file contains the definition of the behaviors encountered in the
 * CRD experiments Elias & Jelena & Francisco C. Santo, et. al.
 */

namespace EGTTools::SED {
    constexpr size_t nb_strategies = 5;

    size_t cooperator(size_t prev_donation, size_t threshold);

    size_t defector(size_t prev_donation, size_t threshold);

    size_t altruist(size_t prev_donation, size_t threshold);

    size_t reciprocal(size_t prev_donation, size_t threshold);

    size_t compensator(size_t prev_donation, size_t threshold);


    struct CrdBehavior {
        CrdBehavior();

        CrdBehavior(size_t type);

        size_t type;
        double payoff;

        size_t (*act)(size_t, size_t);
    };

    #define UNUSED(expr) do { (void)(expr); } while (0)
}

#endif //DYRWIN_CRDBEHAVIORS_HPP
