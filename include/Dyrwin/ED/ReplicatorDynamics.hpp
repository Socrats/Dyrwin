//
// Created by Elias Fernandez on 15/10/2019.
//

#ifndef DYRWIN_ED_REPLICATORDYNAMICS_HPP
#define DYRWIN_ED_REPLICATORDYNAMICS_HPP

#include <Dyrwin/Types.h>
#include <Dyrwin/OpenMPUtils.hpp>

namespace EGTTools::ED {
    /**
     * @brief replicator equation that returns the next state, given an initial state.
     *
     * It requires a matrix of @param payoffs, that will define the dynamics of the system.
     * At each given state, the next state of the population can be computed deterministically.
     * Solving this diferential equation issues all possible states through time.
     *
     * @param state : (Eigen::Vector) indicates the frequency of each type in the population
     * @param payoffs : indicates the payoffs in normal form (rows indicate the focal player,
     *                  columns indicate the opponent).
     * @return : (Eigen::Vector) new_state
     */
    Vector replicator_equation(Vector state, Matrix2D payoffs);
}

#endif //DYRWIN_ED_REPLICATORDYNAMICS_HPP