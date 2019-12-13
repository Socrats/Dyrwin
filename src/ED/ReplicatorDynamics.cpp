//
// Created by Elias Fernandez on 07/11/2019.
//

#include <Dyrwin/ED/ReplicatorDynamics.hpp>

Vector EGTTools::ED::replicator_equation(EGTTools::Vector state, EGTTools::Matrix2D payoffs) {
    return state * ((payoffs * states) - (states * payoffs * states));
}