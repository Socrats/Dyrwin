//
// Created by Elias Fernandez on 09/04/2018.
//

#ifndef DYRWIN_ABOUCRD_UTILS_H
#define DYRWIN_ABOUCRD_UTILS_H

#include <Dyrwin/AbouCRD/CRDPlayer.h>

struct EvoIndividual {
    double *fitness;
    CRDPlayer player;
    unsigned int games_played;

    EvoIndividual(double *fitness, CRDPlayer &player) :
            fitness(fitness), player(player), games_played(0) {}

    EvoIndividual(double *fitness, double mu, double sigma) :
            fitness(fitness), player(CRDPlayer(mu, sigma)), games_played(0) {}

    ~EvoIndividual() = default;

    void init() {
        *fitness = 0;
        this->games_played = 0;
    };
};

#endif //DYRWIN_ABOUCRD_UTILS_H

