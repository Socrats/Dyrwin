
//
// Created by Elias Fernandez on 27/11/2018.
//

#ifndef DYRWIN_ITERATEDPRISONERSDILEMA_H
#define DYRWIN_ITERATEDPRISONERSDILEMA_H

#include <vector>
#include <random>
#include "../Utils.h"
#include "../SeedGenerator.h"


struct GameData {
    bool coop_level;
    // add strategy frequency
};

class IteratedPrisonersDilemma {
    /**
     * Implements the 2-person Iterated Prisoners Dilemma.
     *
     * Here Players are matched randomly into pairs and they play
     * the Prisoner's Dilemma game for a number of rounds.
     */
public:
    IteratedPrisonersDilemma(double T, double R, double P, double S);

    ~IteratedPrisonersDilemma() = default;

    GameData run(unsigned int rounds, std::vector<EvoIndividual *> &players);
};

#endif //DYRWIN_ITERATEDPRISONERSDILEMA_H
