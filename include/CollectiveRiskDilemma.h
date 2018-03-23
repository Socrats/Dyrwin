//
// Created by Elias Fernandez on 14/03/2018.
//

#ifndef DYRWIN_COLLECTIVERISKDILEMMA_H
#define DYRWIN_COLLECTIVERISKDILEMMA_H

#include <vector>
#include <random>
#include "CRDPlayer.h"

class CollectiveRiskDilemma {
    /**
     * Implements the Collective-Risk Dilemma evolutionary game as defined in
     * Chakra, Maria A. and Traulsen, Arne. "Evolutionary Dynamics of Strategic Behavior in a
     * Collective-Risk Dilemma. Plos Compuational Biology. August 2012. Volume 8(8).
     *
     * Such a collective-risk game is played among M individuals selected at random
     * from a well mixed population of size N.
     */
public:
    CollectiveRiskDilemma(unsigned int nb_actions, unsigned int group_size, double target_sum, double risk,
                          unsigned int game_rounds, double beta);

    ~CollectiveRiskDilemma() {};

    void run(unsigned int rounds, std::vector<CRDPlayer *> &players);

    /**
     * This version of run should be called if we want the fitness of each player returned in a vector
     * @param rounds
     * @param players
     * @param fitnessVector
     */

    void run(unsigned int rounds, std::vector<CRDPlayer *> &players, std::vector<double> &fitnessVector) {
        _fitnessVector = fitnessVector;
        run(rounds, players);
    };

    unsigned int get_current_generation();

    double get_public_account();

    unsigned int nb_actions; // 3 = 0, 1, 2
    unsigned int group_size; // M individuals selected to play a game at each generation
    double target_sum; // T amount that the group must reach at the end of the game
    double risk; // Probability that players loose what they have not invested if T is not reached
    unsigned int game_rounds; // number of rounds that each game will have
    double beta; // intensity of selection

private:
    unsigned int _current_generation = 0;
    double _public_account = 0.0; // Ammount acumulated at the public_account on one game
    std::vector<double> _fitnessVector;

    // seed with a real random value, if available
    std::random_device rd;
    // random engine
    std::default_random_engine _random_engine1;
    std::seed_seq _seed{rd(), rd(), rd(), rd(), rd(), rd(), rd(), rd()};
    std::mt19937 _random_engine2;
    // uniform distribution
    std::uniform_int_distribution<int> _uniform_dist;
    // normal distribution
    std::normal_distribution<> _normal_distribution;

    double _calculate_fitness(CRDPlayer &player);
};


#endif //DYRWIN_COLLECTIVERISKDILEMMA_H
