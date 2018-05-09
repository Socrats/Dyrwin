//
// Created by Elias Fernandez on 14/03/2018.
//

#ifndef DYRWIN_COLLECTIVERISKDILEMMA_H
#define DYRWIN_COLLECTIVERISKDILEMMA_H

#include <vector>
#include <random>
#include "Utils.h"

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
                          unsigned int game_rounds, boost::mt19937 &mt);

    ~CollectiveRiskDilemma() {};

    void run(unsigned int rounds, std::vector<EvoIndividual *> &players);

    /**
     * This version of run should be called if we want the fitness of each player returned in a vector
     * @param rounds
     * @param players
     * @param fitnessVector
     */

    void run(unsigned int rounds, std::vector<EvoIndividual *> &players, std::vector<double> &fitnessVector) {
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

private:
    unsigned int _current_generation = 0;
    double _public_account = 0.0; // Ammount acumulated at the public_account on one game
    std::vector<double> _fitnessVector;

    void _update_fitness_met_threshold(EvoIndividual *individual);

    void _update_fitness_not_met_threshold(EvoIndividual *individual);

    // Random generators
    boost::mt19937 &_mt;
    boost::uniform_real<> _uniform = boost::uniform_real<>(0, 1);
    boost::variate_generator<boost::mt19937 &, boost::uniform_real<> > _rng_uniform =
            boost::variate_generator<boost::mt19937 &, boost::uniform_real<> >(_mt, _uniform);
};


#endif //DYRWIN_COLLECTIVERISKDILEMMA_H
