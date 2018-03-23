//
// Created by Elias Fernandez on 14/03/2018.
//

#include <iostream>
#include "../include/CollectiveRiskDilemma.h"

CollectiveRiskDilemma::CollectiveRiskDilemma(unsigned int nb_actions, unsigned int group_size, double target_sum,
                                             double risk, unsigned int game_rounds, double beta) :
        nb_actions(nb_actions), group_size(group_size),
        target_sum(target_sum), risk(risk), game_rounds(game_rounds), beta(beta) {

}

void CollectiveRiskDilemma::run(unsigned int rounds, std::vector<CRDPlayer *> &players) {
    // Defines the game that is being played

    // Finds size of arr[] and stores in 'size'
    size_t size = players.size();
    double total_donations = 0;
    double round_donations = 0;
    int i, j;

    // Iterate for rounds and get each player's contribution
    for (i = 0; i < rounds; i++) {
        // Iterate over the group players
        round_donations = 0;
        for (j = 0; j < size; j++) {
            round_donations += players[j]->getAction(total_donations);
        }
        total_donations += round_donations;
    }

    // Check if the threshold has NOT been reached
    if (target_sum > total_donations) {
        // Generate random number
        double rd_number = 0.4;
        if (rd_number <= risk) {
            // Update all players payoffs with 0
            for (j = 0; j < size; j++) {
                players[j]->updatePayoff(0.0);
            }
        }
    }

}

unsigned int CollectiveRiskDilemma::get_current_generation() {
    return this->_current_generation;
}

double CollectiveRiskDilemma::get_public_account() {
    return this->_public_account;
}

double CollectiveRiskDilemma::_calculate_fitness(CRDPlayer &player) {
    return exp(this->beta * player.getPayoff());
}
