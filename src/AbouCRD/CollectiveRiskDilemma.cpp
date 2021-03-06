//
// Created by Elias Fernandez on 14/03/2018.
//

#include <Dyrwin/AbouCRD/CollectiveRiskDilemma.h>

CollectiveRiskDilemma::CollectiveRiskDilemma(unsigned int nb_actions, unsigned int group_size, double target_sum,
                                             double risk, unsigned int game_rounds, double endowment) :
        nb_actions(nb_actions), group_size(group_size),
        target_sum(target_sum), risk(risk), endowment(endowment), game_rounds(game_rounds) {

}

GameData CollectiveRiskDilemma::run(unsigned int rounds, std::vector<EvoIndividual *> &players) {
    // Defines the game that is being played

    // Finds size of arr[] and stores in 'size'
    size_t size = players.size();
    double total_donations = 0;
    double round_donations = 0;
    size_t i, j;
    // Uniform Distribution
    std::uniform_real_distribution<double> _uniform_real_dist(0.0, 1.0);

    // Initialize players on the first round
    round_donations = 0;
    for (j = 0; j < size; j++) {
        players[j]->player.reset(endowment);
        round_donations += players[j]->player.getAction(total_donations, 0, target_sum);
    }
    total_donations += round_donations;

    // Iterate for rounds and get each player's contribution
    for (i = 1; i < rounds; i++) {
        // Iterate over the group players
        round_donations = 0;
        for (j = 0; j < size; j++) {
            round_donations += players[j]->player.getAction(total_donations, i, target_sum);
        }
        total_donations += round_donations;
    }

    // Indicates whether the group met the threshold
    bool met_threshold = true;
    // Indicates whether the players can keep the amount in its public account
    bool keep_pa = true;

    // Check if the threshold has NOT been reached
    if (target_sum > total_donations) {
        met_threshold = false;
        // Generate random number
        if (risk > _uniform_real_dist(_mt)) {
            keep_pa = false;
        }
    }

    if (keep_pa) {
        // Update all players payoffs with their private account
        for (j = 0; j < size; j++) {
            _update_fitness_met_threshold(players[j]);
        }
    } else {
        // Update all players payoffs with 0
        for (j = 0; j < size; j++) {
            _update_fitness_not_met_threshold(players[j]);
        }
    }
    return {met_threshold, total_donations};
}

void CollectiveRiskDilemma::_update_fitness_met_threshold(EvoIndividual *individual) {
    double scaling = 1. / (double) (individual->games_played + 1);
    // Calculate moving average
    *(individual->fitness) = (individual->player.getPayoff() * scaling) + (*(individual->fitness) * (1. - scaling));
    individual->games_played++;
//    individual->player.updatePayoff(0.0);
}

void CollectiveRiskDilemma::_update_fitness_not_met_threshold(EvoIndividual *individual) {
    individual->player.updatePayoff(0.0);
    double scaling = 1. / (double) (individual->games_played + 1);
    // Calculate moving average
    *(individual->fitness) = (*(individual->fitness) * (1. - scaling));
    individual->games_played++;
}

unsigned int CollectiveRiskDilemma::get_current_generation() {
    return this->_current_generation;
}

double CollectiveRiskDilemma::get_public_account() {
    return this->_public_account;
}
