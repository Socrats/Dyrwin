//
// Created by Elias Fernandez on 15/03/2018.
//

#include <iostream>
#include "../include/Dyrwin/CRDSimulator.h"

CRDSimulator::CRDSimulator(unsigned int population_size) : population_size(population_size), beta(1.0) {
    // Default parameters
    group_size = 6;
    nb_games = 1000;
    double target_sum = 60;
    double risk = 0.9;
    unsigned int game_rounds = 10;
    double mu = 0.003;
    double sigma = 0.15;

    // Initialize fitness vector
    _fitnessVector = std::vector<double>(population_size, 0.0);

    // Vector of players
    std::vector<CRDPlayer> players = std::vector<CRDPlayer>(population_size, CRDPlayer(mu, sigma));

    // Generate population
    // reserve memory for vector of pointers
    _population.reserve(population_size);
    _population_tmp.reserve(population_size);
    _populationTypesHash.reserve(population_size);
    players.reserve(population_size);


    for (unsigned int i = 0; i < population_size; i++) {
        players[i] = CRDPlayer(mu, sigma);
        _population.push_back(EvoIndividual(&_fitnessVector[i], players[i]));
//        auto it = _populationTypesHash.find(players[i].strategy);
//        if (it != _populationTypesHash.end()) {
//            std::cout << "Strategy already exists" << std::endl;
//            it->second++;
//        } else {
//            auto ok = _populationTypesHash.insert(std::make_pair(players[i].strategy, strategy_fq(players[i].strategy, 1)));
//            std::cout << "New strategy added: " << ok.second << std::endl;
//            std::cout << "Freq of the strategy: " << ok.first->second.freq << std::endl;
//        }
    }
    for (auto const& individual: _population) {
        std::cout << "===========================" << std::endl;
        std::cout << "Player fitness: " << *(individual.fitness) << std::endl;
        std::cout << "---------------------------" << std::endl;
        int d = 0;
        for (auto const& strategy: individual.player.strategy.round_strategies) {
            std::cout << "---------------------------" << std::endl;
            std::cout << "Round " << ++d << std::endl;
            std::cout << "First action: " << strategy.first << std::endl;
            std::cout << "Second action: " << strategy.second << std::endl;
            std::cout << "Threshold: " << strategy.threshold << std::endl;
            std::cout << "---------------------------" << std::endl;
        }
    }

    std::cout << "Llega aquí! Rounds: " << _population[0].player.strategy.rounds << std::endl;
    std::cout << "Strategy threshold 0: " << _population[0].player.strategy.round_strategies[1].threshold << std::endl;

    // Make sure that population vector has been initialized
    assert(!_population.empty());
    assert(!_population[0].player.strategy.round_strategies.empty());
    assert(!_populationTypesHash.empty());
    assert(!_populationTypesHash.find(_population[0].player.strategy)->second.strategy.round_strategies.empty());

    // Copy population into population tmp
    _population_tmp = _population;

    // Initialize population indexes
    _population_indexes = std::vector<int>(population_size);
    std::iota(_population_indexes.begin(), _population_indexes.end(), 0);

    // Initialize group structures
    _group = std::vector<EvoIndividual *>(group_size);
    _group_indexes = std::vector<unsigned int>(population_size);
    std::iota(_group_indexes.begin(), _group_indexes.end(), 0);

    // Initialize Game
    _game = new CollectiveRiskDilemma(3, group_size, target_sum, risk, game_rounds);

}

/**
 * Evolves the population through selection and mutation operators.
 * @param generations - numbers of generations through which the population is evolved
 */
void CRDSimulator::evolve(unsigned int generations) {
    int i, j;

    for (j = 0; j < generations; j++) {
        std::cout << "generation " << j << std::endl;

        // First play games and calculate fitness
        for (i = 0; i < nb_games; i++) {
            // Generate group
            auto group = _select_randomly(group_size);
            _game->run(game_rounds, group);
        }

        // Then apply selection - Wright-Fisher Process
        // First update fitnessVector
        _update_fitness_vector();
        // Then, generate new population_indexes drawn at random with fitnessVector as weights
        _update_population_indexes();
        // Finally, update population
//        std::pair<std::unordered_map<SequentialStrategy, strategy_fq, SequentialStrategyHasher>::iterator, bool> ok;
        for (i = 0; i < population_size; i++) {
            // Update player in population
            _population_tmp[i].player.strategy.copy(_population[_population_indexes[i]].player.strategy);

            // Reduce strategy frequency
//            _populationTypesHash[_population_tmp[i].player.strategy]--;

            // Mutate player strategy
            ++_population_tmp[i].player.strategy;

            auto ok = _populationTypesHash.insert(
                    std::make_pair(_population_tmp[i].player.strategy,
                                   strategy_fq(_population_tmp[i].player.strategy, 1))
            );

            if (!ok.second) {
                std::cout << " " << ok.second << std::endl;
//                _populationTypesHash[_population_tmp[i].player.strategy]++;
            }
            std::cout << "Reaches here" << std::endl;
        }
        // Update new population
        _population.swap(_population_tmp);

    }

}

/**
 * This method returns a subgroup formed by randomly selecting members (without replacement) from the population pool.
 * @param size : size of the subgroup
 * @return std::vector<typename playerType>
 */
std::vector<EvoIndividual *> CRDSimulator::_select_randomly(unsigned int size) {

    // copy group_indexes
    std::vector<unsigned int> group_indexes = _group_indexes;

    // reshuffle indexes
    std::random_shuffle(group_indexes.begin(), group_indexes.end());

    // Keep only size elements of the indices vector
    group_indexes.resize(size);
    for (unsigned int i = 0; i < size; i++) {
        _group[i] = &_population[group_indexes[i]];
    }

    return _group;
}

void CRDSimulator::_update_fitness_vector() {
    for (auto &fitness: _fitnessVector) {
        fitness = exp(this->beta * fitness);
    }

}

void CRDSimulator::_update_population_indexes() {
    /**
     * Implements Russian Roulette selection strategy
     */
    // the boost way
    boost::random::discrete_distribution<> dist(_fitnessVector);
    for (int i = 0; i < population_size; i++) {
        _population_indexes[i] = dist(_mt);
        // Initialize fitness and games played
        _population[i].init();
    }
}
