//
// Created by Elias Fernandez on 15/03/2018.
//

#include <iostream>
#include "../include/CRDSimulator.h"

CRDSimulator::CRDSimulator(unsigned int population_size) : population_size(population_size), beta(1.0) {
    // Default parameters
    group_size = 6;
    nb_games = 1000;
    double target_sum = 120;
    double risk = 0.9;
    unsigned int game_rounds = 10;
    double_t sigma_e = 0;
    double_t sigma_t = 0;

    // Initialize fitness vector
    _fitnessVector = std::vector<double>(population_size, 0.0);

    // Generate population
    // reserve memory for vector of pointers
    _population.reserve(population_size);
    _population_tmp.reserve(population_size);

    for (unsigned int i = 0; i < population_size; i++) {
        _population[i] = EvoIndividual(&_fitnessVector[i], CRDPlayer());
        _populationTypesHash.insert(
                std::make_pair(_population[i].player.strategy, strategy_fq(_population[i].player.strategy, 0))
        );
    }

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

//template<typename S, typename M, typename O, typename P>
//CRDSimulator<S, M, O, P>::CRDSimulator(S &selection, M &mutation, O *outputHandler, P &population, unsigned int population_size,
//                           unsigned int group_size) : _selection(selection), _mutation(mutation), population(population),
//                                                      OutputHandler(outputHandler), population_size(population_size),
//                                                      group_size(group_size) {
//    this->_group = std::vector<P>(group_size);
//    this->_group_indexes = std::vector<unsigned int>(this->population->size());
//    std::iota(this->_group_indexes.begin(), this->_group_indexes.end(), 0);
//
//}

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
//        std::pair<std::unordered_map<Strategy, StrategyFrequency, StrategyHasher>::iterator, bool> ok;
        for (i = 0; i < population_size; i++) {
            // Update player in population
            _population_tmp[i].player.strategy.copy(_population[_population_indexes[i]].player.strategy);

            // Mutate player strategy
            ++_population_tmp[i].player.strategy;

//            ok = _populationTypesHash.insert(
//                    std::make_pair(_population_tmp[i].strategy, strategy_fq(_population_tmp[i].strategy, 0))
//            );
//            if (!ok.second) {
//                _populationTypesHash.find(_population_tmp[i].strategy)++;
//            }
        }
        // Update new population
        _population = _population_tmp;


        // And finally mutate
//        _mutation->mutate(&this->population);

    }

}

/**
 * This method returns a subgroup formed by randomly selecting members (without replacement) from the population pool.
 * @param size : size of the subgroup
 * @return std::vector<typename playerType>
 */
std::vector<EvoIndividual *> CRDSimulator::_select_randomly(unsigned int size) {

    // copy group_indexes
    std::vector<unsigned int> group_indexes;
    group_indexes.swap(_group_indexes);

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
    for(auto & fitness: _fitnessVector) {
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
