//
// Created by Elias Fernandez on 15/03/2018.
//

#include <iostream>
#include "../include/Dyrwin/CRDSimulator.h"

CRDSimulator::CRDSimulator(unsigned int population_size) : population_size(population_size), beta(1.0) {
    // Default parameters
    group_size = 6;
    nb_games = 1000;
    game_rounds = 10;
    double target_sum = group_size * game_rounds;
    double endowment = (double) (2 * game_rounds);
    double risk = 0.9;
    double mu = 0.03;
    double sigma = 0.15;

    // Initialize fitness vector
    _fitnessVector = std::vector<double>(population_size, 0.0);

    // Generate population
    // reserve memory for vector of pointers
    _population.reserve(population_size);
    _population_tmp.reserve(population_size);
    _populationTypesHash.reserve(population_size);


    for (unsigned int i = 0; i < population_size; i++) {
        _population.push_back(EvoIndividual(&_fitnessVector[i], *(new CRDPlayer(mu, sigma))));
//        auto it = _populationTypesHash.find(_population.back().player.strategy);
//        if (it != _populationTypesHash.end()) {
//            it->second++;
//        } else {
//            auto ok = _populationTypesHash.insert(std::make_pair(_population.back().player.strategy,
//                                                                 strategy_fq(_population.back().player.strategy, 1)));
//        }
    }

    // Make sure that population vector has been initialized
    assert(!_population.empty());
    assert(!_population[0].player.strategy.round_strategies.empty());
//    assert(!_populationTypesHash.empty());
//    auto it = _populationTypesHash.find(_population.back().player.strategy);
//    if (it == _populationTypesHash.end()) {
//        std::cout << "Strategy is not in the table" << std::endl;
//    }
//    assert(!_populationTypesHash.find(_population[0].player.strategy)->second.strategy.round_strategies.empty());

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
    _game = new CollectiveRiskDilemma(3, group_size, target_sum, risk, game_rounds, endowment);

    // Initialize variables that hold information of a generation
    _target_reached = std::vector<bool>(nb_games);
    _contributions = std::vector<double>(nb_games);

}

/**
 * Evolves the population through selection and mutation operators.
 * @param generations - numbers of generations through which the population is evolved
 */
void CRDSimulator::evolve(unsigned int generations) {
    int i, j;

    for (j = 0; j < generations; j++) {
        // First play games and calculate fitness
        for (i = 0; i < nb_games; i++) {
            // Generate group
            auto group = _select_randomly(group_size);
            auto game_result = _game->run(game_rounds, group);
            _target_reached[i] = game_result.met_threshold;
            _contributions[i] = game_result.public_account;
        }
        printGenerationInfo(j);
//        printPopulation();

        // Then apply selection - Wright-Fisher Process
        // First update fitnessVector
        _update_fitness_vector();
        // Then, generate new population_indexes drawn at random with fitnessVector as weights
        _update_population_indexes();
        // Finally, update population
        for (i = 0; i < population_size; i++) {
            // Update player in population
            _population_tmp[i].player.strategy.copy(_population[_population_indexes[i]].player.strategy);

            // Reduce strategy frequency
//            _populationTypesHash[_population_tmp[i].player.strategy]--;

            // Mutate player strategy
            ++_population_tmp[i].player.strategy;

//            auto ok = _populationTypesHash.insert(
//                    std::make_pair(_population_tmp[i].player.strategy,
//                                   strategy_fq(_population_tmp[i].player.strategy, 1))
//            );
//
//            if (!ok.second) {
//                std::cout << " " << ok.second << std::endl;
////                _populationTypesHash[_population_tmp[i].player.strategy]++;
//            }
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
    // Uniform int distribution
    std::uniform_int_distribution<unsigned long int> dist(0, _population.size() - 1);
    for (unsigned int i = 0; i < size; i++) {
        _group[i] = &_population[dist(_mt)];
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
    std::discrete_distribution<> dist(_fitnessVector.begin(), _fitnessVector.end());
    for (int i = 0; i < population_size; i++) {
        _population_indexes[i] = dist(_mt);
        // Initialize fitness and games played
        _population[i].init();
    }
}

void CRDSimulator::printPopulation() {
    for (auto const &individual: _population) {
        std::cout << std::endl << "---------------------------" << std::endl;
        for (auto const &strategy: individual.player.strategy.round_strategies) {
            std::cout << "(" << strategy.first << ",";
            std::cout << strategy.second << ",";
            std::cout << strategy.threshold << "), ";
        }
    }
}

void CRDSimulator::printCurrentStrategyFitness() {

}

void CRDSimulator::printAvgPopulationFitness(int generation) {
    std::cout << "[Gen " << generation << "] Avg. Fitness of the population: " << _calculateAvgPopulationFitness()
              << std::endl;
}

void CRDSimulator::printAvgContributions(int generation) {
    std::cout << "[Gen " << generation << "] Avg. Contributions of the population: "
              << _calculateAvgContributions() << std::endl;
}

void CRDSimulator::printAvgReachedThreshold(int generation) {
    std::cout << "[Gen " << generation << "] Avg. Fitness of the population: " << _calculateAvgReachedThreshold()
              << std::endl;
}

double CRDSimulator::_calculateAvgPopulationFitness() {
    double average = std::accumulate(_fitnessVector.begin(), _fitnessVector.end(), 0.0) / _fitnessVector.size();
    return average / (double) (2 * game_rounds);
}

double CRDSimulator::_calculateAvgContributions() {
    double average = std::accumulate(_contributions.begin(), _contributions.end(), 0.0) / _contributions.size();
    return average / (double) (group_size * (2 * game_rounds));
}

double CRDSimulator::_calculateAvgReachedThreshold() {
    return std::accumulate(_target_reached.begin(), _target_reached.end(), 0.0) / _target_reached.size();
}

void CRDSimulator::printGenerationInfo(int generation) {
    std::cout << "[Gen " << generation << "] Fitness: " << _calculateAvgPopulationFitness() << " Contributions: "
              << _calculateAvgContributions() << " Threshold: " << _calculateAvgReachedThreshold() << std::endl;
}
