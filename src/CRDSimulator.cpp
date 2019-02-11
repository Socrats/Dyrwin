//
// Created by Elias Fernandez on 15/03/2018.
//

#include <iostream>
#include "../include/Dyrwin/crd/CRDSimulator.h"

CRDSimulator::CRDSimulator(unsigned int population_size, unsigned int group_size, unsigned int nb_games,
                           unsigned int game_rounds,
                           double beta, double risk, double mu, double sigma, std::ofstream &output_file)
        : population_size(population_size), group_size(group_size), nb_games(nb_games), game_rounds(game_rounds),
          beta(beta), risk(risk), mu(mu), sigma(sigma), outFile(output_file) {

    target_sum = (double) (group_size * game_rounds);
    endowment = (double) (2 * game_rounds);

    // Initialize fitness vector
    _fitnessVector = std::vector<double>(population_size, 0.0);

    // Generate population
    // reserve memory for vector of pointers
    _population.reserve(population_size);
    _population_tmp.reserve(population_size);

    for (unsigned int i = 0; i < population_size; i++) {
        _population.emplace_back(&_fitnessVector[i], mu, sigma);
    }

    // Make sure that population vector has been initialized
    assert(!_population.empty());
    assert(!_population[0].player.strategy.round_strategies.empty());

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

    // Index helper
    _index_helper.reserve(population_size);
    for (int i=0; i<population_size; ++i)
    {
        _index_helper.push_back(i);

    }

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
            _select_randomly(group_size);
            auto game_result = _game->run(game_rounds, _group);
            _target_reached[i] = game_result.met_threshold;
            _contributions[i] = game_result.public_account;
        }
        updateGenerationData(j);
//        printGenerationInfo(j);
        saveGenerationData();
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

            // Mutate player strategy
            ++_population_tmp[i].player.strategy;

        }

        // Update new population
        _population.swap(_population_tmp);

    }

}

/**
 * This method returns a subgroup formed by randomly selecting members (with replacement) from the population pool.
 * @param size : size of the subgroup
 * @return std::vector<typename playerType>
 */
void CRDSimulator::_select_randomly_with_replacement(unsigned int size) {
    // Uniform int distribution
    std::uniform_int_distribution<unsigned long int> dist(0, _population.size() - 1);
    for (unsigned int i = 0; i < size; i++) {
        _group[i] = &_population[dist(_mt)];
    }
}

/**
 * This method returns a subgroup formed by randomly selecting members (without replacement) from the population pool.
 * @param size : size of the subgroup
 * @return std::vector<typename playerType>
 */
void CRDSimulator::_select_randomly(unsigned int size) {
    auto indexes = _index_helper;

    std::shuffle (_index_helper.begin(), _index_helper.end(), _mt); //this shuffles the individuals

    // Uniform int distribution
//    std::uniform_int_distribution<unsigned long int> dist(0, _population.size() - 1);
    for (unsigned int i = 0; i < size; i++) {
//        _group[i] = &_population[dist(_mt)];
        _group[i] = &_population[indexes[i]];
    }
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
    return average / endowment;
}

double CRDSimulator::_calculateAvgContributions() {
    double average = std::accumulate(_contributions.begin(), _contributions.end(), 0.0) / _contributions.size();
    return average / (group_size * endowment);
}

double CRDSimulator::_calculateAvgReachedThreshold() {
    return std::accumulate(_target_reached.begin(), _target_reached.end(), 0.0) / _target_reached.size();
}

void CRDSimulator::printGenerationInfo(int generation) {
    std::cout << "[Gen " << generation << "] Fitness: " << _genData.avg_fitness << " Contributions: "
              << _genData.avg_contributions << " Threshold: " << _genData.avg_threshold << std::endl;
}

void CRDSimulator::saveGenerationData() {
    outFile.write( (char *)&_genData, sizeof(CRDSimData));
}

void CRDSimulator::updateGenerationData(int generation) {
    _genData.update(generation, _calculateAvgPopulationFitness(), _calculateAvgContributions(),
                   _calculateAvgReachedThreshold());
}
