/**
 * File: ParameterHandler.h
 * Author: Elias Fernandez
 * Date Created: 15/03/2018
 *
 * Simulator Class. Groups all parameters and functions of a specific Evolutionary Simulation.
 * It must contain the method evolve that is in charge of running the simulation.
 *
 * The simulator must be initialized with Selection and Mutation classes that specify how both processes are executed.
 * Both classes must have a select and mutate method (respectively) that receives a reference to a population as
 * argument.
 *
 * By default. Selection is WrightFisherModel and Mutation a base class that does not perform any kind of mutation.
 *
 * If initialized with debug = true, the simulator will log to cout/cerr by default or to a file if specified.
 *
 * By default, the output of the simulation is streamed to cout. If initialized with and OutputHandler, the output
 * will be directed to such class.
 *
 */

#ifndef DYRWIN_CRDSIMULATOR_H
#define DYRWIN_CRDSIMULATOR_H

#include <fstream>
#include <random>
#include <vector>
#include <unordered_map>
#include "WrightFisherModel.h"
#include "CollectiveRiskDilemma.h"
#include "SeedGenerator.h"

/**
 * Holds the frequency in which each strategy appears
 */
typedef struct StrategyFrequency {
    SequentialStrategy &strategy;
    unsigned int freq;

    StrategyFrequency &operator++() {
        ++freq;
        return *this;
    }

    StrategyFrequency operator++(int) {
        freq++;
        return *this;
    }

    StrategyFrequency &operator--() {
        --freq;
        return *this;
    }

    StrategyFrequency operator--(int) {
        freq--;
        return *this;
    }

    explicit StrategyFrequency(SequentialStrategy &strategy) :
            strategy(strategy), freq(0) {};

    StrategyFrequency(SequentialStrategy &strategy, unsigned int freq) :
            strategy(strategy), freq(freq) {};

    bool operator==(const StrategyFrequency &other) const {
        bool equal = true;
        for (size_t i = 0; i < strategy.rounds; i++) {
            if (strategy.round_strategies[i] == other.strategy.round_strategies[i]) {
                equal = false;
                break;
            }
        }
        return equal;
    }

    StrategyFrequency operator=(const StrategyFrequency &other) const {
        // Enforces that the reference to the random number generator is not changed
        return *this;
    }
} strategy_fq;

struct CRDSimData {
    int generation;
    double avg_fitness;
    double avg_contributions;
    double avg_threshold;
//    int32_t nb_contrib0;
//    int32_t nb_contribR;
//    int32_t nb_contribLessR;
//    int32_t nb_contribMoreR;
//    double avg_contributions_first_half;
//    double avg_contributions_second_half;

    void update(int generation, double avg_fitness, double avg_contributions, double avg_threshold) {
        this->generation = generation;
        this->avg_fitness = avg_fitness;
        this->avg_contributions = avg_contributions;
        this->avg_threshold = avg_threshold;
    }

//    void update(int generation, double avg_fitness, double avg_contributions, double avg_threshold, int nb_contrib0,
//                int nb_contribR, int nb_contribLessR, int nb_conotribMoreR) {
//        this->generation = generation;
//        this->avg_fitness = avg_fitness;
//        this->avg_contributions = avg_contributions;
//        this->avg_threshold = avg_threshold;
//        this->nb_contrib0 = nb_contrib0;
//        this->nb_contribR = nb_contribR;
//        this->nb_contribLessR = nb_contribLessR;
//        this->nb_contribMoreR = nb_conotribMoreR;
//    }

    std::string getCSVHeader() {
        return "generation,avg_fitness,avg_contributions,avg_threshold\n";
    }

    std::string getCSVData() {
        std::stringstream data;
        data << generation << "," << avg_fitness << "," << avg_contributions << "," << avg_threshold << "\n";
        return data.str();
    }
};

class CRDSimulator {
public:
    /**
     * Initializes the simulation with population size and a random generator
     * @param population_size
     * @param mt
     */
    CRDSimulator(unsigned int population_size, unsigned int group_size, unsigned int nb_games, unsigned int game_rounds,
                 double beta, double risk, double mu, double sigma, std::ofstream &output_file);

    virtual ~CRDSimulator() = default;

    void evolve(unsigned int generations);

    void printPopulation();

    void printCurrentStrategyFitness();

    void printAvgPopulationFitness(int generation);

    void printAvgContributions(int generation);

    void printAvgReachedThreshold(int generation);

    void printGenerationInfo(int generation);

    void saveGenerationData();

    void updateGenerationData(int generation);

    unsigned int population_size;
    unsigned int group_size;
    unsigned int nb_games;
    unsigned int game_rounds;
    double beta; // intensity of selection
    double risk;
    double mu;
    double sigma;
    double target_sum;
    double endowment;
    std::ofstream &outFile; // output stream

private:
    std::unordered_map<SequentialStrategy, strategy_fq, SequentialStrategyHasher> _populationTypesHash;
    std::vector<EvoIndividual> _population;  // holds the population at a given generation
    std::vector<EvoIndividual> _population_tmp; // holds a vector of players
    std::vector<double> _fitnessVector; // holds the fitness of each player in the population at a given generation
    std::vector<int> _population_indexes; // holds indexes to the population
    std::vector<EvoIndividual *> _group; // Vector of pointers to player objects
    std::vector<unsigned int> _group_indexes; // Holds indexes to the population for selecting a group
    std::vector<bool> _target_reached; // Holds the vector of target_reached each game during one generation
    std::vector<double> _contributions; // Holds the contributions at each game during one generation
    std::vector<int> _index_helper;

    CollectiveRiskDilemma *_game; // Pointer to Game class

    // Random generators
    std::mt19937_64 _mt{SeedGenerator::getSeed()};

    // Generation data
    CRDSimData _genData;

    /**
     * @brief updates the group vector by selecting random members of the population without replacement
     * @param size Size of the group to be selected randomly
     */
    void _select_randomly_with_replacement(unsigned int size);
    /**
     * @brief updates the group vector by selecting random members of the population with replacement
     * @param size
     */
    void _select_randomly(unsigned int size);

    /**
     * @brief updates the fitness vector with the fermi function
     */
    void _update_fitness_vector();

    /**
     * @brief updates the indexes to the next population randomly with weights equal to the fitness of each individual.
     */
    void _update_population_indexes();

    double _calculateAvgPopulationFitness();

    double _calculateAvgContributions();

    double _calculateAvgReachedThreshold();
};


#endif //DYRWIN_CRDSIMULATOR_H
