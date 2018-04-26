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


#include <vector>
#include <unordered_map>
#include <boost/random.hpp>
#include "WrightFisherModel.h"
#include "CollectiveRiskDilemma.h"

/**
 * Holds the frequency in which each strategy appears
 */
typedef struct StrategyFrequency {
    SequentialStrategy strategy;
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

    StrategyFrequency(SequentialStrategy strategy, unsigned int freq) :
            strategy(strategy), freq(freq) {};

    bool operator==(const StrategyFrequency &other) const {
        bool equal = true;
        for (size_t i=0; i < strategy.rounds; i++) {
            if (strategy.round_strategies[i] == other.strategy.round_strategies[i]) {
                equal = false;
                break;
            }
        }
        return equal;
    }
} strategy_fq;

class CRDSimulator {
public:
    /**
     * Initializes the simulation with population size and a random generator
     * @param population_size
     * @param mt
     */
    CRDSimulator(unsigned int population_size, boost::mt19937& mt);
    virtual ~CRDSimulator() {};

    void evolve(unsigned int generations);


    unsigned int population_size;
    unsigned int group_size;
    unsigned int nb_games;
    unsigned int game_rounds;
    double beta; // intensity of selection

private:
    std::unordered_map<SequentialStrategy, strategy_fq, SequentialStrategyHasher> _populationTypesHash;
    std::vector<EvoIndividual> _population;  // holds the population at a given generation
    std::vector<EvoIndividual> _population_tmp; // holds a vector of players
    std::vector<double> _fitnessVector; // holds the fitness of each player in the population at a given generation
    std::vector<int> _population_indexes; // holds indexes to the population
    std::vector<EvoIndividual *> _group; // Vector of pointers to player objects
    std::vector<unsigned int> _group_indexes; // Holds indexes to the population for selecting a group

    CollectiveRiskDilemma *_game; // Pointer to Game class

    // Random generators
    boost::mt19937& _mt;
    boost::uniform_real<> _uniform = boost::uniform_real<>(0, 1);
    boost::variate_generator<boost::mt19937 &, boost::uniform_real<> > _rng =
            boost::variate_generator<boost::mt19937 &, boost::uniform_real<> >(_mt, _uniform);

    std::vector<EvoIndividual *>
    _select_randomly(unsigned int size); // Selects size individuals randomly with replacement from the population
    void _update_fitness_vector();

    void _update_population_indexes();
};


#endif //DYRWIN_CRDSIMULATOR_H
