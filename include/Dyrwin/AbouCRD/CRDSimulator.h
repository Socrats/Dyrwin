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

#ifndef DYRWIN_ABOUCRD_CRDSIMULATOR_H
#define DYRWIN_ABOUCRD_CRDSIMULATOR_H

#include <fstream>
#include <random>
#include <vector>
#include <iostream>
#include <unordered_map>
#include <cassert>
#include <Dyrwin/AbouCRD/CollectiveRiskDilemma.h>
#include <Dyrwin/AbouCRD/Utils.h>
#include <Dyrwin/AbouCRD/DataStruct.hpp>
#include <Dyrwin/SeedGenerator.h>


namespace EGTTools::AbouCRD {

    class CRDSimulator {
    public:
        /**
         * Initializes the simulation with population size and a random generator
         * @param population_size
         * @param mt
         */
        CRDSimulator(unsigned int population_size, unsigned int group_size, unsigned int nb_games,
                     unsigned int game_rounds,
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
        std::vector<EvoIndividual> _population;  // holds the population at a given generation
        std::vector<EvoIndividual> _population_tmp; // holds a vector of players
        std::vector<double> _fitnessVector; // holds the fitness of each player in the population at a given generation
        std::vector<int> _population_indexes; // holds indexes to the population
        std::vector<EvoIndividual *> _group; // Vector of pointers to player objects
        std::vector<unsigned int> _group_indexes; // Holds indexes to the population for selecting a group
        std::vector<bool> _target_reached; // Holds the vector of target_reached each game during one generation
        std::vector<double> _contributions; // Holds the contributions at each game during one generation

        CollectiveRiskDilemma *_game; // Pointer to Game class

        // Random generators
        std::mt19937_64 _mt{EGTTools::Random::SeedGenerator::getInstance().getSeed()};

        // Generation data
        CRDSimData _genData{};

        /**
         * @brief Selects size individuals randomly with replacement from the population
         *
         * Generates a vector of pointers to random elements of the population.
         *
         * @param size Size of the group to be selected randomly
         * @return vector of pointers to the population with Size size.
         */
//    std::vector<EvoIndividual *> _select_randomly(unsigned int size);
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
}


#endif //DYRWIN_ABOUCRD_CRDSIMULATOR_H
