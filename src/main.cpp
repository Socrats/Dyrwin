#include <iostream>
#include <fstream>
#include "../include/Dyrwin/CRDSimulator.h"
#include "../include/Dyrwin/CommandLineParsing.h"
#include "../include/Dyrwin/Utils.h"

int main(int argc, char *argv[]) {

    // Default parameters
    unsigned int generations;
    unsigned int population_size;
    unsigned int group_size;
    unsigned int nb_games;
    unsigned int game_rounds;
    double risk;
    double mu;
    double sigma;
    double beta;
    std::string filename;
    Options options;

    // Setup options
    options.push_back(makeDefaultedOption("generations,g", &generations, "set the number of generations", 1000u));
    options.push_back(makeDefaultedOption("popSize,N", &population_size, "set the size of the population", 100u));
    options.push_back(makeDefaultedOption("groupSize,M", &group_size, "set the size of the group", 6u));
    options.push_back(
            makeDefaultedOption("nbGames,G", &nb_games, "set the number of games to be played at each generations",
                                1000u));
    options.push_back(makeDefaultedOption("gameRounds,t", &game_rounds, "set the number of rounds of each game", 10u));
    options.push_back(makeDefaultedOption<double>("risk,r", &risk, "set up the risk parameter", 0.9f));
    options.push_back(makeDefaultedOption<double>("mu,m", &mu, "set up the probability of mutation", 0.03f));
    options.push_back(makeDefaultedOption<double>("sigma,s", &sigma, "set up the threshold mutation", 0.15f));
    options.push_back(makeDefaultedOption<double>("beta,b", &beta, "set up intensity of selection", 1.0f));
    options.push_back(makeRequiredOption("output,o", &filename, "set the final output file"));

    if (!parseCommandLine(argc, argv, options))
        return 1;

    // Create output stream
    std::ofstream outFile;
    outFile.open(filename, std::ios::out | std::ios::binary);

    std::cout << "Starting Simulation: " << std::endl;

    clock_t tStart = clock();

    CRDSimulator simulator(population_size, group_size, nb_games, game_rounds, beta, risk, mu, sigma, outFile);

    std::cout << "Starting evolution" << std::endl;

    simulator.evolve(generations);

    std::cout << "Finished Simulation" << std::endl;

    outFile.close();

    // Convert data to CSV
    std::stringstream ss;
    ss << filename << ".csv";
    std::string csvfilename = ss.str();
    convert2CSV<CRDSimData>(filename, csvfilename);

    printf("\nTime taken: %.8fs\n", (double) (clock() - tStart) / CLOCKS_PER_SEC);
    return 0;
}