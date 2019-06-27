//
// Created by Elias Fernandez on 2019-05-31.
//
#include <cmath>
#include <unordered_map>
#include <fstream>

#include <Dyrwin/SED/behaviors/CrdBehaviors.hpp>
#include <Dyrwin/Types.h>
#include <Dyrwin/CommandLineParsing.h>
#include <Dyrwin/SED/games/AbstractGame.hpp>
#include <Dyrwin/SED/games/CrdGame.hpp>
#include <Dyrwin/SED/games/CrdGameTU.hpp>
#include <Dyrwin/SED/PairwiseMoran.hpp>


int main(int argc, char *argv[]) {

    // First we define a vector of possible behaviors
    size_t nb_strategies = EGTTools::SED::CRD::nb_strategies;
    size_t pop_size;
    size_t nb_generations;
    size_t group_size;
//    size_t die, birth;
    size_t nb_rounds, min_rounds, endowment, threshold;
    double beta;
    double mu;
    double risk;
    bool timing_uncertainty;
    double p = 0.0;
    EGTTools::SED::StrategyCounts strategies(EGTTools::SED::CRD::nb_strategies);
    Options options;

    options.push_back(
            makeDefaultedOption<size_t>("generations,g", &nb_generations, "set the number of generations", 1000u));
    options.push_back(makeDefaultedOption<size_t>("popSize,Z", &pop_size, "set the size of the population", 100u));
    options.push_back(
            makeRequiredOption<std::vector<size_t>>("strategies,s", &strategies, "the counts of each strategy"));
    options.push_back(makeDefaultedOption<double>("mu,u", &mu, "set mutation rate", 0.05));
    options.push_back(makeDefaultedOption<double>("beta,b", &beta, "set intensity of selection", 0.001));
    options.push_back(makeDefaultedOption<size_t>("groupSize,n", &group_size, "group size", 6));
    options.push_back(makeDefaultedOption<size_t>("nbRounds,t", &nb_rounds, "number of rounds", 10));
    options.push_back(makeDefaultedOption<size_t>("minRounds,m", &min_rounds, "minimum number of rounds", 8));
    options.push_back(makeDefaultedOption<size_t>("endowment,e", &endowment, "endowment", 40));
    options.push_back(makeDefaultedOption<size_t>("target,p", &threshold, "threshold", 120));
    options.push_back(makeDefaultedOption<double>("risk,r", &risk, "risk", 0.9));
    options.push_back(makeDefaultedOption<bool>("tu,d", &timing_uncertainty, "timing uncertainty", false));
    if (!parseCommandLine(argc, argv, options))
        return 1;

    EGTTools::SED::AbstractGame *game;

    if (timing_uncertainty) {
        p = 1 / static_cast<double>(nb_rounds - min_rounds + 1);
        EGTTools::TimingUncertainty<std::mt19937_64> tu(p);
        game = new EGTTools::SED::CRD::CrdGameTU(endowment, threshold, min_rounds, group_size, risk, tu);
    } else {
        game = new EGTTools::SED::CRD::CrdGame(endowment, threshold, nb_rounds, group_size, risk);
    }

    // Initialise selection mutation process
    auto smProcess = EGTTools::SED::PairwiseMoran(pop_size, *game);

    // Save payoffs
    std::ofstream file("payoffs.txt", std::ios::out | std::ios::trunc);
    if (file.is_open()) {
        file << "Payoffs for each type of player and each possible state:" << std::endl;
        file << "rows: cooperator, defector, altruist, reciprocal, compensator" << std::endl;
        file << "cols: all possible group compositions starting at (0, 0, 0, 0, group_size)" << std::endl;
        file << game->payoffs() << std::endl;
        file << "group_size = " << group_size << std::endl;
        file << "population_size = " << pop_size << std::endl;
        file << "beta = " << beta << std::endl;
        file << "mu = " << mu << std::endl;
        file << "nb_generations = " << nb_generations << std::endl;
        file << "timing_uncertainty = " << timing_uncertainty << std::endl;
        if (timing_uncertainty) {
            file << "min_rounds = " << min_rounds << std::endl;
            file << "mean_rounds = " << nb_rounds << std::endl;
            file << "p = " << p << std::endl;
        } else file << "nb_rounds = " << nb_rounds << std::endl;
        file << "risk = " << risk << std::endl;
        file << "endowment = " << endowment << std::endl;
        file << "threshold = " << threshold << std::endl;
        file << "initial state = (";
        for (size_t i = 0; i < nb_strategies; ++i)
            file << strategies[i] << ", ";
        file << ")" << std::endl;
    }

    std::cout << "initial state: (";
    for (size_t i = 0; i < nb_strategies; ++i)
        std::cout << strategies[i] << ", ";
    std::cout << ")" << std::endl;

    EGTTools::VectorXui init_state = EGTTools::VectorXui(nb_strategies);
    for (size_t i = 0; i < nb_strategies; ++i) init_state(i) = strategies[i];

    auto final_strategies = smProcess.evolve(nb_generations, beta, mu, init_state);

    std::cout << "final state: (";
    for (size_t i = 0; i < nb_strategies; ++i)
        std::cout << final_strategies[i] << ", ";
    std::cout << ")" << std::endl;

    if (file.is_open()) {
        file << "final state: (";
        for (size_t i = 0; i < nb_strategies; ++i)
            file << final_strategies[i] << ", ";
        file << ")" << std::endl;
        file.close();
    }
}
