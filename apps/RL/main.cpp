#include <memory>

#include <memory>

//
//  main.cpp
// Created by Elias F.
//

#include <iostream>
#include <vector>
#include <chrono>
#include <Dyrwin/RL/CrdSim.hpp>
#include <Dyrwin/CommandLineParsing.h>

using namespace std::chrono;
using namespace EGTTools::RL;

int main(int argc, char *argv[]) {
    //parameters
    size_t group_size;
    size_t actions; //0, 2 and 4
    size_t rounds;
    size_t attempts;
    size_t games;
    double cataclysm;
    double alpha, beta;
    double temperature;
    double lambda;
    double threshold;
    double endowment;
    std::string filename;
    std::string agent_type;
    Options options;

    // Setup options
    options.push_back(makeDefaultedOption<size_t>("groupSize,M", &group_size, "set the group size", 6));
    options.push_back(makeDefaultedOption<size_t>("nbActions,p", &actions, "set the number of actions", 3));
    options.push_back(makeDefaultedOption<size_t>("nbAttempts,E", &attempts, "set the number of epochs", 1000));
    options.push_back(makeDefaultedOption<size_t>("nbGames,G", &games, "set the number of games per epoch", 1000));
    options.push_back(makeDefaultedOption<size_t>("nbRounds,t", &rounds, "set the number of rounds", 10));
    options.push_back(
            makeDefaultedOption<std::string>("agentType,A", &agent_type, "set agent type", "BatchQLearning"));
    options.push_back(makeDefaultedOption<double>("risk,r", &cataclysm, "set the risk probability", 0.9));
    options.push_back(makeDefaultedOption<double>("alpha,a", &alpha, "learning rate", 0.3));
    options.push_back(makeDefaultedOption<double>("beta,b", &beta, "learning rate", 0.03));
    options.push_back(
            makeDefaultedOption<double>("temperature,T", &temperature, "temperature of the boltzman distribution",
                                        10.));
    options.push_back(makeDefaultedOption<double>("lambda,l", &lambda, "discount factor", .99));

    if (!parseCommandLine(argc, argv, options))
        return 1;

    // Calculate execution time
    high_resolution_clock::time_point t1 = high_resolution_clock::now();

    // Initialize agents depending on command option
    std::vector<double> args;
    if (agent_type == "rothErevLambda") {
        args.push_back(lambda);
        args.push_back(temperature);
    } else if (agent_type == "QLearning") {
        args.push_back(alpha);
        args.push_back(lambda);
        args.push_back(temperature);
    } else if (agent_type == "HistericQLearning") {
        args.push_back(alpha);
        args.push_back(beta);
        args.push_back(temperature);
    } else if (agent_type == "BatchQLearning") {
        args.push_back(alpha);
        args.push_back(temperature);
    }

    endowment = 2 * rounds;
    threshold = group_size * rounds;
    ActionSpace available_actions = ActionSpace(actions);
    for (size_t i = 0; i < actions; ++i) available_actions[i] = i;

    try {
        CRDSim sim(attempts, games, rounds, actions, group_size, cataclysm, endowment, threshold, available_actions, agent_type, args);
        EGTTools::Matrix2D results = sim.run(attempts, games);
        std::cout << "success: " << results.row(0) << std::endl;
        std::cout << "avg_contrib: " << results.row(1) << std::endl;
        sim.Game.printGroup(sim.population);
    } catch (std::invalid_argument &e) {
        std::cerr << "\033[1;31m[EXCEPTION] Invalid argument: " << e.what() << "\033[0m" << std::endl;
        return -1;
    }

    // Print execution time
    high_resolution_clock::time_point t2 = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(t2 - t1).count();
    std::cout << "Execution time: " << duration << std::endl;

    return 0;
}