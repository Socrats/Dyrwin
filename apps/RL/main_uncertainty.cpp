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
    size_t max_rounds;
    size_t mean_rounds;
    size_t min_rounds;
    size_t attempts;
    size_t games;
    double cataclysm;
    double alpha, beta;
    double temperature;
    double lambda, epsilon;
    double threshold;
    double endowment;
    double p;
    std::string filename;
    std::string agent_type;
    Options options;

    // Setup options
    options.push_back(makeDefaultedOption<size_t>("groupSize,M", &group_size, "set the group size", 6));
    options.push_back(makeDefaultedOption<size_t>("nbActions,p", &actions, "set the number of actions", 3));
    options.push_back(makeDefaultedOption<size_t>("nbAttempts,E", &attempts, "set the number of epochs", 1000));
    options.push_back(makeDefaultedOption<size_t>("nbGames,G", &games, "set the number of games per epoch", 1000));
    options.push_back(makeDefaultedOption<size_t>("maxRounds,z", &max_rounds, "set the max number of rounds", 20));
    options.push_back(makeDefaultedOption<size_t>("avgRounds,v", &mean_rounds, "set avg number of rounds", 10));
    options.push_back(makeDefaultedOption<size_t>("minRounds,t", &min_rounds, "set the min number of rounds", 8));
    options.push_back(
            makeDefaultedOption<std::string>("agentType,A", &agent_type, "set agent type", "BatchQLearningAgent"));
    options.push_back(makeDefaultedOption<double>("risk,r", &cataclysm, "set the risk probability", 0.9));
    options.push_back(makeDefaultedOption<double>("alpha,a", &alpha, "learning rate", 0.3));
    options.push_back(makeDefaultedOption<double>("beta,b", &beta, "learning rate", 0.03));
    options.push_back(
            makeDefaultedOption<double>("temperature,T", &temperature, "temperature of the boltzman distribution",
                                        10.));
    options.push_back(makeDefaultedOption<double>("lambda,l", &lambda, "discount factor", .01));
    options.push_back(makeDefaultedOption<double>("epsilon,e", &epsilon, "discount factor", .01));

    if (!parseCommandLine(argc, argv, options))
        return 1;

    endowment = 2 * mean_rounds; //per player
    threshold = static_cast<double>(mean_rounds * group_size);
    p = static_cast<double>(1. / (1 + (mean_rounds - min_rounds)));
    ActionSpace available_actions = ActionSpace(actions);
    for (size_t i = 0; i < actions; ++i) available_actions[i] = i;

    // Initialize agents depending on command option
    std::vector<double> args;
    if (agent_type == "rothErevLambda") {
        args.push_back(lambda);
        args.push_back(epsilon);
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

    // Calculate execution time
    high_resolution_clock::time_point t1 = high_resolution_clock::now();

    try {
        CRDSim sim(attempts, games, mean_rounds, actions, group_size, cataclysm, endowment, threshold,
                   available_actions, agent_type, args);
        EGTTools::Matrix2D results = sim.runTimingUncertainty(attempts, games, min_rounds, mean_rounds, max_rounds, p,
                                                              cataclysm, args, "milinski");
        std::cout << "success: " << results.row(0) << std::endl;
//        std::cout << "avg_contrib: " << results.row(1) << std::endl;
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