//
//  main.cpp
// Created by Elias F.
//

#include <iostream>
#include <vector>
#include <chrono>
#include <Dyrwin/RL/BatchQLearningAgent.h>
#include <Dyrwin/RL/QLearningAgent.h>
#include <Dyrwin/RL/RothErevAgent.h>
#include <Dyrwin/RL/HistericQLearningAgent.hpp>
#include <Dyrwin/RL/CRDGame.h>
#include <Dyrwin/RL/TimingUncertainty.hpp>
#include <Dyrwin/RL/Utils.h>
#include <Dyrwin/CommandLineParsing.h>

using namespace std::chrono;
using namespace EGTTools::RL;

template<typename A, typename B = void>
void reinforceRothErev(double &pool, size_t &success, double rnd_value, double &cataclysm, double &threshold,
                       size_t &final_round, CRDGame<A, B> &Game, EGTTools::RL::Population &group) {
    if (pool >= threshold) {
        Game.reinforcePath(group, final_round);
        ++success;
    } else if (rnd_value > cataclysm) Game.reinforcePath(group, final_round);
    else Game.setPayoffs(group, 0);
}

template<typename A, typename B = void>
void reinforceBatchQLearning(double &pool, size_t &success, double rnd_value, double &cataclysm, double &threshold,
                             size_t &final_round, CRDGame<A, B> &Game, EGTTools::RL::Population &group) {

    if (pool >= threshold) ++success;
    else if (rnd_value < cataclysm) Game.setPayoffs(group, 0);

    Game.reinforcePath(group, final_round);
}

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
    double lambda;
    std::string filename;
    std::string agent_type;
    Options options;
    CRDGame<Agent, EGTTools::TimingUncertainty<std::mt19937_64>> Game;
    // Random generators
    std::mt19937_64 generator{EGTTools::Random::SeedGenerator::getInstance().getSeed()};

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
    options.push_back(makeDefaultedOption<double>("lambda,l", &lambda, "discount factor", .99));

    if (!parseCommandLine(argc, argv, options))
        return 1;

    unsigned endowment = 2 * mean_rounds; //per player
    auto threshold = static_cast<double>(mean_rounds * group_size);
    auto donations = std::vector<size_t>(actions);
    auto end_probability = static_cast<double>(1. / (1 + (mean_rounds - min_rounds)));
    EGTTools::TimingUncertainty<std::mt19937_64> tu(end_probability, max_rounds);
    for (unsigned i = 0; i < actions; i++) donations[i] = i;
    void (*reinforce)(double &, size_t &, double, double &, double &, size_t &,
                      CRDGame<Agent, EGTTools::TimingUncertainty<std::mt19937_64>> &, EGTTools::RL::Population &);

    // Initialize agents depending on command option
    EGTTools::RL::Population group;
    if (agent_type == "rothErev") {
        reinforce = &reinforceRothErev<Agent, EGTTools::TimingUncertainty<std::mt19937_64>>;
        for (unsigned i = 0; i < group_size; i++) {
            group.push_back(std::make_unique<Agent>(max_rounds, actions, max_rounds, endowment));
        }
    } else if (agent_type == "rothErevLambda") {
        reinforce = &reinforceBatchQLearning<Agent, EGTTools::TimingUncertainty<std::mt19937_64>>;
        for (unsigned i = 0; i < group_size; i++) {
            group.push_back(std::make_unique<RothErevAgent>(max_rounds, actions, max_rounds, endowment, lambda, temperature));
        }
    } else if (agent_type == "QLearning") {
        reinforce = &reinforceBatchQLearning<Agent, EGTTools::TimingUncertainty<std::mt19937_64>>;
        for (unsigned i = 0; i < group_size; i++) {
            group.push_back(std::make_unique<QLearningAgent>(max_rounds, actions, max_rounds, endowment, alpha, lambda, temperature));
        }
    } else if (agent_type == "HistericQLearning") {
        reinforce = &reinforceBatchQLearning<Agent, EGTTools::TimingUncertainty<std::mt19937_64>>;
        for (unsigned i = 0; i < group_size; i++) {
            group.push_back(std::make_unique<HistericQLearningAgent>(max_rounds, actions, max_rounds, endowment, alpha, beta, temperature));
        }
    } else {
        reinforce = &reinforceBatchQLearning<Agent, EGTTools::TimingUncertainty<std::mt19937_64>>;
        for (unsigned i = 0; i < group_size; i++) {
            group.push_back(std::make_unique<BatchQLearningAgent>(max_rounds, actions, max_rounds, endowment, alpha, temperature));
        }
    }

    // Calculate execution time
    high_resolution_clock::time_point t1 = high_resolution_clock::now();

    //testing one group
    for (unsigned int step = 0; step < attempts; step++) {
        size_t success = 0;
        double avgpayoff = 0.;
        double avg_rounds = 0.;
        for (unsigned int game = 0; game < games; game++) {
            // First we play the game
            auto[pool, final_round] = Game.playGame(group, donations, min_rounds, tu);
            avgpayoff += (Game.playersPayoff(group) / double(group_size));
            reinforce(pool, success, EGTTools::probabilityDistribution(generator), cataclysm, threshold, final_round,
                      Game, group);
            avg_rounds += final_round;
        }
        std::cout << (success / double(games)) << " " << (avgpayoff / double(games)) << " "
                  << (avg_rounds / double(games)) << std::endl;
        Game.calcProbabilities(group);
        Game.resetEpisode(group);
        //if(success == games) break;
    }
    Game.printGroup(group);

    // Print execution time
    high_resolution_clock::time_point t2 = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(t2 - t1).count();
    std::cout << "Execution time: " << duration << std::endl;

    return 0;
}