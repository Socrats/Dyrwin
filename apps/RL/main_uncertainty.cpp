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
#include <Dyrwin/CommandLineParsing.h>

using namespace std::chrono;
using namespace EGTTools::RL;

template<typename A, typename B = void>
void reinforceRothErev(double &pool, unsigned &success, double rnd_value, double &cataclysm, double &threshold,
                       CRDGame<A, B> &Game, std::vector<A *> &group) {
    if (pool >= threshold) {
        Game.reinforcePath(group);
        success++;
    } else if (rnd_value > cataclysm) Game.reinforcePath(group);
    else Game.setPayoffs(group, 0);
}

template<typename A, typename B = void>
void reinforceBatchQLearning(double &pool, unsigned &success, double rnd_value, double &cataclysm, double &threshold,
                             CRDGame<A, B> &Game, std::vector<A *> &group) {

    if (pool >= threshold) success++;
    else if (rnd_value < cataclysm) Game.setPayoffs(group, 0);

    Game.reinforcePath(group);
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
    CRDGame <Agent, EGTTools::TimingUncertainty<std::mt19937_64>> Game;
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
    double end_probability = static_cast<double>(1./(1 + (mean_rounds - min_rounds)));
    EGTTools::TimingUncertainty<std::mt19937_64> tu(end_probability, max_rounds);
    for (unsigned i = 0; i < actions; i++) donations[i] = i;
    void (*reinforce)(double &, unsigned &, double, double &, double &,
                      CRDGame <Agent, EGTTools::TimingUncertainty<std::mt19937_64>> &, std::vector<Agent *> &);

    // Initialize agents depending on command option
    std::vector < Agent * > group;
    if (agent_type == "rothErev") {
        reinforce = &reinforceRothErev;
        for (unsigned i = 0; i < group_size; i++) {
            auto a = new Agent(max_rounds, actions, endowment);
            group.push_back(a);
        }
    } else if (agent_type == "rothErevLambda") {
        reinforce = &reinforceBatchQLearning;
        for (unsigned i = 0; i < group_size; i++) {
            auto a = new RothErevAgent(max_rounds, actions, endowment, lambda, temperature);
            group.push_back(a);
        }
    } else if (agent_type == "QLearning") {
        reinforce = &reinforceBatchQLearning;
        for (unsigned i = 0; i < group_size; i++) {
            auto a = new QLearningAgent(max_rounds, actions, endowment, alpha, lambda, temperature);
            group.push_back(a);
        }
    } else if (agent_type == "HistericQLearning") {
        reinforce = &reinforceBatchQLearning;
        for (unsigned i = 0; i < group_size; i++) {
            auto a = new HistericQLearningAgent(max_rounds, actions, endowment, alpha, beta, temperature);
            group.push_back(a);
        }
    } else {
        reinforce = &reinforceBatchQLearning;
        for (unsigned i = 0; i < group_size; i++) {
            auto a = new BatchQLearningAgent(max_rounds, actions, endowment, alpha, temperature);
            group.push_back(a);
        }
    }

    // Calculate execution time
    high_resolution_clock::time_point t1 = high_resolution_clock::now();

    //testing one group
    double pool;
    unsigned final_round;

    for (unsigned int step = 0; step < attempts; step++) {
        unsigned success = 0;
        double avgpayoff = 0.;
        double avg_rounds = 0.;
        for (unsigned int game = 0; game < games; game++) {
            // First we play the game
            std::tie(pool, final_round) = Game.playGame(group, donations, min_rounds, tu);
            avgpayoff += (Game.playersPayoff(group) / double(group_size));
            reinforce(pool, success, EGTTools::probabilityDistribution(generator), cataclysm, threshold, Game, group);
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