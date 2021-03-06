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
#include <Dyrwin/RL/CRDDemocracy.h>
#include <Dyrwin/CommandLineParsing.h>

using namespace std::chrono;
using namespace EGTTools::RL;

template<typename A>
void reinforceRothErev(double &pool, unsigned &success, double rnd_value, double &cataclysm, double &threshold,
                       CRDDemocracy<A> &Game, std::vector<A *> &group) {
    if (pool >= threshold) {
        Game.reinforcePath(group);
        success++;
    } else if (rnd_value > cataclysm) Game.reinforcePath(group);
    else Game.setPayoffs(group, 0);
}

template<typename A>
void reinforceBatchQLearning(double &pool, unsigned &success, double rnd_value, double &cataclysm, double &threshold,
                             CRDDemocracy<A> &Game, std::vector<A *> &group) {

    if (pool >= threshold) success++;
    else if (rnd_value < cataclysm) Game.setPayoffs(group, 0);

    Game.reinforcePath(group);
}

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
    std::string filename;
    std::string agent_type;
    Options options;
    CRDDemocracy<Agent> Game;
    // Random generators
    std::mt19937_64 generator{EGTTools::Random::SeedGenerator::getInstance().getSeed()};

    // Setup options
    options.push_back(makeDefaultedOption<size_t>("groupSize,M", &group_size, "set the group size", 6));
    options.push_back(makeDefaultedOption<size_t>("nbActions,p", &actions, "set the number of actions", 3));
    options.push_back(makeDefaultedOption<size_t>("nbAttempts,E", &attempts, "set the number of epochs", 1000));
    options.push_back(makeDefaultedOption<size_t>("nbGames,G", &games, "set the number of games per epoch", 1000));
    options.push_back(makeDefaultedOption<size_t>("nbRounds,t", &rounds, "set the number of rounds", 10));
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

    unsigned endowment = 2 * rounds; //per player
    auto threshold = static_cast<double>(rounds * group_size);
    auto donations = std::vector<size_t>(actions);
    for (unsigned i = 0; i < actions; i++) donations[i] = i;
    void (*reinforce)(double &, unsigned &, double, double &, double &, CRDDemocracy<Agent> &,
                      std::vector<Agent *> &);

    // Initialize agents depending on command option
    std::vector<Agent *> group;
    if (agent_type == "rothErev") {
        reinforce = &reinforceRothErev;
        for (unsigned i = 0; i < group_size; i++) {
            auto a = new Agent(rounds, actions, rounds, endowment);
            group.push_back(a);
        }
    } else if (agent_type == "rothErevLambda") {
        reinforce = &reinforceBatchQLearning;
        for (unsigned i = 0; i < group_size; i++) {
            auto a = new RothErevAgent(rounds, actions, rounds, endowment, lambda, temperature);
            group.push_back(a);
        }
    } else if (agent_type == "QLearning") {
        reinforce = &reinforceBatchQLearning;
        for (unsigned i = 0; i < group_size; i++) {
            auto a = new QLearningAgent(rounds, actions, rounds, endowment, alpha, lambda, temperature);
            group.push_back(a);
        }
    } else if (agent_type == "HistericQLearning") {
        reinforce = &reinforceBatchQLearning;
        for (unsigned i = 0; i < group_size; i++) {
            auto a = new HistericQLearningAgent(rounds, actions, rounds, endowment, alpha, beta, temperature);
            group.push_back(a);
        }
    } else {
        reinforce = &reinforceBatchQLearning;
        for (unsigned i = 0; i < group_size; i++) {
            auto a = new BatchQLearningAgent(rounds, actions, rounds, endowment, alpha, temperature);
            group.push_back(a);
        }
    }

    // Calculate execution time
    high_resolution_clock::time_point t1 = high_resolution_clock::now();

    //testing one group
    for (unsigned int step = 0; step < attempts; step++) {
        unsigned success = 0;
        double avgpayoff = 0.;
        double avg_rounds = 0.;
        for (unsigned int game = 0; game < games; game++) {
            // First we play the game
            auto[pool, final_round] = Game.playGame(group, donations, rounds);
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