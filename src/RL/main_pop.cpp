/**
 * Created by Elias F. Domingos
 * Please refer to the LICENCE agreement if you want to use this code.
 *
 * The objective of this main function is to add an executable that will
 * handle a population of agents, for which at each iteration a group of
 * group_size will be sampled and will play a CRD. At each epoch the accumulated
 * payoffs are used to update the estimation of the Propensity matrices and
 * the policies.
 */

#include <iostream>
#include <vector>
#include <chrono>
#include <Dyrwin/RL/BatchQLearningAgent.h>
#include <Dyrwin/RL/QLearningAgent.h>
#include <Dyrwin/RL/RothErevAgent.h>
#include <Dyrwin/RL/HistericQLearningAgent.hpp>
#include <Dyrwin/RL/CRDGame.h>
#include <Dyrwin/CommandLineParsing.h>

using namespace std::chrono;
using namespace EGTTools::RL;

// Declaring global functions

template<typename A>
void reinforceRothErev(double &pool, unsigned &success, double rnd_value, double &cataclysm, double &threshold,
                       CRDGame<A> &Game, std::vector<A *> &group);

template<typename A>
void reinforceBatchQLearning(double &pool, unsigned &success, double rnd_value, double &cataclysm, double &threshold,
                             CRDGame<A> &Game, std::vector<A *> &group);

template<typename A>
void selectGroup(std::vector<A *>& pop, std::vector<A *>& group, size_t &group_size);


template<typename A>
void reinforceRothErev(double &pool, unsigned &success, double rnd_value, double &cataclysm, double &threshold,
                       CRDGame<A> &Game, std::vector<A *> &group) {
    if (pool >= threshold) {
        Game.reinforcePath(group);
        success++;
    } else if (rnd_value > cataclysm) Game.reinforcePath(group);
    else Game.setPayoffs(group, 0);
}

template<typename A>
void reinforceBatchQLearning(double &pool, unsigned &success, double rnd_value, double &cataclysm, double &threshold,
                             CRDGame<A> &Game, std::vector<A *> &group) {

    if (pool >= threshold) success++;
    else if (rnd_value < cataclysm) Game.setPayoffs(group, 0);

    Game.reinforcePath(group);
}

/**
 * @brief Samples a group of group size
 * @tparam A : type of agent parent class that defines the population
 * @param pop : vector
 * @param group : vector of references to the agents conforming the group
 * @param group_size : size of the group to sample
 */
template<typename A>
void selectGroup(std::vector<A *>& pop, std::vector<A *>& group, size_t &group_size) {

}

int main(int argc, char *argv[]) {
    //parameters
    size_t pop_size;
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
    CRDGame<Agent> Game;
    // Random generators
    std::mt19937_64 generator{EGTTools::Random::SeedGenerator::getInstance().getSeed()};

    // Setup options
    options.push_back(makeDefaultedOption<size_t>("popSize,Z", &pop_size, "set the population size", 6));
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
    void (*reinforce)(double &, unsigned &, double, double &, double &, CRDGame<Agent> &,
                      std::vector<Agent *> &);

    // Initialize agents depending on command option
    std::vector<Agent *> population(pop_size);
    std::vector<Agent *> group(group_size);
    if (agent_type == "rothErev") {
        reinforce = &reinforceRothErev;
        for (unsigned i = 0; i < pop_size; i++) {
            auto a = new Agent(rounds, actions, endowment);
            population.push_back(a);
        }
    } else if (agent_type == "rothErevLambda") {
        reinforce = &reinforceBatchQLearning;
        for (unsigned i = 0; i < pop_size; i++) {
            auto a = new RothErevAgent(rounds, actions, endowment, lambda, temperature);
            population.push_back(a);
        }
    } else if (agent_type == "QLearning") {
        reinforce = &reinforceBatchQLearning;
        for (unsigned i = 0; i < pop_size; i++) {
            auto a = new QLearningAgent(rounds, actions, endowment, alpha, lambda, temperature);
            population.push_back(a);
        }
    } else if (agent_type == "HistericQLearning") {
        reinforce = &reinforceBatchQLearning;
        for (unsigned i = 0; i < pop_size; i++) {
            auto a = new HistericQLearningAgent(rounds, actions, endowment, alpha, beta, temperature);
            population.push_back(a);
        }
    } else {
        reinforce = &reinforceBatchQLearning;
        for (unsigned i = 0; i < pop_size; i++) {
            auto a = new BatchQLearningAgent(rounds, actions, endowment, alpha, temperature);
            population.push_back(a);
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
            // First we sample the group

            // Then we play the game
            std::tie(pool, final_round) = Game.playGame(group, donations, rounds);
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