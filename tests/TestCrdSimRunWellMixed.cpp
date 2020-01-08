//
// Created by Elias Fernandez on 2019-02-10.
//
#include <iostream>
#include <map>
#include <random>
#include <Dyrwin/RL/CrdSim.hpp>

using namespace std;
using namespace EGTTools;

int main() {
    //parameters
    size_t group_size = 6;
    size_t actions = 3; //0, 2 and 4
    size_t rounds = 10;
    size_t attempts = 1;
    size_t games = 1000;
    double cataclysm = 0.9;
    double alpha = 0.09;
    double temperature = 5;
    std::string agent_type = "BatchQLearning";

    auto endowment = static_cast<double>(2 * rounds);
    auto threshold = static_cast<double>(rounds * group_size);
    EGTTools::RL::ActionSpace available_actions = EGTTools::RL::ActionSpace(actions);
    for (size_t i = 0; i < actions; ++i) available_actions[i] = i;

    std::vector<double> args{alpha, temperature};

    try {
        EGTTools::RL::CRDSim sim(attempts, games, rounds, actions, group_size,
                                 cataclysm, endowment, threshold,
                                 available_actions, agent_type,
                                 args);
        EGTTools::RL::DataTypes::CRDData results = sim.runWellMixedSync(100, group_size, 100, 10, threshold, cataclysm,
                                                                    agent_type, args);
        std::cout << "success: " << results.eta << std::endl;
        std::cout << "avg_contrib: " << results.avg_contribution << std::endl;
        sim.Game.printGroup(results.population);
    } catch (std::invalid_argument &e) {
        std::cerr << "\033[1;31m[EXCEPTION] Invalid argument: " << e.what() << "\033[0m" << std::endl;
        return -1;
    }


    return 0;
}
