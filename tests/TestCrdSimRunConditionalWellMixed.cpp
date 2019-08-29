//
// Created by Elias Fernandez on 2019-02-10.
//
#include <iostream>
#include <iomanip>
#include <string>
#include <map>
#include <random>
#include <Dyrwin/RL/CrdSim.hpp>


using namespace std;
using namespace EGTTools;

int main() {
    //parameters
    size_t group_size = 6;
    size_t actions = 3; //0, 2 and 4
    size_t mean_rounds = 10;
    size_t min_rounds = 8;
    size_t max_rounds = 30;
    size_t attempts = 1;
    size_t games = 1000;
    double p = 1. / (mean_rounds - min_rounds + 1);
    double cataclysm = 0.9;
    double alpha = 0.03;
    double temperature = 5;
    std::string agent_type = "BatchQLearning";

    double endowment = 2 * mean_rounds;
    auto threshold = static_cast<double>(mean_rounds * group_size);
    EGTTools::RL::ActionSpace available_actions = EGTTools::RL::ActionSpace(actions);
    for (size_t i = 0; i < actions; ++i) available_actions[i] = i;

    std::vector<double> args{alpha, temperature};

    try {
        EGTTools::RL::CRDSim sim(attempts, games, mean_rounds, actions, group_size,
                                 cataclysm, endowment, threshold,
                                 available_actions, agent_type,
                                 args);
        EGTTools::RL::DataTypes::CRDData results = sim.runConditionalWellMixedTU(100, group_size, 1000, 200, threshold,
                                                                                 cataclysm,
                                                                                 min_rounds, mean_rounds, max_rounds, p,
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
