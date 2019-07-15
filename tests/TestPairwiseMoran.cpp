//
// Created by Elias Fernandez on 2019-02-10.
//
#include <iostream>
#include <iomanip>
#include <string>
#include <map>
#include <random>
#include <Dyrwin/Types.h>
#include <Dyrwin/SED/behaviors/CrdBehaviors.hpp>
#include <Dyrwin/SED/games/CrdGame.hpp>
#include <Dyrwin/SED/PairwiseMoran.hpp>


using namespace std;
using namespace EGTTools;

int main() {

    // First we define a vector of possible behaviors
    size_t nb_strategies = EGTTools::SED::CRD::nb_strategies;
    size_t pop_size = 100;
    size_t group_size = 6;
    size_t nb_rounds = 10, endowment = 40, threshold = 120;
    double risk = 0.9;
    EGTTools::VectorXui init_state(nb_strategies);
    init_state << 25, 25, 25, 25, 0;

    EGTTools::SED::CRD::CrdGame game(endowment, threshold, nb_rounds, group_size, risk);

    // Initialise selection mutation process
    auto smProcess = EGTTools::SED::PairwiseMoran(pop_size, game, 10000);

    auto dist = smProcess.run(1000, 0.05, 0.0001, init_state);

    std::cout << dist << std::endl;

    return 0;
}
