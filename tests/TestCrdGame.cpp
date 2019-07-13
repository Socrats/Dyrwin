//
// Created by Elias Fernandez on 2019-02-10.
//
#include <iostream>
#include <iomanip>
#include <string>
#include <map>
#include <random>
#include <Dyrwin/SED/games/CrdGame.hpp>


using namespace std;
using namespace EGTTools;

int main() {
    size_t nb_rounds = 10, group_size = 6, endowment = 40, threshold = 120;
    double risk = 0.9;
    EGTTools::SED::PayoffVector payoffs(5, 0);
    EGTTools::SED::StrategyCounts group_composition(5, 0);
    group_composition[1] = 1;
    group_composition[3] = 5;
    std::cout << "group_composition: (";
    for (auto &i : group_composition)
        std::cout << i << ", ";
    std::cout << ")" << std::endl;

    EGTTools::SED::CRD::CrdGame game(endowment, threshold, nb_rounds, group_size, risk);

    game.play(group_composition, payoffs);

    std::cout << "(";
    for (auto &i : payoffs)
        std::cout << i << ", ";
    std::cout << ")" << std::endl;
    assert(payoffs[3] == 14);

    return 0;
}
