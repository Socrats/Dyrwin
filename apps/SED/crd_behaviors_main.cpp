//
// Created by Elias Fernandez on 2019-02-10.
//
#include <iostream>

#include <Dyrwin/SED/behaviors/CrdBehaviors.hpp>
#include <Dyrwin/Types.h>
#include <Dyrwin/CommandLineParsing.h>

using namespace std;
using namespace EGTTools;

int main(int argc, char *argv[]) {

    // First initialize population and global parameters
    unsigned int pop_size = 100, nb_betas = 7;
    size_t group_size, nb_groups, runs_grad;
    double split_prob;
    float mu = 1e-3;
//    float coop_freq = 0.5;
    EGTTools::Matrix2D payoff_matrix;
    payoff_matrix << 1, 4,
            0, 3;
    std::vector<float> betas(nb_betas);
    EGTTools::Vector strategy_freq(2);
    strategy_freq << 0.01, 0.99;
    float beta = 1e-4;
    unsigned int generations = 1000;
    unsigned int runs = 100;
    bool test;
    Options options;

    options.push_back(makeDefaultedOption("generations,g", &generations, "set the number of generations", 1000u));
    options.push_back(makeDefaultedOption("popSize,Z", &pop_size, "set the size of the population", 100u));
    options.push_back(makeDefaultedOption("mu,u", &mu, "set mutation rate", 1e-3f));
    options.push_back(makeDefaultedOption("beta,b", &beta, "set intensity of selection", 1e-4f));
    options.push_back(makeDefaultedOption("runs,R", &runs, "set the number of runs", 100u));
    options.push_back(makeDefaultedOption<size_t>("runsGrad,i", &runs_grad, "set the number of runs", 100));
    options.push_back(makeDefaultedOption("test,t", &test, "test StochDynamics", false));
    options.push_back(makeDefaultedOption("splitProb,q", &split_prob, "split probability", 0.0));
    options.push_back(makeDefaultedOption<size_t>("groupSize,n", &group_size, "split probability", 50));
    options.push_back(makeDefaultedOption<size_t>("nbGroups,m", &nb_groups, "split probability", 1));

    if (!parseCommandLine(argc, argv, options))
        return 1;

    clock_t tStart = clock();



    printf("\nExecution time: %.8fs\n", (double) (clock() - tStart) / CLOCKS_PER_SEC);
    return 0;
}
