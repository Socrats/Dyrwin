//
// Created by Elias Fernandez on 2019-02-10.
//
#include <iostream>
#include "../../include/Dyrwin/PyMoran/PDImitation.h"
#include "../../include/Dyrwin/PyMoran/StochDynamics.h"
#include "../../include/Dyrwin/CommandLineParsing.h"

using namespace std;
using namespace egt_tools;

int main(int argc, char *argv[]) {

    // First initialize population and global parameters
    unsigned int pop_size = 50, nb_betas = 7;
    float mu = 1e-3;
    float coop_freq = 0.5;
    std::vector<float> payoff_matrix({1, 4, 0, 3});
    std::vector<float> betas(nb_betas);
    float beta = 1e-4;
    unsigned int generations = 10000;
    unsigned int runs = 100;
    bool test;
    Options options;

    options.push_back(makeDefaultedOption("generations,g", &generations, "set the number of generations", 1000u));
    options.push_back(makeDefaultedOption("popSize,N", &pop_size, "set the size of the population", 50u));
    options.push_back(makeDefaultedOption("mu,m", &mu, "set mutation rate", 1e-3f));
    options.push_back(makeDefaultedOption("beta,b", &mu, "set intensity of selection", 1e-4f));
    options.push_back(makeDefaultedOption("runs,R", &runs, "set the number of runs",100u));
    options.push_back(makeDefaultedOption("test,t", &test, "test StochDynamics",false));

    if (!parseCommandLine(argc, argv, options))
        return 1;

    PDImitation pd(generations, pop_size, beta, mu, coop_freq, payoff_matrix);

    // Random generators
    std::mt19937_64 _mt{SeedGenerator::getInstance().getSeed()};
    // Uniform int distribution
    std::uniform_int_distribution<unsigned int> dist(0, pop_size - 1);
    std::uniform_real_distribution<double> _uniform_real_dist(0.0, 1.0);

    clock_t tStart = clock();
    betas[0] = beta;
    for (size_t i=1; i< nb_betas; i++) {
        betas[i] = betas[i - 1] * 10;
    }

    auto result = pd.evolve(betas, runs);

    for (size_t i=0; i< nb_betas; i++) {
        cout << "[beta " << betas[i] << "] freq_coop: " << result[i] << endl;
    }

    if (test) {
        Eigen::Matrix2d payoffs;
        payoffs << 1, 4,
                   0, 3;
        StochDynamics sdy(pop_size, 2, payoffs);
        cout << "Trans prob: " << std::get<1>(sdy.calculate_transition_fixations(1.))(0, 0) << endl;
    }

    printf("\nExecution time: %.8fs\n", (double) (clock() - tStart) / CLOCKS_PER_SEC);
    return 0;
}
