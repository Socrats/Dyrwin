//
// Created by Elias Fernandez on 2019-02-10.
//
#include <iostream>
#include "../../include/Dyrwin/PyMoran/PDImitation.h"

float fermifunc(float beta, float a, float b);

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

    printf("\nExecution time: %.8fs\n", (double) (clock() - tStart) / CLOCKS_PER_SEC);
    return 0;
}


float fermifunc(float beta, float a, float b) {
    return 1 / (1 + exp(beta * (a - b)));
}
