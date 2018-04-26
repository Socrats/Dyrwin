#include <iostream>
#include <vector>
#include "../include/CRDSimulator.h"
//#include <boost/random.hpp>

#define CHOICES 1000
#define SPINS 10

int main() {
    // Initialize random generator
    auto seed = static_cast<unsigned int>(std::chrono::high_resolution_clock::now().time_since_epoch().count());
    boost::mt19937 mt{seed};

    std::cout << "Starting Simulation" << std::endl;

    clock_t tStart = clock();

    CRDSimulator simulator(100, mt);

    std::cout << "Starting evolution" << std::endl;

    simulator.evolve(10000);

    std::cout << "Finished Simulation" << std::endl;

    // init random numbers
//    boost::mt19937 mt;
//    boost::uniform_real<> uniform(1, 10);
//    boost::variate_generator<boost::mt19937 &, boost::uniform_real<> > rng(mt, uniform);
//
//    // build_probabilities
//    std::vector<double> p(CHOICES, 0);
//    for (int i = 0; i < CHOICES; i++)
//        p[i] = rng();
//
//    // perform selections
//    int k;
//
//    // the boost way
//    boost::random::discrete_distribution<> dist(p);
//    for (int i = 0; i < SPINS; i++) {
//        k = dist(mt);
//        std::cout << "index: " << k << " value: " << p[k] << std::endl;
//    }

//    return k;

    printf("\nTime taken: %.8fs\n", (double) (clock() - tStart) / CLOCKS_PER_SEC);
    return 0;
}