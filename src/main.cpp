#include <iostream>
#include "../include/CRDSimulator.h"

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

    printf("\nTime taken: %.8fs\n", (double) (clock() - tStart) / CLOCKS_PER_SEC);
    return 0;
}