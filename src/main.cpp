#include <iostream>
#include <random>
#include "../include/Dyrwin/CRDSimulator.h"

int main() {
    std::cout << "Starting Simulation: " << std::endl;

    clock_t tStart = clock();

    CRDSimulator simulator(100);

    std::cout << "Starting evolution" << std::endl;

    simulator.evolve(10000);

    std::cout << "Finished Simulation" << std::endl;

    printf("\nTime taken: %.8fs\n", (double) (clock() - tStart) / CLOCKS_PER_SEC);
    return 0;
}