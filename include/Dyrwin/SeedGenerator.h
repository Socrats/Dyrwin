//
// Created by Elias Fernandez on 17/05/2018.
//

#ifndef DYRWIN_RANDGEN_H
#define DYRWIN_RANDGEN_H

#include <random>
#include <iostream>
#include <fstream>

class SeedGenerator {
public:

    /**
     * @brief This functions provices a pointer to a Seeder class
     * @return SeedGenerator
     */
    static SeedGenerator& getInstance();

    /**
     * @brief This function generates a random number to seed other generators
     * @return A random unsigned long number
     */
    unsigned long int getSeed();

    /**
     * @brief This function sets the seed for the seed generator
     *
     * By default the generator is seeded either from a seed.in file or from random_device
     *
     * @param seed The seed for the random generator used to generate new seeds
     */
    void setMainSeed(unsigned long int seed);

private:
    // Random generator
    std::mt19937_64 rng_engine;

    // seed
    unsigned long int _rng_seed = 0;

    // Private constructor to prevent instancing
    SeedGenerator();

    /**
     * @brief This function initializes the main seed either from a file or from random_device
     *
     * It also store the seed used to initialize the seed generator into a seed.out file
     */
    void _initSeed();
};


#endif //DYRWIN_RANDGEN_H
