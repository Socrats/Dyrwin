//
// Created by Elias Fernandez on 17/05/2018.
//

#ifndef DYRWIN_RANDGEN_H
#define DYRWIN_RANDGEN_H

#include <random>
#include <iostream>
#include <fstream>

namespace EGTTools::Random {
    class SeedGenerator {
    public:

        /**
         * @brief This functions provices a pointer to a Seeder class
         * @return SeedGenerator
         */
        static SeedGenerator &getInstance();
        ~SeedGenerator() = default;

        /**
         * @brief This function generates a random number to seed other generators
         *
         * You can use this function to generate a random seed to seed other random generators
         * in you project. This avoids concurrency problems when doing parallel execution.
         *
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
        void setMainSeed(unsigned long int seed) { _rng_seed = seed; }

        /**
         * @brief This function sets the seed for the seed generator
         *
         * By default the generator is seeded either from a seed.in file or from random_device
         *
         * @return main seed (unsigned long int)
         */
        unsigned long int getMainSeed() { return _rng_seed; }

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
}

#endif //DYRWIN_RANDGEN_H
