//
// Created by Elias Fernandez on 17/05/2018.
//

#include "../include/Dyrwin/SeedGenerator.h"

SeedGenerator SeedGenerator::_instance;

SeedGenerator::SeedGenerator() {
    _initSeed();
}

SeedGenerator *SeedGenerator::getInstance() {
    return &_instance;
}

unsigned long int SeedGenerator::getSeed() {
    std::uniform_int_distribution<unsigned long int> distribution(0, std::numeric_limits<unsigned>::max());
    return distribution(_instance.rng_engine);
}

void SeedGenerator::setMainSeed(unsigned long int seed) {
    _instance._rng_seed = seed;
}

void SeedGenerator::_initSeed() {
    std::ifstream filein;
    std::ofstream fileout;
    fileout.open("seed.out");
    filein.open("seed.in");

    if (!filein) {

        std::random_device sysrand;
        _instance._rng_seed = sysrand();

        _instance.rng_engine.seed(_instance._rng_seed); //seed RNGengine

        fileout << _instance._rng_seed << std::endl;
        fileout.close();
        std::cout << "#seed.in not found. Creating a new seed: " << _instance._rng_seed << std::endl;
    } else {
        std::cout << "#reading seed.in" << std::endl;
        filein >> _instance._rng_seed;
        fileout << _instance._rng_seed << std::endl;

        //srandom(seed);
        _instance.rng_engine.seed(_instance._rng_seed); //seed RNGengine
        fileout.close();
        filein.close();
    }
}
