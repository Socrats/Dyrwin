//
// Created by Elias Fernandez on 17/05/2018.
//

#include <Dyrwin/SeedGenerator.h>

using namespace EGTTools::Random;

SeedGenerator::SeedGenerator() {
    _initSeed();
}

SeedGenerator &SeedGenerator::getInstance() {
    static SeedGenerator _instance;
    return _instance;
}

unsigned long int SeedGenerator::getSeed() {
    std::uniform_int_distribution<unsigned long int> distribution(0, std::numeric_limits<unsigned>::max());
    return distribution(rng_engine);
}

void SeedGenerator::_initSeed() {
    std::ifstream filein;
    std::ofstream fileout;
    fileout.open("seed.out");
    filein.open("seed.in");

    if (!filein) {

        std::random_device sysrand;
        _rng_seed = sysrand();

        rng_engine.seed(_rng_seed); //seed RNGengine

        fileout << _rng_seed << std::endl;
        fileout.close();
    } else {
        filein >> _rng_seed;
        fileout << _rng_seed << std::endl;

        rng_engine.seed(_rng_seed); //seed rndEngine
        fileout.close();
        filein.close();
    }
}
