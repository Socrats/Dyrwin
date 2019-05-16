//
// Created by Elias Fernandez on 2019-05-15.
//

#include <Dyrwin/SED/behaviors/CrdBehaviors.hpp>

size_t EGTTools::SED::cooperator(size_t prev_donation, size_t threshold) {
    UNUSED(prev_donation);
    UNUSED(threshold);
    return 2;
}

size_t EGTTools::SED::defector(size_t prev_donation, size_t threshold) {
    UNUSED(prev_donation);
    UNUSED(threshold);
    return 0;
}

size_t EGTTools::SED::altruist(size_t prev_donation, size_t threshold) {
    UNUSED(prev_donation);
    UNUSED(threshold);
    return 4;
}

size_t EGTTools::SED::reciprocal(size_t prev_donation, size_t threshold) {
    if (prev_donation >= threshold)
        return 4;
    else
        return 0;
}

size_t EGTTools::SED::compensator(size_t prev_donation, size_t threshold) {
    if (prev_donation <= threshold)
        return 0;
    else
        return 4;
}

EGTTools::SED::CrdBehavior::CrdBehavior() {
    // Random generators
    std::mt19937_64 mt{EGTTools::Random::SeedGenerator::getInstance().getSeed()};
    std::uniform_int_distribution<size_t> dist(0, nb_strategies - 1);

    type = dist(mt);
    payoff = 0;
    switch (type) {
        case 0: act = cooperator;
        case 1: act = defector;
        case 2: act = altruist;
        case 3: act = reciprocal;
        case 4: act = compensator;
    }
}

EGTTools::SED::CrdBehavior::CrdBehavior(size_t type) {
    this->type = type;
    payoff = 0;
    switch (type) {
        case 0: act = cooperator;
        case 1: act = defector;
        case 2: act = altruist;
        case 3: act = reciprocal;
        case 4: act = compensator;
        default: {
            act = cooperator;
            this->type = 0;
        }
    }
}
