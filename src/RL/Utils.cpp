//
// Created by Elias Fernandez on 2019-05-14.
//

#include <Dyrwin/RL/Utils.h>

size_t EGTTools::RL::factorSpace(const Factors &space) {
    size_t retval = 1;
    for (const auto f : space) {
        // Detect wraparound
        if (std::numeric_limits<size_t>::max() / f < retval)
            return std::numeric_limits<size_t>::max();
        retval *= f;
    }
    return retval;
}

void EGTTools::RL::toFactors(const Factors &space, size_t id, Factors *out) {
    assert(out);

    auto &f = *out;

    for (size_t i = 0; i < space.size(); ++i) {
        f[i] = id % space[i];
        id /= space[i];
    }
}

EGTTools::RL::Factors EGTTools::RL::toFactors(const Factors &space, size_t id) {
    Factors f(space.size());
    EGTTools::RL::toFactors(space, id, &f);
    return f;
}

size_t EGTTools::RL::toIndex(const Factors &space, const Factors &f) {
    size_t result = 0;
    size_t multiplier = 1;
    for (size_t i = 0; i < f.size(); ++i) {
        result += multiplier * f[i];
        multiplier *= space[i];
    }
    return result;
}


EGTTools::RL::FlattenState::FlattenState(const EGTTools::RL::Factors &space) : space(space) {
    assert(space.size() > 1);
    factor_space = EGTTools::RL::factorSpace(space);
}

EGTTools::RL::Factors EGTTools::RL::FlattenState::toFactors(size_t id) {
    return EGTTools::RL::toFactors(space, id);
}

void EGTTools::RL::FlattenState::toFactors(size_t id, Factors *out) {
    EGTTools::RL::toFactors(space, id, out);
}

size_t EGTTools::RL::FlattenState::toIndex(const Factors &f) {
    return EGTTools::RL::toIndex(space, f);
}