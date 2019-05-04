/**
 * Created by Elias Fernandez on 2019-03-11.
 *
 * Some function are taken from https://github.com/Svalorzen/AI-Toolbox/blob/master/src/Factored/Utils/Core.cpp
 */

#ifndef DYRWIN_RL_UTILS_H
#define DYRWIN_RL_UTILS_H

#include <vector>

namespace EGTTools::RL {
    using Factors = std::vector<size_t>;

    size_t factorSpace(const Factors &space) {
        size_t retval = 1;
        for (const auto f : space) {
            // Detect wraparound
            if (std::numeric_limits<size_t>::max() / f < retval)
                return std::numeric_limits<size_t>::max();
            retval *= f;
        }
        return retval;
    }

    void toFactors(const Factors &space, size_t id, Factors *out) {
        assert(out);

        auto &f = *out;

        for (size_t i = 0; i < space.size(); ++i) {
            f[i] = id % space[i];
            id /= space[i];
        }
    }

    Factors toFactors(const Factors &space, size_t id) {
        Factors f(space.size());
        EGTTools::RL::toFactors(space, id, &f);
        return f;
    }

    size_t toIndex(const Factors &space, const Factors &f) {
        size_t result = 0;
        size_t multiplier = 1;
        for (size_t i = 0; i < f.size(); ++i) {
            result += multiplier * f[i];
            multiplier *= space[i];
        }
        return result;
    }

    struct FlattenState {
        explicit FlattenState(const Factors& space) : space(space) {
            assert(space.size() > 1);
            factor_space = EGTTools::RL::factorSpace(space);
        }

        Factors space;
        size_t factor_space;

        Factors toFactors(size_t id) { return EGTTools::RL::toFactors(space, id); }

        void toFactors(size_t id, Factors *out) { EGTTools::RL::toFactors(space, id, out); }

        size_t toIndex(const Factors &f) { return EGTTools::RL::toIndex(space, f); }
    };
}

#endif //DYRWIN_RL_UTILS_H
