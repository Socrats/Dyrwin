//
// Created by Elias Fernandez on 2019-06-12.
//

#include <Dyrwin/SED/Utils.hpp>

double EGTTools::SED::fermi(double beta, double a, double b) {
    return 1 / (1 + std::exp(beta * (a - b)));
}

double EGTTools::SED::contest_success(double z, double a, double b) {
    double tmp = 1 / z;
    double tmp1 = std::pow(a, tmp);
    return tmp1 / (tmp1 + std::pow(b, tmp));
}

double EGTTools::SED::contest_success(double a, double b) {
    if (a > b) return 1.0;
    else return 0.0;
}

size_t EGTTools::SED::calculate_state(const size_t &group_size, const EGTTools::Factors &current_group) {
    size_t retval = 0;
    auto remaining = group_size;

    // In order to find the index for the input combination, we are basically
    // counting the number of combinations we have 'behind us", and we're going
    // to be the next. So for example if we have 10 combinations behind us,
    // we're going to be number 11.
    //
    // We do this recursively, element by element. For each element we count
    // the number of combinations we left behind. If data[i] is the highest
    // possible (i.e. it accounts for all remaining points), then it is the
    // first and we didn't miss anything.
    //
    // Otherwise we count how many combinations we'd have had with the max (1),
    // add it to the number, and decrease the h. Then we try again: are we
    // still lower? If so, count again how many combinations we'd have had with
    // this number (size() - 1). And so on, until we match the number we have.
    //
    // Then we go to the next element, considering the subarray of one element
    // less (thus the size() - i), and we keep going.
    //
    // Note that by using this algorithm the last element in the array is never
    // needed (since it is determined by the others), and additionally when we
    // have no remaining elements to parse we can just break.
    for (size_t i = 0; i < current_group.size() - 1; ++i) {
        auto h = remaining;
        while (h > current_group[i]) {
            retval += EGTTools::starsBars(remaining - h, current_group.size() - i - 1);
            --h;
        }
        if (remaining == current_group[i])
            break;
        remaining -= current_group[i];
    }

    return retval;
}

void EGTTools::SED::sample_simplex(size_t i, const size_t &pop_size, const size_t &nb_strategies,
                                   EGTTools::VectorXui &state) {
    auto remaining = pop_size;

    for (size_t a = 0; a < nb_strategies; ++a) {
        for (size_t j = remaining; j > 0; --j) {
            auto count = EGTTools::starsBars(remaining - j, nb_strategies - a - 1);
            if (i >= count) {
                i -= count;
            } else {
                state(a) = j;
                remaining -= j;
                break;
            }
        }
    }
}

void EGTTools::SED::sample_simplex(size_t i, const size_t &pop_size, const size_t &nb_strategies,
                                   std::vector<size_t> &state) {
    // To be able to infer the multi-dimensional state from the index
    // we apply a recursive algorithm that will complete a vector of size
    // nb_strategies from right to left

    auto remaining = pop_size;

    for (size_t a = 0; a < nb_strategies; ++a) {
        // reset the state container
        state[a] = 0;
        for (size_t j = remaining; j > 0; --j) {
            auto count = EGTTools::starsBars(remaining - j, nb_strategies - a - 1);
            if (i >= count) {
                i -= count;
            } else {
                state[a] = j;
                remaining -= j;
                break;
            }
        }
    }
}
