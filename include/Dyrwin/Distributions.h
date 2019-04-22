//
// Adapted from https://github.com/Svalorzen/AI-Toolbox/
//

#ifndef DYRWIN_DISTRIBUTIONS_H
#define DYRWIN_DISTRIBUTIONS_H

#include <random>
#include <algorithm>
#include <Eigen/Dense>
#include <Dyrwin/Types.h>

namespace EGTTools {
    static std::uniform_real_distribution<double> probabilityDistribution(0.0, 1.0);

    /**
     * @brief This function samples an index from a probability vector.
     *
     * @tparam T Type of vector container
     * @tparam G Type of random number generator
     * @param d vector size
     * @param in probability vector
     * @param generator random number generator
     * @return An index in range [0, d-1].
     */
    template<typename T, typename G>
    size_t choice(const size_t d, const T &in, G &generator) {
        double p = probabilityDistribution(generator);
        double container;

        for ( size_t i = 0; i < d; ++i) {
            container = in[i];
            if (container > p) return i;
            p -= container;
        }
        return d - 1;
    }

    /**
     * @brief This function samples and index from a sparse probability vector.
     *
     * This function randomly samples an index between 0 and d, given a vector
     * containing the probabilities of sampling each of the indexes.
     *
     * @tparam G
     * @param d
     * @param in
     * @param generator
     * @return
     */
    template<typename G>
    size_t choice(const size_t d, const SparseMatrix2D::ConstRowXpr &in, G &generator) {
        double p = probabilityDistribution(generator);

        for (SparseMatrix2D::ConstRowXpr::InnerIterator i(in, 0);; ++i) {
            if (i.value() > p) return i.col();
            p -= i.value();
        }
        return d - 1;
    }

};

#endif //DYRWIN_DISTRIBUTIONS_H
