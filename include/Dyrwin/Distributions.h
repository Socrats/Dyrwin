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
        std::uniform_real_distribution<double> probabilityDistribution(0.0, 1.0);
        double p = probabilityDistribution(generator);
        double container;

        for (size_t i = 0; i < d; ++i) {
            container = in[i];
            if (container > p) return i;
            p -= container;
        }
        return d - 1;
    }

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
    size_t choice(const size_t d, const T &in, std::uniform_real_distribution<double> &dist, G &generator) {
        double p = dist(generator);
        double container;

        for (size_t i = 0; i < d; ++i) {
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
        std::uniform_real_distribution<double> probabilityDistribution(0.0, 1.0);
        double p = probabilityDistribution(generator);

        for (SparseMatrix2D::ConstRowXpr::InnerIterator i(in, 0);; ++i) {
            if (i.value() > p) return i.col();
            p -= i.value();
        }
        return d - 1;
    }

    /**
     * @brief Calculates the binomial coefficient C(n, k)
     *
     * @param n size of the fixed set
     * @param k size of the unordered subset
     * @return C(n, k)
     */
    size_t binomialCoeff(size_t n, size_t k) {
        size_t res = 1;

        // Since C(n, k) = C(n, n-k)
        if (k > n - k) k = n - k;

        // Calculate value of [n * (n-1) * ... * (n-k+1)] / [k * (k-1) * ... * 1]
        for (size_t i = 0; i < k; ++i) {
            res *= (n - i);
            res /= (i + 1);
        }

        return res;
    }

    /**
     * @brief Calculates the binomial coefficient C(n, k)
     *
     * @param n size of the fixed set
     * @param k size of the unordered subset
     * @return C(n, k)
     */
    double binomialCoeff(double n, double k) {
        double res = 1.0;

        // Since C(n, k) = C(n, n-k)
        if (k > n - k) k = n - k;

        // Calculate value of [n * (n-1) * ... * (n-k+1)] / [k * (k-1) * ... * 1]
        for (int64_t i = 0; i < k; ++i) {
            res *= n - i;
            res /= i + 1;
        }

        return res;
    }

    /**
     * @brief Calculates the probability density function of a multivariate hypergeometric distribution.
     *
     * This function returns the probability that a sample of size @param n in a population of @param k
     * objects with have @param sample_counts counts of each object in a sample, given a population D
     * with @param population_counts counts of each object.
     *
     * The sampling is without replacement.
     *
     * @param m size of the population
     * @param k number of objects in the population
     * @param n size of the sample
     * @param sample_counts a vector containing the counts of each objects in the sample
     * @param population_counts a vector containing the counts of each objects in the population
     * @return probability of a sample occurring in the population.
     */
    double
    multivariateHypergeometricPDF(size_t m, size_t k, size_t n, const std::vector<size_t> &sample_counts,
                                  const std::vector<size_t> &population_counts) {

        size_t res = 1;
        // First we calculate the number of unordered samples of size n chosen from the population
        auto denominator = EGTTools::binomialCoeff(m, n);

        // Then we calculate the multiplication of the number of all unordered subsets of a subset of the population
        // with only 1 type of object
        for (size_t i = 0; i < k; ++i) {
            res *= EGTTools::binomialCoeff(population_counts[i], sample_counts[i]);
        }

        return static_cast<double>(res) / denominator;
    }

}

#endif //DYRWIN_DISTRIBUTIONS_H
