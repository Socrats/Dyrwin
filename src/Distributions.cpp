//
// Created by Elias Fernandez on 2019-06-10.
//

#include <Dyrwin/Distributions.h>

/**
     * @brief Calculates the binomial coefficient C(n, k)
     *
     * @param n size of the fixed set
     * @param k size of the unordered subset
     * @return C(n, k)
     */
size_t EGTTools::binomialCoeff(size_t n, size_t k) {
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
double EGTTools::binomialCoeff(double n, double k) {
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
EGTTools::multivariateHypergeometricPDF(size_t m, size_t k, size_t n, const std::vector<size_t> &sample_counts,
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