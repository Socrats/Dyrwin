//
// Created by Elias Fernandez on 2019-06-10.
//

#include <Dyrwin/Distributions.h>

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

size_t EGTTools::starsBars(size_t stars, size_t bins) {
    return EGTTools::binomialCoeff(stars + bins - 1, stars);
}