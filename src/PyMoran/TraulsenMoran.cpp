//
// Created by Elias Fernandez on 2019-04-02.
//

#include "../../include/Dyrwin/PyMoran/TraulsenMoran.h"

egt_tools::TraulsenMoran::TraulsenMoran(uint64_t generations, unsigned int group_size, unsigned int nb_groups,
                                        double beta, double mu, double coop_freq, double split_prob,
                                        Eigen::Ref<const MatrixXd> payoff_matrix) : _generations(generations),
                                                                                    _group_size(group_size),
                                                                                    _nb_groups(nb_groups),
                                                                                    _beta(beta),
                                                                                    _mu(mu),
                                                                                    _coop_freq(coop_freq),
                                                                                    _split_prob(split_prob),
                                                                                    _payoff_matrix(
                                                                                            payoff_matrix) {

    _pop_size = _nb_groups * _group_size;
    _nb_coop = (unsigned int) floor(_coop_freq * _pop_size);
    _final_coop_freq = _coop_freq;
}

void egt_tools::TraulsenMoran::initialize_population(std::vector<unsigned int> &population) {
    unsigned int i;

    for (i = _nb_coop; i < _pop_size; i++) {
        population[i] = 1;
    }

    // Then we shuffle it randomly
    std::shuffle(population.begin(), population.end(), _mt);
}

void egt_tools::TraulsenMoran::initialize_group_coop(std::vector<unsigned int> &group_coop,
                                                     std::vector<unsigned int> &population) {
    unsigned int i;
    // Calculate the number of cooperators in each group
    for (i = 0; i < _pop_size; i++) {
        group_coop[floor(i / _group_size)] += population[i];
    }
}

double egt_tools::TraulsenMoran::fermifunc(double beta, double a, double b) {
    return 1 / (1 + exp(beta * (a - b)));
}

double egt_tools::TraulsenMoran::evolve(double beta) {
    uint64_t i;
    unsigned int p1, p2;
    double freq1, freq2, fitness1, fitness2;
    double ref = _nb_coop;
    int gradient = 0;
    std::uniform_int_distribution<unsigned int> dist(0, _pop_size - 1);
    std::uniform_real_distribution<double> _uniform_real_dist(0.0, 1.0);
    std::vector<unsigned int> population = std::vector<unsigned int>(_pop_size, 0.0);
    std::vector<unsigned int> group_coop = std::vector<unsigned int>(_nb_groups, 0.0);

    initialize_population(population);
    initialize_group_coop(group_coop, population);

    // Now run a Moran Process loop
    if (_mu == 0.) { // Without mutation
        for (i = 0; i < _generations; i++) {
            _moran_step(p1, p2, gradient, ref, freq1, freq2, fitness1, fitness2, beta, group_coop, population, dist,
                        _uniform_real_dist);
            if ((ref == _pop_size) || (ref == 0)) break;
        }
    } else { // With mutation
        for (i = 0; i < _generations; i++) {
            _moran_step_mutation(p1, p2, gradient, ref, freq1, freq2, fitness1, fitness2, beta, group_coop,
                                 population, dist,
                                 _uniform_real_dist);
            if ((ref == _pop_size) || (ref == 0)) break;
        }
    }
    _final_coop_freq = ref / (double) _pop_size;
    return _final_coop_freq;
}

double egt_tools::TraulsenMoran::evolve(unsigned int runs, double beta) {
    float coop_freq = 0;
        // Run loop
#pragma omp parallel for shared(coop_freq)
        for (unsigned int j = 0; j < runs; j++) {
            coop_freq += evolve(beta);
        }
    _final_coop_freq = coop_freq / runs;
    return _final_coop_freq;
}

std::vector<double> egt_tools::TraulsenMoran::evolve(std::vector<double> betas) {
    std::vector<double> coop_freqs(betas.size());

    for (unsigned int j = 0; j < betas.size(); j++) {
        coop_freqs[j] = evolve(betas[j]);
    }
    _final_coop_freq = coop_freqs.back();
    return coop_freqs;
}

std::vector<double> egt_tools::TraulsenMoran::evolve(std::vector<double> betas, unsigned int runs) {
    std::vector<double> coop_freqs(betas.size());

    for (unsigned int w = 0; w < betas.size(); w++) {
        coop_freqs[w] = evolve(runs, betas[w]);
    }
    _final_coop_freq = coop_freqs.back();
    return coop_freqs;
}

/**
 * @brief Multi-level selection with exponential mapping of fitness described in Traulsen & Nowak, 2006.
 *
 * In each time step, a random individual from the entire population is chosen for reproduction
 * proportional to fitness. The offspring is added to the same group, If the new groups size is
 * less than or equal to n, nothing else happens. If the group size exceeds n, then with probability
 * q, the group splits into two. In this case, a random group is eliminated in order to maintain a
 * constant number of groups. With probability 1-q, however, teh groups does not divide, but instead
 * a random individual from that group is eliminated.
 *
 * @param p1 : index of player 1 in the population
 * @param p2 : index of player 2 in the population
 * @param gradient : gradient of selection (whether the number of cooperator, increases, decreases or is maintained)
 * @param ref : current number of cooperators in the population
 * @param freq1 : frequency of cooperators for player 1
 * @param freq2 : frequency of cooperators for player 2
 * @param fitness1 : fitness of player 1
 * @param fitness2 : fitness of player 2
 * @param beta : intensity of selection
 * @param group_coop : reference to vector where the group compositions are stored
 * @param population : reference to the vector where the population is stored
 * @param dist : random integer generator
 * @param _uniform_real_dist random real number generator
 */
void egt_tools::TraulsenMoran::_moran_step(unsigned int &p1, unsigned int &p2, int &gradient, double &ref,
                                           double &freq1, double &freq2,
                                           double &fitness1, double &fitness2,
                                           double &beta,
                                           std::vector<unsigned int> &group_coop,
                                           std::vector<unsigned int> &population,
                                           std::uniform_int_distribution<unsigned int> &dist,
                                           std::uniform_real_distribution<double> &_uniform_real_dist) {
    unsigned int g1, g2;
    // Randomly select 2 individuals from the population
    p1 = dist(_mt);
    p2 = dist(_mt);
    while (p2 == p1) p2 = dist(_mt);
    if (population[p2] == population[p1]) return;
    // Group index
    g1 = p1 / _group_size;
    g2 = p2 / _group_size;

    // Calculate frequencies
    freq1 = (group_coop[g1] - population[p1]) / (float) (_pop_size - 1);
    freq2 = (group_coop[g2] - population[p2]) / (float) (_pop_size - 1);

    // Calculate payoffs
    fitness1 = (_payoff_matrix(population[p1], 1) * freq1) + (_payoff_matrix(population[p1], 0) * (1 - freq1));
    fitness2 = (_payoff_matrix(population[p2], 1) * freq2) + (_payoff_matrix(population[p2], 0) * (1 - freq2));

    // Select according to fermi function
    if (_uniform_real_dist(_mt) < fermifunc(beta, fitness1, fitness2)) {

        gradient = population[p2] - population[p1];
        population[p1] = population[p2];
        group_coop[g1] += gradient;
    } else {
        gradient = population[p1] - population[p2];
        population[p2] = population[p1];
        group_coop[g2] += gradient;
    }
    ref += gradient;
}

void egt_tools::TraulsenMoran::_moran_step_mutation(unsigned int &p1, unsigned int &p2, int &gradient, double &ref,
                                                    double &freq1, double &freq2,
                                                    double &fitness1, double &fitness2,
                                                    double &beta,
                                                    std::vector<unsigned int> &group_coop,
                                                    std::vector<unsigned int> &population,
                                                    std::uniform_int_distribution<unsigned int> &dist,
                                                    std::uniform_real_distribution<double> &_uniform_real_dist) {
    unsigned int g1, g2;
    // Randomly select 2 individuals from the population
    p1 = dist(_mt);
    p2 = dist(_mt);
    while (p2 == p1) p2 = dist(_mt);
    if (population[p2] == population[p1]) return;
    // Group index
    g1 = p1 / _group_size;
    g2 = p2 / _group_size;

    // Calculate frequencies
    freq1 = (group_coop[g1] - population[p1]) / (float) (_pop_size - 1);
    freq2 = (group_coop[g2] - population[p2]) / (float) (_pop_size - 1);

    // Calculate payoffs
    fitness1 = (_payoff_matrix(population[p1], 1) * freq1) + (_payoff_matrix(population[p1], 0) * (1 - freq1));
    fitness2 = (_payoff_matrix(population[p2], 1) * freq2) + (_payoff_matrix(population[p2], 0) * (1 - freq2));

    // Select according to fermi function
    if (_uniform_real_dist(_mt) < fermifunc(beta, fitness1, fitness2)) {
        if (_uniform_real_dist(_mt) < _mu) {
            if (_uniform_real_dist(_mt) < 0.5) {
                gradient = -population[p1];
                population[p1] = 0;
                group_coop[g1] += gradient;
            } else {
                gradient = 1 - population[p1];
                population[p1] = 1;
                group_coop[g1] += gradient;
            }
        } else {
            gradient = population[p2] - population[p1];
            population[p1] = population[p2];
            group_coop[g1] += gradient;
        }
    } else {
        if (_uniform_real_dist(_mt) < _mu) {
            if (_uniform_real_dist(_mt) < 0.5) {
                gradient = -population[p2];
                population[p2] = 0;
                group_coop[g2] += gradient;
            } else {
                gradient = 1 - population[p2];
                population[p2] = 1;
                group_coop[g2] += gradient;
            }
        } else {
            gradient = population[p1] - population[p2];
            population[p2] = population[p1];
            group_coop[g2] += gradient;
        }
    }
    ref += gradient;
}