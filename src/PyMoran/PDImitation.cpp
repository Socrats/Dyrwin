//
// Created by Elias Fernandez on 2019-02-11.
//

#include "../../include/Dyrwin/PyMoran/PDImitation.h"

egt_tools::PDImitation::PDImitation(unsigned int generations, unsigned int pop_size, float beta,
                                    float mu, float coop_freq,
                                    MatrixXd payoff_matrix) : _generations(generations),
                                                                        _pop_size(pop_size), _beta(beta),
                                                                        _mu(mu),
                                                                        _coop_freq(coop_freq),
                                                                        _payoff_matrix(std::move(payoff_matrix)) {

    std::uniform_real_distribution<double> _uniform_real_dist(0.0, 1.0);
    _nb_coop = (unsigned int) floor((double) (_coop_freq * _pop_size));
    _population = std::vector<unsigned int>(_pop_size);
    _final_coop_freq = _coop_freq;

    initialize_population(_population);

}

void egt_tools::PDImitation::initialize_population(std::vector<unsigned int> &population) {
    unsigned int i;

    for (i = 0; i < _nb_coop; i++) {
        population[i] = 0;
    }
    for (i = _nb_coop; i < _pop_size; i++) {
        population[i] = 1;
    }

    // Then we shuffle it randomly
    std::shuffle(population.begin(), population.end(), _mt);
}

float egt_tools::PDImitation::fermifunc(float beta, float a, float b) {
    return 1 / (1 + exp(beta * (a - b)));
}

float egt_tools::PDImitation::evolve(float beta) {

    unsigned int i, p1, p2;
    float freq1, freq2, fitness1, fitness2;
    float ref = _nb_coop;
    int gradient = 0;
    std::uniform_int_distribution<unsigned int> dist(0, _pop_size - 1);
    std::uniform_real_distribution<double> _uniform_real_dist(0.0, 1.0);

    initialize_population(_population);

    // Now run a Moran Process loop
    for (i = 0; i < _generations; i++) {
        _moran_step(p1, p2, gradient, ref, freq1, freq2, fitness1, fitness2, beta, _population, dist, _uniform_real_dist);
        if ((ref == _pop_size) || (ref == 0)) break;
    }
    _final_coop_freq = ref / (float) _pop_size;
    return _final_coop_freq;
}

float egt_tools::PDImitation::evolve(unsigned int runs, float beta) {
    float coop_freq = 0;

    // Run loop
    for (unsigned int j = 0; j < runs; j++) {
        coop_freq += evolve(beta);
    }
    _final_coop_freq = coop_freq / runs;
    return _final_coop_freq;
}

std::vector<float> egt_tools::PDImitation::evolve(std::vector<float> betas) {
    std::vector<float> coop_freqs(betas.size());

    for (unsigned int j = 0; j < betas.size(); j++) {
        coop_freqs[j] = evolve(betas[j]);
    }
    _final_coop_freq = coop_freqs.back();
    return coop_freqs;
}

std::vector<float> egt_tools::PDImitation::evolve(std::vector<float> betas, unsigned int runs) {
    std::vector<float> coop_freqs(betas.size());

    for (unsigned int w = 0; w < betas.size(); w++) {
        coop_freqs[w] = evolve(runs, betas[w]);
    }
    _final_coop_freq = coop_freqs.back();
    return coop_freqs;
}

void egt_tools::PDImitation::_moran_step(unsigned int &p1, unsigned int &p2, int &gradient, float &ref,
                                         float &freq1, float &freq2,
                                         float &fitness1, float &fitness2,
                                         float &beta,
                                         std::vector<unsigned int> &population,
                                         std::uniform_int_distribution<unsigned int> dist,
                                         std::uniform_real_distribution<double> _uniform_real_dist) {
    // Randomly select 2 individuals from the population
    p1 = dist(_mt);
    p2 = dist(_mt);
    while (p2 == p1) p2 = dist(_mt);

    // Calculate frequencies
    freq1 = (ref - population[p1]) / (float) (_pop_size - 1);
    freq2 = (ref - population[p2]) / (float) (_pop_size - 1);

    // Calculate payoffs
    fitness1 = _payoff_matrix(population[p1], 1) * freq1 + _payoff_matrix(population[p1], 0) * (1 - freq1);
    fitness2 = _payoff_matrix(population[p2], 1) * freq2 + _payoff_matrix(population[p2], 0) * (1 - freq2);

    // Select according to fermi function
    if (_uniform_real_dist(_mt) < fermifunc(beta, fitness1, fitness2)) {
        if (_uniform_real_dist(_mt) < _mu) {
            if (_uniform_real_dist(_mt) < 0.5) {
                gradient = -population[p1];
                population[p1] = 0;
            } else {
                gradient = 1 - population[p1];
                population[p1] = 1;
            }
        } else {
            gradient = population[p2] - population[p1];
            population[p1] = population[p2];
        }
    } else {
        if (_uniform_real_dist(_mt) < _mu) {
            if (_uniform_real_dist(_mt) < 0.5) {
                gradient = -population[p2];
                population[p1] = 0;
            } else {
                gradient = 1 - population[p2];
                population[p1] = 1;
            }
        } else {
            gradient = population[p1] - population[p2];
            population[p2] = population[p1];
        }
    }
    ref += gradient;
}

