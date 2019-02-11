#include <utility>

//
// Created by Elias Fernandez on 2019-02-11.
//

#include "../../include/Dyrwin/PyMoran/PDImitation.h"

egt_tools::PDImitation::PDImitation(unsigned int generations, unsigned int pop_size, float beta,
                                    float mu, float coop_freq,
                                    std::vector<float> &payoff_matrix) : _generations(generations),
                                                                        _pop_size(pop_size), _beta(beta),
                                                                        _mu(mu),
                                                                        _coop_freq(coop_freq),
                                                                        _payoff_matrix(payoff_matrix) {

    std::uniform_real_distribution<double> _uniform_real_dist(0.0, 1.0);
    _nb_coop = (unsigned int) floor((double) (_coop_freq * _pop_size));
    _population = std::vector<unsigned int>(_pop_size);

    initialize_population();

}

void egt_tools::PDImitation::initialize_population() {
    unsigned int i;

    for (i = 0; i < _nb_coop; i++) {
        _population[i] = 0;
    }
    for (i = _nb_coop; i < _pop_size; i++) {
        _population[i] = 1;
    }

    // Then we shuffle it randomly
    shuffle(_population.begin(), _population.end(), _mt);
}

float egt_tools::PDImitation::fermifunc(float beta, float a, float b) {
    return 1 / (1 + exp(beta * (a - b)));
}

float egt_tools::PDImitation::evolve() {

    unsigned int i, p1, p2;
    float freq1, freq2, fitness1, fitness2;
    float ref = _nb_coop;
    int gradient = 0;
    std::uniform_int_distribution<unsigned int> dist(0, _pop_size - 1);
    std::uniform_real_distribution<double> _uniform_real_dist(0.0, 1.0);

    // Now run a Moran Process loop
    for (i = 0; i < _generations; i++) {
        _moran_step(p1, p2, gradient, ref, freq1, freq2, fitness1, fitness2, dist, _uniform_real_dist);
        if ((ref == _pop_size) || (ref == 0)) break;
    }
    _final_coop_freq = ref / (float) _pop_size;
    return _final_coop_freq;
}

float egt_tools::PDImitation::evolve(unsigned int &runs) {
    unsigned int i, j, p1, p2;
    float freq1, freq2, fitness1, fitness2, coop_freq = 0;
    float ref;
    int gradient = 0;
    std::uniform_int_distribution<unsigned int> dist(0, _pop_size - 1);
    std::uniform_real_distribution<double> _uniform_real_dist(0.0, 1.0);

    // Run loop
    for (j = 0; j < runs; j++) {
        ref = _nb_coop;
        initialize_population();

        // Now run a Moran Process loop
        for (i = 0; i < _generations; i++) {
            _moran_step(p1, p2, gradient, ref, freq1, freq2, fitness1, fitness2, dist, _uniform_real_dist);
            if ((ref == _pop_size) || (ref == 0)) break;
        }
        coop_freq += ref / (float) _pop_size;
    }
    _final_coop_freq = coop_freq / runs;
    return _final_coop_freq;
}

std::vector<float> egt_tools::PDImitation::evolve(std::vector<float> &betas) {
    unsigned int i, j, p1, p2;
    float freq1, freq2, fitness1, fitness2;
    float ref;
    int gradient = 0;
    std::uniform_int_distribution<unsigned int> dist(0, _pop_size - 1);
    std::uniform_real_distribution<double> _uniform_real_dist(0.0, 1.0);

    std::vector<float> coop_freqs(betas.size());

    for (j = 0; j < betas.size(); j++) {
        _beta = betas[j];
        ref = _nb_coop;
        initialize_population();

        // Now run a Moran Process loop
        for (i = 0; i < _generations; i++) {
            _moran_step(p1, p2, gradient, ref, freq1, freq2, fitness1, fitness2, dist, _uniform_real_dist);
            if ((ref == _pop_size) || (ref == 0)) break;
        }
        coop_freqs[j] = ref / (float) _pop_size;
    }
    _final_coop_freq = coop_freqs.back();
    return coop_freqs;
}

std::vector<float> egt_tools::PDImitation::evolve(std::vector<float> &betas, unsigned int &runs) {
    unsigned int i, j, w, p1, p2;
    float freq1, freq2, fitness1, fitness2, coop_freq = 0;
    float ref;
    int gradient = 0;
    std::uniform_int_distribution<unsigned int> dist(0, _pop_size - 1);
    std::uniform_real_distribution<double> _uniform_real_dist(0.0, 1.0);

    std::vector<float> coop_freqs(betas.size());

    for (w = 0; w < betas.size(); w++) {
        _beta = betas[w];
        coop_freq = 0;
        // Run loop
        for (j = 0; j < runs; j++) {
            ref = _nb_coop;
            initialize_population();
            // Now run a Moran Process loop
            for (i = 0; i < _generations; i++) {
                _moran_step(p1, p2, gradient, ref, freq1, freq2, fitness1, fitness2, dist, _uniform_real_dist);
                if ((ref == _pop_size) || (ref == 0)) break;
            }
            coop_freq += ref / (float) _pop_size;
        }
        coop_freqs[w] = coop_freq / runs;
    }
    _final_coop_freq = coop_freqs.back();
    return coop_freqs;
}

void egt_tools::PDImitation::_moran_step(unsigned int &p1, unsigned int &p2, int &gradient, float &ref,
                                         float &freq1, float &freq2,
                                         float &fitness1, float &fitness2,
                                         std::uniform_int_distribution<unsigned int> dist,
                                         std::uniform_real_distribution<double> _uniform_real_dist) {
    // Randomly select 2 individuals from the population
    p1 = dist(_mt);
    p2 = dist(_mt);
    while (p2 == p1) p2 = dist(_mt);

    // Calculate frequencies
    freq1 = (ref - _population[p1]) / (float) (_pop_size - 1);
    freq2 = (ref - _population[p2]) / (float) (_pop_size - 1);

    // Calculate payoffs
    fitness1 = _payoff_matrix[(2 * _population[p1]) + 1] * freq1 +
               _payoff_matrix[(2 * _population[p1]) + 0] * (1 - freq1);
    fitness2 = _payoff_matrix[(2 * _population[p2]) + 1] * freq2 +
               _payoff_matrix[(2 * _population[p2]) + 0] * (1 - freq2);

    // Select according to fermi function
    if (_uniform_real_dist(_mt) < fermifunc(_beta, fitness1, fitness2)) {
        if (_uniform_real_dist(_mt) < _mu) {
            if (_uniform_real_dist(_mt) < 0.5) {
                gradient = -_population[p1];
                _population[p1] = 0;
            } else {
                gradient = 1 - _population[p1];
                _population[p1] = 1;
            }
        } else {
            gradient = _population[p2] - _population[p1];
            _population[p1] = _population[p2];
        }
    } else {
        if (_uniform_real_dist(_mt) < _mu) {
            if (_uniform_real_dist(_mt) < 0.5) {
                gradient = -_population[p2];
                _population[p1] = 0;
            } else {
                gradient = 1 - _population[p2];
                _population[p1] = 1;
            }
        } else {
            gradient = _population[p1] - _population[p2];
            _population[p2] = _population[p1];
        }
    }
    ref += gradient;
}

