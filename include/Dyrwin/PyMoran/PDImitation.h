#include <utility>

//
// Created by Elias Fernandez on 2019-02-11.
//

#ifndef DYRWIN_PDIMITATION_H
#define DYRWIN_PDIMITATION_H

#include <iostream>
#include <random>
#include <vector>
#include <cmath>
#include "../SeedGenerator.h"

/**
 * This class implements the moran process for a prisoner's dilema game
 */

namespace egt_tools {
    class PDImitation {
    public:
        PDImitation(unsigned int generations, unsigned int pop_size, float beta, float mu,
                    float coop_freq, std::vector<float> payoff_matrix);

        ~PDImitation() = default;

        float fermifunc(float beta, float a, float b);

        float evolve(float beta);

        float evolve(unsigned int runs, float beta);

        std::vector<float> evolve(std::vector<float> betas);

        std::vector<float> evolve(std::vector<float> betas, unsigned int runs);

        inline void initialize_population(std::vector<unsigned int> &population);


        // Getters
        unsigned int generations() { return _generations; }

        unsigned int pop_size() { return _pop_size; }

        unsigned int nb_coop() { return _nb_coop; }

        float mu() { return _mu; }

        float beta() { return _beta; }

        float coop_freq() { return _coop_freq; }

        float result_coop_freq() { return _final_coop_freq; }

        std::vector<float> payoff_matrix() {return _payoff_matrix;}

        // Setters
        void set_generations(unsigned int generations) { _generations = generations; }

        void set_pop_size(unsigned int pop_size) { _pop_size = pop_size; }

        void set_beta(float beta) { _beta = beta; }

        void set_coop_freq(float coop_freq) { _coop_freq = coop_freq; }

        void set_mu(float mu) { _mu = mu; }

        void set_payoff_matrix(std::vector<float> payoff_matrix) {
            _payoff_matrix = std::move(payoff_matrix);
        }

    private:
        unsigned int _generations, _pop_size, _nb_coop;
        float _beta, _mu, _coop_freq, _final_coop_freq;
        std::vector<unsigned int> _population;
        std::vector<float> _payoff_matrix;

        inline void _moran_step(unsigned int &p1, unsigned int &p2, int &gradient, float &ref,
                               float &freq1, float &freq2, float &fitness1, float &fitness2,
                               float &beta,
                               std::vector<unsigned int> &population,
                               std::uniform_int_distribution<unsigned int> dist,
                               std::uniform_real_distribution<double> _uniform_real_dist);

        // Random generators
        std::mt19937_64 _mt{SeedGenerator::getInstance().getSeed()};
    };
}

#endif //DYRWIN_PDIMITATION_H
