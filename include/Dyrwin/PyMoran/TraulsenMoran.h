//
// Created by Elias Fernandez on 2019-04-02.
//

#ifndef DYRWIN_TRAULSENMORAN_H
#define DYRWIN_TRAULSENMORAN_H

#include <random>
#include <algorithm>
#include <vector>
#include <cmath>
#include <iostream>
#include "../SeedGenerator.h"
#include "../rl/RLUtils.h"

/**
 * @brief This class implements the moran process described in Traulsen & Nowak 2006.
 *
 * "Evolution of cooperation by multilevel selection"
 * Here the population evolves in ilands and the offspring may split into
 * a new islan with a probability 1-q, when the iland grows over its
 * population capacity n.
 */

namespace egt_tools {
    class TraulsenMoran {
    public:
        TraulsenMoran(uint64_t generations, unsigned int group_size, unsigned int nb_groups, double beta, double mu,
                      double coop_freq, MatrixXd payoff_matrix);

        ~TraulsenMoran() = default;

        double fermifunc(double beta, double a, double b);

        double evolve(double beta);

        double evolve(unsigned int runs, double beta);

        std::vector<double> evolve(std::vector<double> betas);

        std::vector<double> evolve(std::vector<double> betas, unsigned int runs);

        inline void initialize_population(std::vector<unsigned int> &population);

        inline void initialize_group_coop(std::vector<unsigned int> &group_coop);


        // Getters
        uint64_t generations() { return _generations; }

        unsigned int pop_size() { return _pop_size; }

        unsigned int group_size() { return _group_size; }

        unsigned int nb_groups() { return _nb_groups; }

        unsigned int nb_coop() { return _nb_coop; }

        double mu() { return _mu; }

        double beta() { return _beta; }

        double coop_freq() { return _coop_freq; }

        double result_coop_freq() { return _final_coop_freq; }

        std::vector<unsigned int> group_cooperation() { return _group_coop; }

        MatrixXd payoff_matrix() { return _payoff_matrix; }

        // Setters
        void set_generations(uint64_t generations) { _generations = generations; }

        void set_pop_size(unsigned int pop_size) { _pop_size = pop_size; }

        void set_group_size(unsigned group_size) {
            _group_size = group_size;
            _pop_size = _nb_groups * _group_size;
        }

        void set_nb_groups(unsigned nb_groups) {
            _nb_groups = nb_groups;
            _pop_size = _nb_groups * _group_size;
        }

        void set_beta(double beta) { _beta = beta; }

        void set_coop_freq(double coop_freq) {
            _coop_freq = coop_freq;
            _nb_coop = (unsigned int) floor(_coop_freq * _pop_size);
        }

        void set_nb_coop(unsigned int nb_coop) { _nb_coop = nb_coop; }

        void set_mu(double mu) { _mu = mu; }

        void set_payoff_matrix(MatrixXd payoff_matrix) {
            _payoff_matrix = std::move(payoff_matrix);
        }

    private:
        /**
         * A population with m groups, which all have a maximum size n. Therefore, the maximum population
         * size N = nm. Each group must contain at least one individual. The minimum population size is m.
         * In each time step, an individual is chosen from a population with a probability proportional
         * to its fitness. The individual produces an identical offspring that is added to the same group.
         * If the group size is greater than n after this step, then either a randomly chosen individual
         * from the group is eliminated (with probability 1-q) or the group splits into two groups
         * (with probability q). Each individual of the splitting group has probability 1/2 to end up
         * in each of the daughter groups. One daughter group remains empty with probability 2^(1-n). In this
         * case, the repeating process is repeated to avoid empty groups. In order to keep the number of
         * groups constant, a randomly chosen group is eliminated whenever a group splits in two.
         */
        uint64_t _generations;
        unsigned int _pop_size, _group_size, _nb_groups, _nb_coop;
        double _beta, _mu, _coop_freq, _final_coop_freq;
        // group_coop stores the cooperation level of each group
        // when the group splits, we must select another randomly
        std::vector<unsigned int> _population, _group_coop;
        MatrixXd _payoff_matrix;

        inline void _moran_step(unsigned int &p1, unsigned int &p2, int &gradient, double &ref,
                                double &freq1, double &freq2, double &fitness1, double &fitness2,
                                double &beta,
                                std::vector<unsigned int> &group_coop,
                                std::vector<unsigned int> &population,
                                std::uniform_int_distribution<unsigned int> &dist,
                                std::uniform_real_distribution<double> &_uniform_real_dist);

        // Random generators
        std::mt19937_64 _mt{SeedGenerator::getInstance().getSeed()};
    };
}

#endif //DYRWIN_TRAULSENMORAN_H
