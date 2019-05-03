//
// Created by Elias Fernandez on 2019-02-27.
//

#ifndef DYRWIN_SED_STOCHDYNAMICS_H
#define DYRWIN_SED_STOCHDYNAMICS_H

#include <random>
#include <vector>
#include <cmath>
#include <tuple>
#include <Eigen/Dense>
#include <iostream>
#include <Dyrwin/SeedGenerator.h>

typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> MatrixXd;

// TODO: Include calculations for the infinite population case
// TODO: Include the possibility to do calculate the full transition matrix (outside the small mutation limit)

namespace EGTTools {
    class StochDynamics {
    public:
        StochDynamics(unsigned int population_size, unsigned int nb_strategies, Eigen::Ref<const MatrixXd> payoff_matrix);
        StochDynamics(unsigned int population_size, unsigned int nb_strategies, unsigned int group_size,
                      Eigen::Ref<const MatrixXd> payoff_matrix);

        ~StochDynamics() = default;

        double fermi(double beta, double a, double b);

        inline std::tuple<double, double> calculate_fitness(int k, unsigned int invader, unsigned int resident);

        std::tuple<double, double>
        probIncreaseDecrease(double beta, int k, unsigned int invader, unsigned int resident);

        double fixation(double beta, unsigned int invader, unsigned int resident);

        std::tuple<MatrixXd, MatrixXd> calculate_transition_fixations(double beta);

        // Getters
        unsigned int pop_size() { return _pop_size; }

        unsigned int nb_strategies() { return _nb_strategies; }

        MatrixXd payoff_matrix() { return _payoff_matrix; }

        // Setters
        void set_pop_size(unsigned int pop_size) { _pop_size = pop_size; }

        void set_nb_strategies(unsigned int nb_strategies) { _nb_strategies = nb_strategies; }

        void set_payoff_matrix(const Eigen::Ref<const MatrixXd> &payoff_matrix) {
            _payoff_matrix = payoff_matrix;
        }

    private:
        unsigned int _pop_size, _nb_strategies, _group_size;
        MatrixXd _payoff_matrix;

        // Random generators
        std::mt19937_64 _mt{EGTTools::Random::SeedGenerator::getInstance().getSeed()};

    };
}

#endif //DYRWIN_SED_STOCHDYNAMICS_H
