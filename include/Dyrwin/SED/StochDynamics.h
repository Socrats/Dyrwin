//
// Created by Elias Fernandez on 2019-02-27.
//

#ifndef DYRWIN_SED_STOCHDYNAMICS_H
#define DYRWIN_SED_STOCHDYNAMICS_H

#include <cmath>
#include <random>
#include <vector>
#include <tuple>
#include <Eigen/Dense>
#include <iostream>
#include <Dyrwin/SeedGenerator.h>

typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> MatrixXd;

// TODO: Include the possibility to do calculate the full transition matrix (outside the small mutation limit)

namespace EGTTools {
    /**
     * @brief Stochastic Dynamics among populations.
     *
     * This class implements methods to compute/estimate relevant parameters to characterize
     * the stochastic dynamics among populations of competing individuals. These methods assume
     * that the payoffs of each strategy against another can be computed numerically. The class
     * expects a matrix indicating the payoffs relationships among strategies.
     *
     * In this class we always assume that selection is defined by a birth-death process
     * (or Moran process), and that individuals update their strategies through social learning,
     * i.e., two individuals are selected randomly for pair-wise comparison through the fermi
     * rule. Individual A is selected for birth and individual B is selected for death. With
     * a probability $\mu$ B will mutate into a random strategy. With probability $1-\mu$,
     * if the fitness of A is bigger than the fitness of B, f_A > f_B, then B will imitate
     * the strategy of A with probability $p = [1 + e^{\beta(f_B - f_A)}]^-1$. Otherwise,
     * B will remain with the same strategy.
     *
     * To analyse the stochastic dynamics in finite populations, we estimate the Markov Chain
     * defined by the possible states the population can be in. A population state is a tuple,
     * where every element indicates the strategy of each individual in the population
     * S = (0, 1, 0, 3, ..., s_Z). This allows us to find which states are in equilibrium,
     * i.e., the population will spend more time in equilibrium states. For example, in a
     * Prisoner's Dilemma, the population will spend most of the time under full defection.
     *
     * We may find saddle points and absorbing states, by calculating the gradient of selection
     */
    class StochDynamics {
    public:
        /**
         * @brief Stochastic evolutionary dynamics in finite populations
         *
         * This constructor assumes a 2-player game.
         *
         * @param population_size : size of the population (Z)
         * @param nb_strategies : number of strategies
         * @param payoff_matrix : payoff matrix containing the payoffs of all strategies against each other
         */
        StochDynamics(unsigned int population_size, unsigned int nb_strategies,
                      Eigen::Ref<const MatrixXd> payoff_matrix);

        /**
         * @brief Stochastic evolutionary dynamics in finite populations
         *
         * This constructor accepts N-player games.
         *
         * @param population_size
         * @param nb_strategies
         * @param group_size
         * @param payoff_matrix
         */
        StochDynamics(unsigned int population_size, unsigned int nb_strategies, unsigned int group_size,
                      Eigen::Ref<const MatrixXd> payoff_matrix);

        ~StochDynamics() = default;

        /**
         * @brief Implements the fermi rule
         *
         * @deprecated This function has been deprecated and will be dropped in future version.
         * Now it is favored to use EGTTools::SED::fermi
         *
         * @param beta
         * @param a
         * @param b
         * @return (double) the probability of imitation/survival
         */
        double fermi(double beta, double a, double b);

        /**
         * @brief calculates the fitness of 2-player games
         *
         * @param k : number of individuals of the invader type in the population
         * @param invader : index of the invader type
         * @param resident : index of the resident type
         * @return : a tuple with the fitness of both resident and invader types given the state of the population
         */
        inline std::tuple<double, double> calculate_fitness(int k, unsigned int invader, unsigned int resident);

        /**
         * @brief calculates the fitness of N-player games
         * @param k : number of individuals of the invater type in the population
         * @param invader : index of the invader type
         * @param resident : index of the resident type
         * @return : a tuple with the fitness of both resident and invader types given the state of the population
         */
        inline std::tuple<double, double> calculate_fitness_nplayer(int k, unsigned int invader, unsigned int resident);

        /**
         * @brief Calculates the probability of increasing and decreasing by 1 the number of invading type individuals.
         *
         * Calculates T+ (probability of increasing by 1 the number of invading individuals)
         * and T- (probability of decreasing by 1 the number of invading individuals).
         *
         * @param beta : intensity of selection
         * @param k : number of individuals of the invading type
         * @param invader : index of the invading type
         * @param resident : index of the resident type
         * @return : a tuple containing T+ and T-
         */
        std::tuple<double, double>
        probIncreaseDecrease(double beta, int k, unsigned int invader, unsigned int resident);

        /**
         * @brief Calculates the fixation probability
         *
         * Fixation probabilities indicate the probability that 1 mutant individual
         * of an invading type will take over the population.
         * @f[phi = (\sum_{k=0}^{N-1}\prod_{i=1}^{k}(\lambda^i))^{-1} @f]
         *
         * @param beta : intensity of selection
         * @param invader : index of the invading type
         * @param resident : index of the resident type
         * @return : the fixation probability
         */
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
