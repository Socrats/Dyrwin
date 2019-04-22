//
// Created by Elias Fernandez on 2019-02-27.
//

#include "../../include/Dyrwin/PyMoran/StochDynamics.h"

EGTTools::StochDynamics::StochDynamics(unsigned int population_size, unsigned int nb_strategies,
                                        Eigen::Ref<const MatrixXd> payoff_matrix) :
        _pop_size(population_size),
        _nb_strategies(nb_strategies),
        _group_size(2),
        _payoff_matrix(payoff_matrix) {

}

EGTTools::StochDynamics::StochDynamics(unsigned int population_size, unsigned int nb_strategies,
                                        unsigned int group_size, Eigen::Ref<const MatrixXd> payoff_matrix) :
        _pop_size(population_size),
        _nb_strategies(nb_strategies),
        _group_size(group_size),
        _payoff_matrix(payoff_matrix) {

    // For N-person dilemmas, we must sample with the hypergeometric
    if (_group_size > 2) {

    }

}

double EGTTools::StochDynamics::fermi(double beta, double a, double b) {
    return 1 / (1 + exp(beta * (b - a)));
}

/**
 * @brief Calculates the fitness of the invader and the resident given k invaders and N-k residents.
 * @param k
 * @param invader
 * @param resident
 * @return a tuple with the fitness of the invader and the resident.
 */
std::tuple<double, double>
EGTTools::StochDynamics::calculate_fitness(int k, unsigned int invader, unsigned int resident) {
    auto resA = (((k - 1) * _payoff_matrix(invader, invader)) +
                 ((_pop_size - k) * _payoff_matrix(invader, resident))) / (double) (_pop_size - 1);
    auto resB = ((k * _payoff_matrix(resident, invader)) +
                 ((_pop_size - k - 1) * _payoff_matrix(resident, resident))) / (double) (_pop_size - 1);

    return std::tuple<double, double>(resA, resB);
}

/**
 * @brief Calculates the probability of increasing the number of invaders in the population.
 * @param k
 * @param invader
 * @param resident
 * @return a tuple with the probability of increasing and decreasing the number of invaders in the population.
 */
std::tuple<double, double>
EGTTools::StochDynamics::probIncreaseDecrease(double beta, int k, unsigned int invader, unsigned int resident) {
    auto fitness = calculate_fitness(k, invader, resident);
    double tmp = (k / (double) _pop_size) * ((_pop_size - k) / (double) _pop_size);
    auto increase = tmp * fermi(-beta, std::get<0>(fitness), std::get<1>(fitness));
    auto decrease = tmp * fermi(beta, std::get<0>(fitness), std::get<1>(fitness));
    return std::tuple<double, double>(increase, decrease);
}

/**
 * @brief Calculates the fixation probability of the invader in a population of residents.
 * @param invader
 * @param resident
 * @return the fixation probability
 */
double EGTTools::StochDynamics::fixation(double beta, unsigned int invader, unsigned int resident) {
    unsigned int i, j;
    double result = 0;

    for (i = 0; i < _pop_size; i++) {
        double sub = 1.;
        for (j = 1; j < i + 1; j++) {
            auto tmp = probIncreaseDecrease(beta, j, invader, resident);
            sub *= (std::get<1>(tmp) / std::get<0>(tmp));
        }
        result += sub;
    }
    return 1 / result;
}

/**
 * @brief Calculates the fixation probability and the transition matrix of the markov chain formed by the
 * homogenous populations.
 * @param beta
 * @return a tuple with the transition probabilities and the fixation probabilities
 */
std::tuple<MatrixXd, MatrixXd>
EGTTools::StochDynamics::calculate_transition_fixations(double beta) {
    unsigned int i, j;
    double tmp;
    MatrixXd transitions = MatrixXd::Zero(_nb_strategies, _nb_strategies);
    MatrixXd fixations = MatrixXd::Zero(_nb_strategies, _nb_strategies);

    for (i = 0; i < _nb_strategies; i++) {
        transitions(i, i) = 1;
        for (j = 0; j < _nb_strategies; j++) {
            if (j != i) {
                auto fp = fixation(beta, i, j);
                fixations(i, j) = fp * _pop_size;
                tmp = fp / (double) (_nb_strategies - 1);
                transitions(i, j) = tmp;
                transitions(i, i) -= tmp;
            }
        }
    }

    return std::tuple<MatrixXd, MatrixXd>(transitions, fixations);
}
