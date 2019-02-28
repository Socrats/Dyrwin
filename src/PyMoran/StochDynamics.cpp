//
// Created by Elias Fernandez on 2019-02-27.
//

#include "../../include/Dyrwin/PyMoran/StochDynamics.h"

egt_tools::StochDynamics::StochDynamics(unsigned int population_size, unsigned int nb_strategies,
                                        std::vector<double> payoff_matrix) :
        _pop_size(population_size),
        _nb_strategies(nb_strategies),
        _payoff_matrix(std::move(payoff_matrix)) {

}

double egt_tools::StochDynamics::fermi(double beta, double a, double b) {
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
egt_tools::StochDynamics::calculate_fitness(int k, unsigned int invader, unsigned int resident) {
    auto resA = (((k - 1) * _payoff_matrix[(_nb_strategies * invader) + invader]) +
            ((_pop_size - k) * _payoff_matrix[(_nb_strategies * invader) + resident])) / (double) (_pop_size - 1);
    auto resB = ((k * _payoff_matrix[(_nb_strategies * resident) + invader]) +
            ((_pop_size - k - 1) * _payoff_matrix[(_nb_strategies * resident) + resident])) / (double) (_pop_size - 1);

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
egt_tools::StochDynamics::probIncreaseDecrease(double beta, int k, unsigned int invader, unsigned int resident) {
    auto fitness = calculate_fitness(k, invader, resident);
    double tmp = ((_pop_size - k) * k) / (double) _pop_size;
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
double egt_tools::StochDynamics::fixation(double beta, unsigned int invader, unsigned int resident) {
    unsigned int i, j;
    double result = 0;

    for (i = 1; i < _pop_size; i++) {
        double sub = 1.;
        for (j = 1; j < i + 1; j++) {
            auto tmp = probIncreaseDecrease(beta, j, invader, resident);
            sub *= (std::get<1>(tmp) / std::get<0>(tmp));
        }
        result += sub;
    }
    return 1 / (1. + result);
}

/**
 * @brief Calculates the fixation probability and the transition matrix of the markov chain formed by the
 * homogenous populations.
 * @param beta
 * @return a tuple with the transition probabilities and the fixation probabilities
 */
std::tuple<std::vector<double>, std::vector<double>>
egt_tools::StochDynamics::calculate_transition_fixations(double beta) {
    unsigned int i, j;
    double tmp;
    std::vector<double> transitions(_nb_strategies * _nb_strategies, 0), fixations(_nb_strategies * _nb_strategies, 0);

    for (i = 0; i < _nb_strategies; i++) {
        transitions[(_nb_strategies * i) + i] = 1;
        for (j = 0; j < _nb_strategies; j++) {
            if (j != i) {
                auto fp = fixation(beta, i, j);
                fixations[(_nb_strategies * i) + j] = fp * _pop_size;
                tmp = fp / (double) (_nb_strategies - 1);
                transitions[(_nb_strategies * i) + j] = tmp;
                transitions[(_nb_strategies * i) + i] -= tmp;
            }
        }
    }

    return std::tuple<std::vector<double>, std::vector<double>>(transitions, fixations);
}
