//
// Created by Elias Fernandez on 2019-04-25.
//

#include <Dyrwin/SED/structure/Group.hpp>

using namespace EGTTools;

/**
 * @brief Adds a new member of a given strategy to the group (proportional to the fitness).
 *
 * @tparam G : random generator container class
 * @param generator : random generator
 * @return true if group_size <= max_group_size, else false
 */
template<typename G>
bool SED::Group::createOffspring(G &generator) {
    auto new_strategy = payoffProportionalSelection<G>(generator);
    ++_strategies(new_strategy);
    return ++_group_size <= _max_group_size;
}

/**
 * @brief adds a mutant of an invading strategy and reduces one member of the resident strategy.
 *
 * @param invader : index of the invading strategy
 * @param resident : index of the resident strategy
 */
void SED::Group::createMutant(size_t invader, size_t resident) {
    ++_strategies(invader);
    --_strategies(resident);
}

/**
 * @brief Calculates the total fitness of the group
 *
 * @return group fitness
 */
double SED::Group::totalPayoff() {
    if (_group_size == 1) return (1.0 - _w);
    _group_fitness = 0.0;

    for (size_t i = 0; i < _nb_strategies; ++i) {
        if (_strategies(i) == 0) {
            _fitness(i) = 0;
            continue;
        }
        _fitness(i) = 0.0;
        for (size_t j = 0; j < _nb_strategies; ++j) {
            if (j == i) {
                _fitness(i) += _payoff_matrix(i, i) * (_strategies(i) - 1);
            } else {
                _fitness(i) += _payoff_matrix(i, j) * _strategies(j);
            }
        }
        _fitness(i) = ((1.0 - _w) + _w * (_fitness(i) / (_group_size - 1))) * static_cast<double>(_strategies(i));
        _group_fitness += _fitness(i);
    }

    return _group_fitness;
}

bool SED::Group::addMember(size_t new_strategy) {
    ++_strategies(new_strategy);
    return ++_group_size <= _max_group_size;
}

/**
 * @brief deletes a random member from the group
 *
 * @tparam G : random generator container class
 * @param generator : random generator
 * @return index to the deleted member
 */
template<typename G>
size_t SED::Group::deleteMember(G &generator) {
    // choose random member for deletion
    std::uniform_int_distribution<size_t> dist(0, _group_size - 1);
    auto die = dist(generator);
    --_strategie(die);
    return die;
}

/**
 * @brief selects an individual from a strategy proportionally to the payoff
 *
 * @tparam G : random generator container class
 * @param generator : random generator
 * @return : index of the strategy selected
 */
template<typename G>
size_t SED::Group::payoffProportionalSelection(G &generator) {
    double sum = 0.0;
    auto p = _urand(generator) * _group_size;
    for (size_t i = 0; i < _nb_strategies; ++i) {
        sum += _fitness(i);
        if (p < sum) return i;
    }
    // It should never get here
    assert(p < sum);
}
