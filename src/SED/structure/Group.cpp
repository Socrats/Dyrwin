//
// Created by Elias Fernandez on 2019-04-25.
//

#include <Dyrwin/SED/structure/Group.hpp>

using namespace EGTTools;

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
 * @brief Checks whether the population inside the group is monomorphic (only one strategy)
 *
 * @return true if monomorphic, otherwise false
 */
bool SED::Group::isPopulationMonomorphic() {
    for (size_t i = 0; i < _nb_strategies; ++i)
        if (_strategies(i) > 0 && _strategies(i) < _group_size)
            return false;
    return true;
}

/**
 * @brief makes the population in the group homonegous
 * @param strategy
 */
void SED::Group::setPopulationHomogeneous(size_t strategy) {
    _group_size = _max_group_size;
    _strategies.setZero();
    _strategies(strategy) = _max_group_size;
}
