//
// Created by Elias Fernandez on 2019-04-25.
//

#include <Dyrwin/SED/structure/GarciaGroup.hpp>

using namespace EGTTools;

/**
 * @brief adds a mutant of an invading strategy and reduces one member of the resident strategy.
 *
 * @param invader : index of the invading strategy
 * @param resident : index of the resident strategy
 */
void SED::GarciaGroup::createMutant(size_t invader, size_t resident) {
    ++_strategies(invader);
    --_strategies(resident);
}


double SED::GarciaGroup::totalPayoff(const double &alpha, EGTTools::VectorXui &strategies) {
    size_t out_pop_size = strategies.sum() - _group_size;
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
                _fitness(i) += alpha * ((_payoff_matrix_in(i, i) * (_strategies(i) - 1)) / (_group_size - 1)) +
                               (1 - alpha) *
                               ((_payoff_matrix_out(i, i) * (strategies(i) - _strategies(i))) / out_pop_size);
            } else {
                _fitness(i) += alpha * (_payoff_matrix_in(i, j) * _strategies(j) / (_group_size - 1)) +
                               (1 - alpha) *
                               ((_payoff_matrix_out(i, j) * (strategies(j) - _strategies(j))) / out_pop_size);
            }
        }
        _fitness(i) = ((1.0 - _w) + _w * (_fitness(i) / (_group_size - 1))) * static_cast<double>(_strategies(i));
        _group_fitness += _fitness(i);
    }

    return _group_fitness;
}

bool SED::GarciaGroup::addMember(size_t new_strategy) {
    ++_strategies(new_strategy);
    return ++_group_size <= _max_group_size;
}

bool SED::GarciaGroup::deleteMember(const size_t &member_strategy) {
    if (_strategies(member_strategy) <= 0) return false;
    --_strategies(member_strategy);
    --_group_size;
    return true;
}

/**
 * @brief Checks whether the population inside the group is monomorphic (only one strategy)
 *
 * @return true if monomorphic, otherwise false
 */
bool SED::GarciaGroup::isPopulationMonomorphic() {
    for (size_t i = 0; i < _nb_strategies; ++i)
        if (_strategies(i) > 0 && _strategies(i) < _group_size)
            return false;
    return true;
}

/**
 * @brief makes the population in the group homonegous
 * @param strategy
 */
void SED::GarciaGroup::setPopulationHomogeneous(size_t strategy) {
    _group_size = _max_group_size;
    _strategies.setZero();
    _strategies(strategy) = _max_group_size;
}

SED::GarciaGroup &SED::GarciaGroup::operator=(const SED::GarciaGroup &grp) {
    if (this == &grp) return *this;

    _nb_strategies = grp.nb_strategies();
    _max_group_size = grp.max_group_size();
    _group_size = grp.group_size();
    _group_fitness = grp.group_fitness();
    _w = grp.selection_intensity();
    _strategies = grp.strategies();

    return *this;
}

bool SED::GarciaGroup::isGroupOversize() {
    return _group_size > _max_group_size;
}
