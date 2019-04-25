//
// Created by Elias Fernandez on 2019-04-25.
//

#ifndef DYRWIN_GROUP_HPP
#define DYRWIN_GROUP_HPP

#include <random>
#include <algorithm>
#include <vector>
#include <cmath>
#include <iostream>
#include <Dyrwin/SeedGenerator.h>
#include <Dyrwin/RL/RLUtils.h>
#include <Dyrwin/Types.h>

namespace EGTTools::SED {
    class Group {
    public:
        /**
         * @brief creates a new group that can undergo stochastic dynamics
         *
         * This class creates a group structure. Each of the groups is subject to an internal selection process.
         * Using this population structure, there is a selection process at an individual level, but also another
         * at group level. Certain groups may invade others. At each time step a player is randomly selected
         * proportional to its payoff. In this implementation, first we select the group with a higher aggregated
         * payoff, and then we select a player inside that group proportional to its payoff.
         *
         * @param nb_strategies : number of possible strategies in the population
         * @param max_group_size : maximum capacity (size) of the group
         * @param w : intensity of selection
         * @param init_strategies : number of individuals of each strategy in the group
         * @param payoff_matrix : reference to the payoff matrix
         */
        Group(size_t nb_strategies, size_t max_group_size, double w, VectorXui init_strategies,
              const Matrix2D &payoff_matrix) : _nb_strategies(nb_strategies),
                                               _max_group_size(max_group_size),
                                               _w(w),
                                               _payoff_matrix(payoff_matrix){
            if (payoff_matrix.rows() != payoff_matrix.cols())
                throw std::invalid_argument("Payoff matrix must be a square Matrix (n,n)");
            if (static_cast<size_t>(payoff_matrix.rows()) != nb_strategies)
                throw std::invalid_argument("Payoff matrix must have the same number of rows and columns as trategies");
            if (static_cast<size_t>(init_strategies.size()) != nb_strategies)
                throw std::invalid_argument("size of init strategies must be equal to the number of strategies");

            // Initialize the number of individuals of each strategy
            // we take ownership of the init_strategies vector
            _strategies = std::move(init_strategies);
            _group_size = _strategies.sum();
            _fitness = Vector::Zero(_nb_strategies);
            _group_fitness = 0.0;
            _urand = std::uniform_real_distribution<double>(0.0, 1.0);
            // the number of individuals in the group must be smaller or equal to the maximum capacity
            assert(_group_size <= _max_group_size);
        }

        template <typename G = std::mt19937_64>
        bool createOffspring(G& generator);
        void createMutant(size_t invader, size_t resident);
        double totalPayoff();
        bool addMember(size_t new_strategy); // adds a new member to the group
        template <typename G = std::mt19937_64>
        size_t deleteMember(G& generator);    // delete one randomly chosen member
        template <typename G = std::mt19937_64>
        inline size_t payoffProportionalSelection(G& generator);

        // Getters
        size_t nb_strategies() { return _nb_strategies; }

        size_t max_group_size() { return _max_group_size; }

        size_t group_size() { return _group_size; }

        double group_fitness() { return _group_fitness; }

        double selection_intensity() { return _w; }

        VectorXui &strategies() { return _strategies; }

        const Matrix2D &payoff_matrix() { return _payoff_matrix; }

        // Setters
        void set_max_group_size(size_t max_group_size) { _max_group_size = max_group_size; }

        void set_selection_intensity(double w) { _w = w; }

        void set_strategy_count(const Eigen::Ref<const VectorXui> &strategies) {
            if (strategies.sum() <= _max_group_size)
                throw std::invalid_argument("The sum of all individuals must not be bigger to the maximum group size!");
            _strategies.array() = strategies;
        }

    private:
        // maximum group size (n) and current group size
        size_t _nb_strategies, _max_group_size, _group_size;
        double _group_fitness;                           // group fitness
        double _w;                                      // intensity of selection
        VectorXui _strategies;                         // vector containing the number of individuals of each strategy
        Vector _fitness;                               // container for the fitness of each strategy
        const Matrix2D &_payoff_matrix;                // reference to a payoff matrix
        std::uniform_real_distribution<double> _urand; // uniform random distribution
    };
}


#endif //DYRWIN_GROUP_HPP
