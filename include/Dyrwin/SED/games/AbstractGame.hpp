//
// Created by Elias Fernandez on 2019-06-13.
//

#ifndef DYRWIN_SED_GAMES_ABSTRACTGAME_HPP
#define DYRWIN_SED_GAMES_ABSTRACTGAME_HPP

#include <Dyrwin/Utils.h>
#include <Dyrwin/SED/Utils.hpp>

namespace EGTTools::SED {
    using PayoffVector = std::vector<double>;
    using RandomDist = std::uniform_real_distribution<double>;

    /**
     * @brief This class defines the interface of a game to be used in an evolutionary process.
     * @tparam G : type of random generator.
     */
    template<typename G=std::mt19937_64>
    class AbstractGame {
    public:
        /**
         * @brief updates the vector of payoffs with the payoffs of each player after playing the game.
         *
         * This method will run the game using the players and player types defined in @param group_composition,
         * and will update the vector @param game_payoffs with the resulting payoff of each player.
         *
         * @param nb_strategies number of strategies in the population
         * @param group_composition number of players of each strategy in the group
         * @param game_payoffs container for the payoffs of each player
         * @param urand distribution for uniform random numbers
         * @param generator random generator
         */
        virtual void play(const EGTTools::SED::StrategyCounts &group_composition,
                          PayoffVector &game_payoffs, RandomDist &urand, G &generator);

        /**
         * @brief Estimates the payoff matrix for each strategy.
         *
         * @param urand : uniform random distribution [0, 1).
         * @param generator : random generator
         * @return a payoff matrix
         */
        virtual const GroupPayoffs& calculate_payoffs(RandomDist &urand, G &generator) = 0;

        /**
         * @brief Estimates the fitness for a @param player_type in the population with state @param strategies.
         *
         * @param player_type : index of the strategy used by the player
         * @param pop_size : size of the population
         * @param strategies : current state of the population
         * @param payoffs : the payoff matrix of the game
         * @return a fitness value
         */
        virtual double
        calculate_fitness(const size_t &player_type, const size_t &pop_size, const std::vector<size_t> &strategies) = 0;

        virtual size_t nb_strategies() const;

        /**
         * @return Returns a small description of the game.
         */
        virtual std::string toString() const;

        /**
         *
         * @return The type of game
         */
        virtual std::string type() const;

        /**
         *
         * @return payoff matrix of the game
         */
        virtual const GroupPayoffs & payoffs() const = 0;

        /**
         * @brief stores the payoff matrix in a txt file
         *
         * @param file_name : name of the file in which the data will be stored
         */
        virtual void save_payoffs(std::string file_name) const = 0;
    };

    template<typename G>
    void AbstractGame<G>::play(const EGTTools::SED::StrategyCounts &group_composition,
                               PayoffVector &game_payoffs, RandomDist &urand, G &generator) {
        UNUSED(group_composition);
        UNUSED(game_payoffs);
        UNUSED(urand);
        UNUSED(generator);
    }

    template<typename G>
    std::string AbstractGame<G>::type() const {
        return "Abstract Game";
    }

    template<typename G>
    std::string AbstractGame<G>::toString() const {
        return "This is an abstract game.\n"
               "It should only be used as interface (parent class) for other games.";
    }

    template<typename G>
    size_t AbstractGame<G>::nb_strategies() const {
        return 0;
    }
}

#endif //DYRWIN_ABSTRACTGAME_HPP
