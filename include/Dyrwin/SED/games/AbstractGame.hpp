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
    template <typename G=std::mt19937_64>
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
    };

    template<typename G>
    void AbstractGame<G>::play(const EGTTools::SED::StrategyCounts &group_composition,
                               PayoffVector &game_payoffs, RandomDist &urand, G &generator) {
        UNUSED(group_composition);
        UNUSED(game_payoffs);
        UNUSED(urand);
        UNUSED(generator);
    }
}

#endif //DYRWIN_ABSTRACTGAME_HPP
