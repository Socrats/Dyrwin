//
// Created by Elias Fernandez on 2019-06-13.
//

#ifndef DYRWIN_SED_GAMES_CRDGAMETU_HPP
#define DYRWIN_SED_GAMES_CRDGAMETU_HPP

#include <Dyrwin/Distributions.h>
#include <Dyrwin/SED/games/AbstractGame.hpp>
#include <Dyrwin/SED/behaviors/CrdBehaviors.hpp>
#include <Dyrwin/RL/TimingUncertainty.hpp>

namespace EGTTools::SED::CRD {
    using PayoffVector = std::vector<double>;
    using RandomDist = std::uniform_real_distribution<double>;

    class CrdGameTU : public EGTTools::SED::AbstractGame {
    public:
        /**
         * @brief This class will update the payoffs according to the Collective-risk dilemma with timing uncertianty.
         *
         * The number of rounds of the game is uncertain and follows a geometric distribution.
         *
         * @param endowment : initial budget of the player
         * @param threshold : minimum target to avoid losing all the endowment
         * @param min_rounds : minimum number of rounds
         * @param group_size : group size
         * @param risk : probability of losing the endomwent if the target isn't reached
         * @param tu : class that calculates the total number of rounds through a geometric distribution
         */
        CrdGameTU(size_t endowment, size_t threshold, size_t min_rounds, size_t group_size, double risk,
                  EGTTools::TimingUncertainty<std::mt19937_64> &tu);

        void play(const EGTTools::SED::StrategyCounts &group_composition, PayoffVector &game_payoffs) override;

        /**
         * @brief Gets an action from the strategy defined by player type.
         *
         * This method will call one of the behaviors specified in CrdBehaviors.hpp indexed by
         * @param player_type with the parameters @param prev_donation, threshold, current_round.
         *
         * @param player_type : type of strategy (as an unsigned integer).
         * @param prev_donation : previous donation of the group.
         * @param threshold : Aspiration level of the strategy.
         * @param current_round : current round of the game
         * @return action of the strategy
         */
        static inline size_t get_action(const size_t &player_type, const size_t &prev_donation, const size_t &threshold,
                                 const size_t &current_round);

        const GroupPayoffs &calculate_payoffs() override;

        double
        calculate_fitness(const size_t &player_type, const size_t &pop_size,
                          const std::vector<size_t> &strategies) override;

        size_t nb_strategies() const override;

        std::string toString() const override;

        std::string type() const override;

        const GroupPayoffs &payoffs() const override;

        void save_payoffs(std::string file_name) const override;

    private:
        size_t endowment_, threshold_, min_rounds_, group_size_, nb_strategies_, nb_states_;
        double risk_;
        GroupPayoffs payoffs_;
        EGTTools::TimingUncertainty<std::mt19937_64> tu_;

        // Random distributions
        std::uniform_real_distribution<double> real_rand_;

        // Random generators
        std::mt19937_64 generator_{EGTTools::Random::SeedGenerator::getInstance().getSeed()};
    };
}

#endif //DYRWIN_CRDGAME_HPP
