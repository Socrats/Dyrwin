//
// Created by Elias Fernandez on 2019-06-13.
//

#ifndef DYRWIN_SED_GAMES_CRDGAMETU_HPP
#define DYRWIN_SED_GAMES_CRDGAMETU_HPP

#include <Dyrwin/SED/games/AbstractGame.hpp>
#include <Dyrwin/SED/behaviors/CrdBehaviors.hpp>
#include <Dyrwin/RL/TimingUncertainty.hpp>

namespace EGTTools::SED::CRD {
    using PayoffVector = std::vector<double>;
    using RandomDist = std::uniform_real_distribution<double>;

    template<typename G=std::mt19937_64>
    class CrdGameTU : public EGTTools::SED::AbstractGame<G> {
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
                  EGTTools::TimingUncertainty<G> &tu);

        void play(const EGTTools::SED::StrategyCounts &group_composition,
                  PayoffVector &game_payoffs, RandomDist &urand, G &generator) override;

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
        inline size_t get_action(const size_t &player_type, const size_t &prev_donation, const size_t &threshold,
                                 const size_t &current_round);

    private:
        size_t endowment_, threshold_, min_rounds_, group_size_;
        double risk_;
        EGTTools::TimingUncertainty<G> tu_;
    };

    template<typename G>
    CrdGameTU<G>::CrdGameTU(size_t endowment, size_t threshold, size_t min_rounds, size_t group_size, double risk,
                            EGTTools::TimingUncertainty<G> &tu)
            : endowment_(endowment),
              threshold_(threshold), min_rounds_(min_rounds), group_size_(group_size), risk_(risk), tu_(std::move(tu)) {
    }

    template<typename G>
    void CrdGameTU<G>::play(const EGTTools::SED::StrategyCounts &group_composition,
                            PayoffVector &game_payoffs, RandomDist &urand, G &generator) {
        size_t prev_donation = 0, current_donation = 0;
        size_t public_account = 0;
        size_t action = 0;
        size_t player_aspiration = (group_size_ - 1) * 2;
        size_t game_rounds = tu_.calculateFullEnd(min_rounds_, generator);

        // Initialize payoffs
        for (size_t j = 0; j < EGTTools::SED::CRD::nb_strategies; ++j) {
            if (group_composition[j] > 0) {
                game_payoffs[j] = endowment_;
            } else {
                game_payoffs[j] = 0;
            }
        }

        for (size_t i = 0; i < game_rounds; ++i) {
            for (size_t j = 0; j < EGTTools::SED::CRD::nb_strategies; ++j) {
                if (group_composition[j] > 0) {
                    action = get_action(j, prev_donation - action, player_aspiration, i);
                    if (game_payoffs[j] - action > 0) {
                        game_payoffs[j] -= action;
                        current_donation += group_composition[j] * action;
                    }
                }
            }
            public_account += current_donation;
            prev_donation = current_donation;
            current_donation = 0;
            if (public_account >= threshold_) break;
        }

        if (public_account < threshold_)
            if (urand(generator) < risk_) for (auto &type: game_payoffs) type = 0;
    }

    template<typename G>
    size_t CrdGameTU<G>::get_action(const size_t &player_type, const size_t &prev_donation, const size_t &threshold,
                                  const size_t &current_round) {
        switch (player_type) {
            case 0:
                return EGTTools::SED::CRD::cooperator(prev_donation, threshold, current_round);
            case 1:
                return EGTTools::SED::CRD::defector(prev_donation, threshold, current_round);
            case 2:
                return EGTTools::SED::CRD::altruist(prev_donation, threshold, current_round);
            case 3:
                return EGTTools::SED::CRD::reciprocal(prev_donation, threshold, current_round);
            case 4:
                return EGTTools::SED::CRD::compensator(prev_donation, threshold, current_round);
            default:
                assert(false);
                throw std::invalid_argument("invalid player type: " + std::to_string(player_type));
        }
    }
}


#endif //DYRWIN_CRDGAME_HPP
