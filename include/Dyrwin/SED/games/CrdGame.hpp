//
// Created by Elias Fernandez on 2019-06-13.
//

#ifndef DYRWIN_SED_GAMES_CRDGAME_HPP
#define DYRWIN_SED_GAMES_CRDGAME_HPP

#include <Dyrwin/SED/games/AbstractGame.hpp>
#include <Dyrwin/SED/behaviors/CrdBehaviors.hpp>

namespace EGTTools::SED::CRD {
    using PayoffVector = std::vector<double>;
    using RandomDist = std::uniform_real_distribution<double>;

    template<typename G=std::mt19937_64>
    class CrdGame : public EGTTools::SED::AbstractGame<G> {
    public:
        /**
         * @brief This class will update the payoffs according to the Collective-risk dilemma defined in Milinski et al. 2008.
         * @param endowment : initial amount each player receives
         * @param threshold : minimum target to avoid losing the endowment
         * @param nb_rounds : number of rounds of the game
         * @param group_size : number of players in the group
         * @param risk : probability that all players will lose their endowment if the target isn't reached
         */
        CrdGame(size_t endowment, size_t threshold, size_t nb_rounds, size_t group_size, double risk);

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

        const GroupPayoffs &calculate_payoffs(RandomDist &urand, G &generator) override;

        double
        calculate_fitness(const size_t &player_type, const size_t &pop_size,
                          const std::vector<size_t> &strategies) override;

        size_t nb_strategies() const override;

        std::string toString() const override;

        std::string type() const override;

        const GroupPayoffs &payoffs() const override;

        void save_payoffs(std::string file_name) const override;

    protected:
        size_t endowment_, threshold_, nb_rounds_, group_size_, nb_strategies_, nb_states_;
        double risk_;
        GroupPayoffs payoffs_;
    };

    template<typename G>
    CrdGame<G>::CrdGame(size_t endowment, size_t threshold, size_t nb_rounds, size_t group_size, double risk)
            : endowment_(endowment),
              threshold_(threshold), nb_rounds_(nb_rounds), group_size_(group_size), risk_(risk) {
        nb_strategies_ = nb_strategies();
        // number of possible group combinations
        nb_states_ = EGTTools::binomialCoeff(nb_strategies_ + group_size_ - 1, group_size_);
        payoffs_ = GroupPayoffs::Zero(nb_strategies_, nb_states_);
        // Initialise payoff matrix
        calculate_payoffs();
    }

    template<typename G>
    void CrdGame<G>::play(const EGTTools::SED::StrategyCounts &group_composition,
                          PayoffVector &game_payoffs, RandomDist &urand, G &generator) {
        size_t prev_donation = 0, current_donation = 0;
        size_t public_account = 0;
        size_t action = 0;
        size_t player_aspiration = (group_size_ - 1) * 2;

        // Initialize payoffs
        for (size_t j = 0; j < EGTTools::SED::CRD::nb_strategies; ++j) {
            if (group_composition[j] > 0) {
                game_payoffs[j] = endowment_;
            } else {
                game_payoffs[j] = 0;
            }
        }

        for (size_t i = 0; i < nb_rounds_; ++i) {
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
    size_t CrdGame<G>::get_action(const size_t &player_type, const size_t &prev_donation, const size_t &threshold,
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

    template<typename G>
    std::string CrdGame<G>::toString() const {
        return "Collective-risk dilemma game.\n"
               "It only plays the game with the strategies described in EGTtools::SED::CRD::behaviors";
    }

    template<typename G>
    std::string CrdGame<G>::type() const {
        return "CrdGame";
    }

    template<typename G>
    size_t CrdGame<G>::nb_strategies() const {
        return EGTTools::SED::CRD::nb_strategies;
    }

    template<typename G>
    const GroupPayoffs &
    CrdGame<G>::calculate_payoffs(RandomDist &urand, G &generator) {
        EGTTools::RL::Factors group_composition(nb_strategies_, 0);
        std::vector<double> game_payoffs(nb_strategies_, 0);
        size_t sum = 0;

        // For every possible group composition run the game and store the payoff of each strategy
        for (size_t i = 0; i < nb_states_; ++i) {
            group_composition.back() = group_size_ - sum;
            play(group_composition, game_payoffs, urand, generator);

            // Fill payoff table
            for (size_t j = 0; j < group_size_; ++j) payoffs_(j, i) = game_payoffs[j];

            // update group composition
            for (size_t k = 0; k < group_size_ - 1; ++k) {
                if (sum < group_size_) {
                    if (group_composition[k] < group_size_) {
                        group_composition[k] += 1;
                        sum += 1;
                        break;
                    } else {
                        sum -= group_composition[k];
                        group_composition[k] = 0;
                    }
                } else {
                    sum -= group_composition[k];
                    group_composition[k] = 0;
                }
            }
        }

        return payoffs_;
    }

    template<typename G>
    double CrdGame<G>::calculate_fitness(const size_t &player_type, const size_t &pop_size,
                                         const std::vector<size_t> &strategies) {
        double fitness, payoff;
        size_t sum;

        fitness = 0.0;
        sum = 0;
        std::vector<size_t> sample_counts(nb_strategies_, 0);

        // number of possible group combinations
        auto total_nb_states = EGTTools::binomialCoeff(nb_strategies_ + group_size_ - 1, group_size_);
        size_t nb_states = EGTTools::binomialCoeff(nb_strategies_ + group_size_ - 2, group_size_ - 1);

        // If it isn't, then we must calculate the fitness for every possible group combination
        for (size_t i = 0; i < nb_states; ++i) {
            sample_counts[player_type] += 1;
            sample_counts.back() = group_size_ - 1 - sum;

            // First update sample_counts with new group composition
            payoff = payoffs_(player_type, EGTTools::SED::calculate_state(group_size_, total_nb_states, sample_counts));
            sample_counts[player_type] -= 1;

            auto prob = EGTTools::multivariateHypergeometricPDF(pop_size - 1, nb_strategies_, group_size_ - 1,
                                                                sample_counts,
                                                                strategies);

            fitness += payoff * prob;

            // update group composition
            for (size_t k = 0; k < nb_strategies_ - 1; ++k) {
                if (sum < group_size_ - 1) {
                    if (sample_counts[k] < group_size_ - 1) {
                        sample_counts[k] += 1;
                        sum += 1;
                        break;
                    } else {
                        sum -= sample_counts[k];
                        sample_counts[k] = 0;
                    }
                } else {
                    sum -= sample_counts[k];
                    sample_counts[k] = 0;
                }
            }

        }

        return fitness;
    }

    template<typename G>
    void CrdGame<G>::save_payoffs(std::string file_name) const {
        // Save payoffs
        std::ofstream file(file_name, std::ios::out | std::ios::trunc);
        if (file.is_open()) {
            file << "Payoffs for each type of player and each possible state:" << std::endl;
            file << "rows: cooperator, defector, altruist, reciprocal, compensator" << std::endl;
            file << "cols: all possible group compositions starting at (0, 0, 0, 0, group_size)" << std::endl;
            file << payoffs_ << std::endl;
            file << "group_size = " << group_size_ << std::endl;
            file << "timing_uncertainty = false" << std::endl;
            file << "nb_rounds = " << nb_rounds_ << std::endl;
            file << "risk = " << risk_ << std::endl;
            file << "endowment = " << endowment_ << std::endl;
            file << "threshold = " << threshold_ << std::endl;
            file.close();
        }
    }

    template<typename G>
    const GroupPayoffs &CrdGame<G>::payoffs() const {
        return payoffs_;
    }
}


#endif //DYRWIN_CRDGAME_HPP
