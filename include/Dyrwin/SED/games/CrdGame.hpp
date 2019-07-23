//
// Created by Elias Fernandez on 2019-06-13.
//

#ifndef DYRWIN_SED_GAMES_CRDGAME_HPP
#define DYRWIN_SED_GAMES_CRDGAME_HPP

#include <Dyrwin/Distributions.h>
#include <Dyrwin/SED/games/AbstractGame.hpp>
#include <Dyrwin/SED/behaviors/CrdBehaviors.hpp>

namespace EGTTools::SED::CRD {
    using PayoffVector = std::vector<double>;
    using RandomDist = std::uniform_real_distribution<double>;

    class CrdGame : public EGTTools::SED::AbstractGame {
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
                  PayoffVector &game_payoffs) override;

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

        /**
         * @brief updates private payoff matrix and returns it
         *
         * @return payoff matrix of the game
         */
        const GroupPayoffs &calculate_payoffs() override;

        double
        calculate_fitness(const size_t &player_type, const size_t &pop_size,
                          const Eigen::Ref<const VectorXui> &strategies) override;

        /**
         * @brief Calculates the group achievement for all possible groups
         *
         * If the strategies are deterministic, the output matrix will consist
         * of ones and zeros indicating whether the group reached or not the target.
         * If they are stochastic, it will indicate the probability of success of
         * the group.
         *
         * @return a matrix with the group achievement for each possible group
         */
        const Vector &calculate_success_per_group_composition();

        /**
         * @brief Calculates the probability of success given a population state
         * @param pop_size : size of the population
         * @param population_state : state of the population (number of players of each strategy)
         * @return the group achievement of that population state
         */
        double
        calculate_population_group_achievement(size_t pop_size, const Eigen::Ref<const VectorXui> &population_state);

        /**
         * @brief estimates the group achievement from a stationary distribution
         * @param pop_size : size of the population
         * @param stationary_distribution
         * @return group achievement (probability of group success)
         */
        double calculate_group_achievement(size_t pop_size, const Eigen::Ref<const Vector> &stationary_distribution);

        /**
         * @brief Calculates the fraction of players that invest >, < or = to E/2.
         *
         * Calculates the fraction of players that invest above, below or equal to the fair donation
         * given a population state.
         *
         * @param pop_size : size of the population
         * @param population_state : state of the population
         * @return an array of 3 elements [C < E/2, C = E/2, C > E/2]
         */
        double *calculate_polarization(size_t pop_size, const Eigen::Ref<const VectorXui> &population_state);

        // getters
        size_t endowment() const;
        size_t target() const;
        size_t nb_rounds() const;
        size_t group_size() const;

        size_t nb_strategies() const override;

        size_t nb_states() const;

        double risk() const;

        std::string toString() const override;

        std::string type() const override;

        const GroupPayoffs &payoffs() const override;

        double payoff(size_t strategy, const EGTTools::SED::StrategyCounts &group_composition) const override;

        void save_payoffs(std::string file_name) const override;

        const Vector &group_achievements() const;

    protected:
        size_t endowment_, threshold_, nb_rounds_, group_size_, nb_strategies_, nb_states_;
        double risk_;
        GroupPayoffs payoffs_;
        Vector group_achievement_;

        // Random distributions
        std::uniform_real_distribution<double> real_rand_;

        // Random generators
        std::mt19937_64 generator_{EGTTools::Random::SeedGenerator::getInstance().getSeed()};

        // Check if game is successful and update state in group_achievement_
        void _check_success(size_t state, PayoffVector &game_payoffs, const EGTTools::SED::StrategyCounts &group_composition);
    };
}

#endif //DYRWIN_CRDGAME_HPP
