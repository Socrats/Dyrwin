#include <utility>

//
// Created by Elias Fernandez on 2019-05-10.
//

#ifndef DYRWIN_RL_CRDSIM_HPP
#define DYRWIN_RL_CRDSIM_HPP

#include <Dyrwin/RL/BatchQLearningAgent.h>
#include <Dyrwin/RL/QLearningAgent.h>
#include <Dyrwin/RL/RothErevAgent.h>
#include <Dyrwin/RL/HistericQLearningAgent.hpp>
#include <Dyrwin/RL/CRDGame.h>
#include <Dyrwin/RL/PopContainer.hpp>
#include <Dyrwin/RL/Utils.h>

namespace EGTTools::RL {
    class CRDSim {
    public:
        /**
         * @brief Implements a simulator for the Collective Risk dilemma.
         *
         * The main objective of this class is to provide a container for the main
         * simulations that need to be performed in the context of the Collective-risk Dilemma
         * with RL agents.
         *
         * @param nb_episodes number of episodes to be run for each group
         * @param nb_games number of games per episode
         * @param nb_rounds number of rounds per game
         * @param nb_actions number of actions that each player can take
         * @param group_size size of a group
         * @param agent_type algorithm that the agent uses to learn
         * @param risk probability of cataclysm
         * @param args extra arguments for the particular agent_type
         */
        CRDSim(size_t nb_episodes, size_t nb_games,
               size_t nb_rounds,
               size_t nb_actions,
               size_t group_size,
               double risk,
               double endowment,
               double threshold,
               const ActionSpace &available_actions,
               const std::string &agent_type,
               const std::vector<double> &args);

        /**
         * @brief Runs a simulation of the CRD for only 1 group.
         *
         * This method runs a simulation of the CRD for @param nb_episodes.
         * Each episode consists of @param nb_games. The policy of the agents
         * is only updated once an episode finishes.
         *
         * @param nb_episodes number of episodes during which the agents will learn
         * @param nb_games number of games per episode
         * @return EGTTools::Matrix2D containing the group achievement and average contribution
         *         for each episode.
         */
        Matrix2D run(size_t nb_episodes, size_t nb_games);

        /**
         * @brief Runs a simulation of the CRD for multiple independent groups.
         *
         * This method runs a simulation of the CRD for @param nb_episodes.
         * Each episode consists of @param nb_games. The policy of the agents
         * is only updated once an episode finishes. The simulation is performed
         * for @param nb_groups independent groups, and the results of each episode
         * are averaged over groups.
         *
         * @param nb_episodes number of episodes during which the agents will learn
         * @param nb_games number of games per episode
         * @param nb_groups number of independent groups
         * @return EGTTools::Matrix2D containing the group achievement and average contribution
         *         for each episode.
         */
        Matrix2D run(size_t nb_episodes, size_t nb_games, size_t nb_groups);

        void resetPopulation();

        void reinforceOnlyPositive(double &pool, size_t &success);

        void reinforceAll(double &pool, size_t &success);

        void reinforceXico(double &pool, size_t &success);

        size_t nb_games() const;

        size_t nb_episodes() const;

        size_t nb_rounds() const;

        size_t nb_actions() const;

        double endowment() const;

        double risk() const;

        double threshold() const;

        const ActionSpace &available_actions() const;

        void set_nb_games(size_t nb_games);

        void set_nb_episodes(size_t nb_episodes);

        void set_nb_rounds(size_t nb_rounds);

        void set_nb_actions(size_t nb_actions);

        void set_risk(double risk);

        void set_threshold(double threshold);

        void set_available_actions(const ActionSpace &available_actions);

        CRDGame<PopContainer> Game;
        PopContainer population;

    private:
        size_t _nb_episodes, _nb_games, _nb_rounds, _nb_actions, _group_size;
        double _risk, _threshold, _endowment;
        ActionSpace _available_actions;

        void (EGTTools::RL::CRDSim::* _reinforce)(double &, size_t &) = nullptr;

        // Random generators
        std::mt19937_64 _generator{EGTTools::Random::SeedGenerator::getInstance().getSeed()};
    };

}


#endif //DYRWIN_RL_CRDSIM_HPP
