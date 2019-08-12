//
// Created by Elias Fernandez on 2019-03-11.
//

#ifndef DYRWIN_RL_CRDDEMOCRACY_H
#define DYRWIN_RL_CRDDEMOCRACY_H

#include <random>
#include <vector>
#include <cmath>
#include <tuple>
#include <iostream>
#include <Dyrwin/SeedGenerator.h>
#include <Dyrwin/RL/Agent.h>
#include <Dyrwin/RL/TimingUncertainty.hpp>
#include <Dyrwin/RL/Utils.h>


namespace EGTTools::RL {
    /**
     * @brief Implements the Collective-risk dilemma defined in Milinski et. al 2008.
     *
     * @tparam A. Container for the agents.
     */
    template<typename A = Agent, typename R = void>
    class CRDDemocracy {

    public:
        /**
         * @brief Model of the Collective-Risk dillemma game.
         *
         * This game constitutes an MDP.
         *
         * This function plays the game for a number of rounds
         *
         * @param players
         * @param actions
         * @param rounds
         * @return std::tuple (donations, rounds)
         */
        std::pair<double, size_t>
        playGame(std::vector<A> &players, std::vector<size_t> &actions, size_t rounds, R &gen_round) {

            auto final_round = gen_round.calculateEnd(rounds, _mt);

            double total = 0.0, partial;
            size_t nb_players = players.size();
            for (auto &player : players) {
                player.resetPayoff();
            }
            for (size_t i = 0; i < final_round; ++i) {
                partial = 0.0;
//#pragma omp parallel for shared(partial)
                for (size_t j = 0; j < players.size(); ++j) {
                    auto idx = players[j].selectAction(i);
                    partial += actions[idx];
                }
                partial = std::round(partial / nb_players);
                total += partial * nb_players;

//#pragma omp parallel for shared(partial)
                for (size_t j = 0; j < players.size(); ++j) {
                    players[j].decrease(partial);
                }
            }
            return std::make_pair(total, final_round);
        }

        std::pair<double, size_t>
        playGame(std::vector<A *> &players, std::vector<size_t> &actions, size_t rounds, R &gen_round) {

            auto final_round = gen_round.calculateEnd(rounds, _mt);

            double total = 0.0, partial;
            size_t nb_players = players.size();
            for (auto &player : players) {
                player->resetPayoff();
            }
            for (size_t i = 0; i < final_round; ++i) {
                partial = 0.0;
//#pragma omp parallel for shared(partial)
                for (size_t j = 0; j < players.size(); ++j) {
                    auto idx = players[j]->selectAction(i);
                    partial += actions[idx];
                }
                partial = std::round(partial / nb_players);
                total += partial * nb_players;

//#pragma omp parallel for shared(partial)
                for (size_t j = 0; j < players.size(); ++j) {
                    players[j]->decrease(partial);
                }
            }
            return std::make_pair(total, final_round);
        }

        bool reinforcePath(std::vector<A> &players) {
//#pragma omp parallel
            for (size_t j = 0; j < players.size(); ++j) {
                players[j].reinforceTrajectory();
            }
            return true;
        }

        bool reinforcePath(std::vector<A> &players, size_t final_round) {
//#pragma omp parallel
            for (size_t j = 0; j < players.size(); ++j) {
                players[j].reinforceTrajectory(final_round);
            }
            return true;
        }

        bool reinforcePath(std::vector<A *> &players) {
//#pragma omp parallel
            for (size_t j = 0; j < players.size(); ++j) {
                players[j]->reinforceTrajectory();
            }
            return true;
        }

        bool reinforcePath(std::vector<A *> &players, size_t final_round) {
//#pragma omp parallel
            for (size_t j = 0; j < players.size(); ++j) {
                players[j]->reinforceTrajectory(final_round);
            }
            return true;
        }

        bool printGroup(std::vector<A> &players) {
            for (auto &player : players) {
                std::cout << player << std::endl;
            }
            return true;
        }

        bool printGroup(std::vector<A *> &players) {
            for (auto &player : players) {
                std::cout << *player << std::endl;
            }
            return true;
        }

        bool calcProbabilities(std::vector<A> &players) {
//#pragma omp parallel
            for (size_t j = 0; j < players.size(); ++j) {
                players[j].inferPolicy();
            }
            return true;
        }

        bool calcProbabilities(std::vector<A *> &players) {
//#pragma omp parallel
            for (size_t j = 0; j < players.size(); ++j) {
                players[j]->inferPolicy();
            }
            return true;
        }

        bool resetEpisode(std::vector<A> &players) {
            for (auto player : players) {
                player.resetTrajectory();
            }
            return true;
        }

        bool resetEpisode(std::vector<A *> &players) {
            for (auto player : players) {
                player->resetTrajectory();
            }
            return true;
        }

        double playersPayoff(std::vector<A> &players) {
            double total = 0;
            for (auto &player : players) {
                total += double(player.payoff());
            }
            return total;
        }

        double playersPayoff(std::vector<A *> &players) {
            double total = 0;
            for (auto &player : players) {
                total += player->payoff();
            }
            return total;
        }

        void setPayoffs(std::vector<A> &players, unsigned int value) {
            for (auto &player: players) {
                player.set_payoff(value);
            }
        }

        void setPayoffs(std::vector<A *> &players, unsigned int value) {
            for (auto &player: players) {
                player->set_payoff(value);
            }
        }

    private:

        // Random generators
        std::mt19937_64 _mt{EGTTools::Random::SeedGenerator::getInstance().getSeed()};
    };

    template<typename A>
    class CRDDemocracy<A, void> {

    public:
        /**
         * @brief Model of the Collective-Risk dillemma game.
         *
         * This game constitutes an MDP.
         *
         * This function plays the game for a number of rounds
         *
         * @param players
         * @param actions
         * @param rounds
         * @return std::tuple (donations, rounds)
         */
        std::pair<double, size_t>
        playGame(std::vector<A> &players, std::vector<size_t> &actions, size_t rounds) {
            double total = 0.0, partial;
            size_t nb_players = players.size();
            for (auto &player : players) {
                player.resetPayoff();
            }
            for (size_t i = 0; i < rounds; ++i) {
                partial = 0.0;
//#pragma omp parallel for shared(partial)
                for (size_t j = 0; j < players.size(); ++j) {
                    auto idx = players[j].selectAction(i);
                    partial += actions[idx];
                }
                partial = std::round(partial / nb_players);
                total += partial * nb_players;

//#pragma omp parallel for shared(partial)
                for (size_t j = 0; j < players.size(); ++j) {
                    players[j].decrease(partial);
                }
            }
            return std::make_pair(total, rounds);
        }

        std::pair<double, size_t>
        playGame(std::vector<A *> &players, std::vector<size_t> &actions, size_t rounds) {
            double total = 0.0, partial;
            size_t nb_players = players.size();
            for (auto &player : players) {
                player->resetPayoff();
            }
            for (size_t i = 0; i < rounds; ++i) {
                partial = 0.0;
//#pragma omp parallel for shared(partial)
                for (size_t j = 0; j < players.size(); ++j) {
                    auto idx = players[j]->selectAction(i);
                    partial += actions[idx];
                }
                partial = std::round(partial / nb_players);
                total += partial * nb_players;

//#pragma omp parallel for shared(partial)
                for (size_t j = 0; j < players.size(); ++j) {
                    players[j]->decrease(partial);
                }
            }
            return std::make_pair(total, rounds);
        }

        bool reinforcePath(std::vector<A> &players) {
            for (auto &player : players) {
                player.reinforceTrajectory();
            }
            return true;
        }

        bool reinforcePath(std::vector<A *> &players) {
            for (auto &player : players) {
                player->reinforceTrajectory();
            }
            return true;
        }

        bool printGroup(std::vector<A> &players) {
            for (auto &player : players) {
                std::cout << player << std::endl;
            }
            return true;
        }

        bool printGroup(std::vector<A *> &players) {
            for (auto &player : players) {
                std::cout << *player << std::endl;
            }
            return true;
        }

        bool calcProbabilities(std::vector<A> &players) {
            for (auto &player : players) {
                player.inferPolicy();
            }
            return true;
        }

        bool calcProbabilities(std::vector<A *> &players) {
            for (auto &player : players) {
                player->inferPolicy();
            }
            return true;
        }

        bool resetEpisode(std::vector<A> &players) {
            for (auto player : players) {
                player.resetTrajectory();
            }
            return true;
        }

        bool resetEpisode(std::vector<A *> &players) {
            for (auto player : players) {
                player->resetTrajectory();
            }
            return true;
        }

        double playersPayoff(std::vector<A> &players) {
            double total = 0;
            for (auto &player : players) {
                total += double(player.payoff());
            }
            return total;
        }

        double playersPayoff(std::vector<A *> &players) {
            double total = 0;
            for (auto &player : players) {
                total += player->payoff();
            }
            return total;
        }

        void setPayoffs(std::vector<A> &players, unsigned int value) {
            for (auto &player: players) {
                player.set_payoff(value);
            }
        }

        void setPayoffs(std::vector<A *> &players, unsigned int value) {
            for (auto &player: players) {
                player->set_payoff(value);
            }
        }

    private:

        // Random generators
        std::mt19937_64 _mt{EGTTools::Random::SeedGenerator::getInstance().getSeed()};
    };

    template <>
    class CRDDemocracy<PopContainer, void> {

    public:
        /**
         * @brief Model of the Collective-Risk dillemma game.
         *
         * This game constitutes an MDP.
         *
         * This function plays the game for a number of rounds
         *
         * @param players
         * @param actions
         * @param rounds
         * @return std::tuple (donations, rounds)
         */
        std::pair<double, size_t>
        playGame(PopContainer &players, EGTTools::RL::ActionSpace &actions, size_t rounds) {
            double total = 0.0, partial = 0.0;
            size_t nb_players = players.size();
            for (auto &player : players) {
                player->resetPayoff();
            }
            // Main game loop
            for (size_t i = 0; i < rounds; ++i) {
                partial = 0.0;
                // Gather votes
                for (auto & player : players) {
                    auto idx = player->selectAction(i);
                    partial += actions[idx];
                }
                // Decide by average
                // (Other options: majority vote, etc.)
                partial = std::round(partial / nb_players);
                total += partial * nb_players;
                // Force players to invest
                // according to the results of the voting
                for (auto & player : players) {
                    player->decrease(partial);
                }
            }
            return std::make_pair(total, rounds);
        }

        bool reinforcePath(PopContainer &players) {
            for (auto& player : players)
                player->reinforceTrajectory();
            return true;
        }

        bool printGroup(PopContainer &players) {
            for (auto &player : players) {
                std::cout << *player << std::endl;
            }
            return true;
        }

        bool calcProbabilities(PopContainer &players) {
            for (auto& player : players)
                player->inferPolicy();
            return true;
        }

        bool resetEpisode(PopContainer &players) {
            for (auto &player : players) {
                player->resetTrajectory();
            }
            return true;
        }

        double playersPayoff(PopContainer &players) {
            double total = 0;
            for (auto& player : players)
                total += player->payoff();

            return total;
        }

        void setPayoffs(PopContainer &players, unsigned int value) {
            for (auto &player: players) {
                player->set_payoff(value);
            }
        }

        double playersContribution(PopContainer &players) {
            double total = 0;
            for (auto& player : players)
                total += player->endowment() - player->payoff();

            return total;
        }

    private:

        // Random generators
        std::mt19937_64 _mt{EGTTools::Random::SeedGenerator::getInstance().getSeed()};
    };


}


#endif //DYRWIN_RL_CRDDemocracy_H
