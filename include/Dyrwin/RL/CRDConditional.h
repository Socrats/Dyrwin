#include <utility>

#include <utility>

#include <utility>

//
// Created by Elias Fernandez on 2019-03-11.
//

#ifndef DYRWIN_RL_CRDCONDITIONAL_H
#define DYRWIN_RL_CRDCONDITIONAL_H

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
    class CRDConditional {

    public:
        explicit CRDConditional(FlattenState  flatten) : _flatten(std::move(flatten)) {
            _state = EGTTools::RL::Factors(2);
        }

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
        std::tuple<double, unsigned>
        playGame(std::vector<A> &players, std::vector<size_t> &actions, size_t rounds, R &gen_round) {

            auto final_round = gen_round.calculateEnd(rounds, _mt);

            double total = 0.0, partial = 0.0;
            for (auto &player : players) {
                player.resetPayoff();
            }
            for (size_t i = 0; i < final_round; ++i) {
                _state[0] = i, _state[1] = static_cast<size_t>(partial);
                partial = 0.0;
#pragma omp parallel for shared(total)
                for (size_t j = 0; j < players.size(); ++j) {
                    unsigned idx = players[j].selectAction(_flatten.toIndex(_state));
                    players[j].decrease(actions[idx]);
                    partial += actions[idx];
                }
                total += partial;
            }
            return std::make_tuple(total, final_round);
        }

        std::tuple<double, unsigned>
        playGame(std::vector<A *> &players, std::vector<size_t> &actions, size_t rounds, R &gen_round) {

            auto final_round = gen_round.calculateEnd(rounds, _mt);

            double total = 0.0, partial = 0.0;
            for (auto &player : players) {
                player->resetPayoff();
            }
            for (size_t i = 0; i < final_round; i++) {
                _state[0] = i, _state[1] = static_cast<size_t>(partial);
                partial = 0.0;
#pragma omp parallel for shared(total)
                for (size_t j = 0; j < players.size(); ++j) {
                    unsigned idx = players[j]->selectAction(_flatten.toIndex(_state));
                    players[j]->decrease(actions[idx]);
                    partial += actions[idx];
                }
                total += partial;
            }
            return std::make_tuple(total, final_round);
        }

        bool reinforcePath(std::vector<A> &players) {
#pragma omp parallel
            for (size_t j = 0; j < players.size(); ++j) {
                players[j].reinforceTrajectory();
            }
            return true;
        }

        bool reinforcePath(std::vector<A *> &players) {
#pragma omp parallel
            for (size_t j = 0; j < players.size(); ++j) {
                players[j]->reinforceTrajectory();
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
#pragma omp parallel
            for (size_t j = 0; j < players.size(); ++j) {
                players[j].inferPolicy();
            }
            return true;
        }

        bool calcProbabilities(std::vector<A *> &players) {
#pragma omp parallel
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

        FlattenState& flatten() { return _flatten; }

    private:
        FlattenState _flatten;
        EGTTools::RL::Factors _state;
        // Random generators
        std::mt19937_64 _mt{EGTTools::Random::SeedGenerator::getInstance().getSeed()};
    };

    template<typename A>
    class CRDConditional<A, void> {

    public:
        explicit CRDConditional(FlattenState  flatten) : _flatten(std::move(flatten)) {
            _state = EGTTools::RL::Factors(2);
        }

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
        std::tuple<double, unsigned>
        playGame(std::vector<A> &players, std::vector<size_t> &actions, size_t rounds) {
            double total = 0.0, partial = 0.0;
            for (auto &player : players) {
                player.resetPayoff();
            }
            for (unsigned i = 0; i < rounds; ++i) {
                _state[0] = i, _state[1] = static_cast<size_t>(partial);
                partial = 0.0;
                for (auto a : players) {
                    unsigned idx = a.selectAction(_flatten.toIndex(_state));
                    a.decrease(actions[idx]);
                    partial += actions[idx];
                }
                total += partial;
            }
            return std::make_tuple(total, rounds);
        }

        std::tuple<double, unsigned>
        playGame(std::vector<A *> &players, std::vector<size_t> &actions, size_t rounds) {
            double total = 0.0, partial = 0.0;
            for (auto &player : players) {
                player->resetPayoff();
            }
            for (size_t i = 0; i < rounds; ++i) {
                _state[0] = i, _state[1] = static_cast<size_t>(partial);
                partial = 0.0;
                for (auto &a : players) {
                    unsigned idx = a->selectAction(_flatten.toIndex(_state));
                    a->decrease(actions[idx]);
                    partial += actions[idx];
                }
                total += partial;
            }
            return std::make_tuple(total, rounds);
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

        FlattenState& flatten() { return _flatten; }

    private:
        FlattenState _flatten;
        EGTTools::RL::Factors _state;
        // Random generators
        std::mt19937_64 _mt{EGTTools::Random::SeedGenerator::getInstance().getSeed()};
    };


}


#endif //DYRWIN_RL_CRDGAME_H