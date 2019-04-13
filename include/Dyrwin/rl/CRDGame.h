//
// Created by Elias Fernandez on 2019-03-11.
//

#ifndef DYRWIN_CRDGAME_H
#define DYRWIN_CRDGAME_H

#include <random>
#include <vector>
#include <cmath>
#include <tuple>
#include <Eigen/Dense>
#include <iostream>
#include "../SeedGenerator.h"
#include "Agent.h"
#include "RLUtils.h"


namespace egt_tools {
/**
 * Implements the Collective-risk dilemma defined in Milinski et. al 2008.
 */
    class CRDGame {
    public:
        CRDGame() = default;

        virtual std::tuple<float, unsigned>
        playGame(std::vector<Agent> &players, std::vector<unsigned>& actions, unsigned rounds) {
            float total = 0.0;
            for (auto &player : players) {
                player.resetPayoff();
            }
            for (unsigned i = 0; i < rounds; i++) {
                for (auto a : players) {
                    unsigned indx = a.selectAction(i);
                    a.decrease(actions[indx]);
                    total += actions[indx];
                }
            }
            return std::make_tuple(total, rounds);
        }

        bool reinforcePath(std::vector<Agent> &players) {
            for (auto &player : players) {
                player.reinforceTrajectory();
            }
            return true;
        }

        bool printGroup(std::vector<Agent> &players) {
            for (auto &player : players) {
                std::cout << player << std::endl;
            }
            return true;
        }

        bool calcProbabilities(std::vector<Agent> &players) {
            for (auto &player : players) {
                player.inferPolicy();
            }
            return true;
        }

        bool resetCounts(std::vector<Agent> &players) {
            for (auto player : players) {
                player.resetTrajectory();
            }
            return true;
        }

        float playersPayoff(std::vector<Agent> &players) {
            float total = 0;
            for (auto &player : players) {
                total += float(player.payoff());
            }
            return total;
        }

        void setPayoffs(std::vector<Agent> &players, unsigned value) {
            for (auto &player: players) {
                player.set_payoff(value);
            }
        }

    private:

        // Random generators
        std::mt19937_64 _mt{SeedGenerator::getInstance().getSeed()};
    };
}


#endif //DYRWIN_CRDGAME_H
