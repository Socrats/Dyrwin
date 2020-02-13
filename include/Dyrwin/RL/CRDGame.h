//
// Created by Elias Fernandez on 2019-03-11.
//

#ifndef DYRWIN_RL_CRDGAME_H
#define DYRWIN_RL_CRDGAME_H

#include <random>
#include <vector>
#include <cmath>
#include <tuple>
#include <iostream>
#include <Dyrwin/SeedGenerator.h>
#include <Dyrwin/RL/Agent.h>
#include <Dyrwin/RL/TimingUncertainty.hpp>
#include <Dyrwin/RL/PopContainer.hpp>
#include <Dyrwin/RL/Utils.h>

namespace EGTTools::RL {
/**
 * @brief Implements the Collective-risk dilemma defined in Milinski et. al 2008.
 *
 * @tparam A. Container for the agents.
 */
template<typename A = Agent, typename R = void>
class CRDGame {

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
  playGame(std::vector<A> &players, EGTTools::RL::ActionSpace &actions, size_t rounds, R &gen_round) {

    auto final_round = gen_round.calculateEnd(rounds, _mt);

    double total = 0.0;
    for (auto &player : players) {
      player.resetPayoff();
    }
    for (size_t i = 0; i < final_round; ++i) {
//#pragma omp parallel for shared(total)
      for (size_t j = 0; j < players.size(); ++j) {
        unsigned idx = players[j].selectAction(i);
        players[j].decrease(actions[idx]);
        total += actions[idx];
      }
    }
    return std::make_pair(total, final_round);
  }

  std::pair<double, size_t>
  playGame(std::vector<std::unique_ptr<A>> &players, EGTTools::RL::ActionSpace &actions, size_t rounds, R &gen_round) {

    auto final_round = gen_round.calculateEnd(rounds, _mt);

    double total = 0.0;
    for (auto &player : players) {
      player->resetPayoff();
    }
    for (size_t i = 0; i < final_round; ++i) {
//#pragma omp parallel for shared(total)
      for (size_t j = 0; j < players.size(); ++j) {
        unsigned idx = players[j]->selectAction(i);
        players[j]->decrease(actions[idx]);
        total += actions[idx];
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

  bool reinforcePath(std::vector<std::unique_ptr<A>> &players) {
//#pragma omp parallel
    for (size_t j = 0; j < players.size(); ++j) {
      players[j]->reinforceTrajectory();
    }
    return true;
  }

  bool reinforcePath(std::vector<std::unique_ptr<A>> &players, size_t final_round) {
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

  bool printGroup(std::vector<std::unique_ptr<A>> &players) {
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

  bool calcProbabilities(std::vector<std::unique_ptr<A>> &players) {
//#pragma omp parallel
    for (size_t j = 0; j < players.size(); ++j) {
      players[j]->inferPolicy();
    }
    return true;
  }

  bool resetEpisode(std::vector<A> &players) {
    for (auto &player : players) {
      player.resetTrajectory();
    }
    return true;
  }

  bool resetEpisode(std::vector<std::unique_ptr<A>> &players) {
    for (auto &player : players) {
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

  double playersPayoff(std::vector<std::unique_ptr<A>> &players) {
    double total = 0;
    for (auto &player : players) {
      total += player->payoff();
    }
    return total;
  }

  double playersContribution(std::vector<A> &players) {
    double total = 0;
    for (auto &player : players) {
      total += player.endowment() - player.payoff();
    }
    return total;
  }

  double playersContribution(std::vector<std::unique_ptr<A>> &players) {
    double total = 0;
    for (auto &player : players) {
      total += player->endowment() - player->payoff();
    }
    return total;
  }

  void setPayoffs(std::vector<A> &players, unsigned int value) {
    for (auto &player: players) {
      player.set_payoff(value);
    }
  }

  void setPayoffs(std::vector<std::unique_ptr<A>> &players, unsigned int value) {
    for (auto &player: players) {
      player->set_payoff(value);
    }
  }

 private:

  // Random generators
  std::mt19937_64 _mt{EGTTools::Random::SeedGenerator::getInstance().getSeed()};
};

template<typename A>
class CRDGame<A, void> {

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
  playGame(std::vector<A> &players, EGTTools::RL::ActionSpace &actions, size_t rounds) {
    double total = 0.0;
    for (auto &player : players) {
      player.resetPayoff();
    }
    for (size_t i = 0; i < rounds; i++) {
      for (auto a : players) {
        unsigned idx = a.selectAction(i);
        a.decrease(actions[idx]);
        total += actions[idx];
      }
    }
    return std::make_pair(total, rounds);
  }

  std::pair<double, size_t>
  playGame(std::vector<std::unique_ptr<A>> &players, EGTTools::RL::ActionSpace &actions, size_t rounds) {
    double total = 0.0;
    for (auto &player : players) {
      player->resetPayoff();
    }
    for (size_t i = 0; i < rounds; i++) {
      for (auto &a : players) {
        unsigned idx = a->selectAction(i);
        a->decrease(actions[idx]);
        total += actions[idx];
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

  bool reinforcePath(std::vector<std::unique_ptr<A>> &players) {
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

  bool printGroup(std::vector<std::unique_ptr<A>> &players) {
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

  bool calcProbabilities(std::vector<std::unique_ptr<A>> &players) {
    for (auto &player : players) {
      player->inferPolicy();
    }
    return true;
  }

  bool resetEpisode(std::vector<A> &players) {
    for (auto &player : players) {
      player.resetTrajectory();
    }
    return true;
  }

  bool resetEpisode(std::vector<std::unique_ptr<A>> &players) {
    for (auto &player : players) {
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

  double playersPayoff(std::vector<std::unique_ptr<A>> &players) {
    double total = 0;
    for (auto &player : players) {
      total += player->payoff();
    }
    return total;
  }

  double playersContribution(std::vector<A> &players) {
    double total = 0;
    for (auto &player : players) {
      total += player.endowment() - player.payoff();
    }
    return total;
  }

  double playersContribution(std::vector<std::unique_ptr<A>> &players) {
    double total = 0;
    for (auto &player : players) {
      total += player->endowment() - player->payoff();
    }
    return total;
  }

  void setPayoffs(std::vector<A> &players, unsigned int value) {
    for (auto &player: players) {
      player.set_payoff(value);
    }
  }

  void setPayoffs(std::vector<std::unique_ptr<A>> &players, unsigned int value) {
    for (auto &player: players) {
      player->set_payoff(value);
    }
  }

 private:

  // Random generators
  std::mt19937_64 _mt{EGTTools::Random::SeedGenerator::getInstance().getSeed()};
};

template<typename R>
class CRDGame<PopContainer, R> {

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
   * @param min_rounds
   * @return std::tuple (donations, rounds)
   */
  std::pair<double, size_t>
  playGame(PopContainer &players, EGTTools::RL::ActionSpace &actions, size_t min_rounds, R &gen_round) {
    auto final_round = gen_round.calculateEnd(min_rounds, _mt);

    double total = 0.0;
    size_t idx;
    for (auto &player : players) {
      player->resetPayoff();
    }
    for (size_t i = 0; i < final_round; i++) {
      for (auto &player : players) {
        idx = player->selectAction(i);
        // In case nothing is left of the player's endowment, then donate 0
        if (!player->decrease(actions[idx])) {
          // Select the next best action
          if (idx > 1) {
            for (size_t n = 0; n < idx; ++n) {
              if (player->decrease(actions[idx - n - 1])) {
                idx = idx - n - 1;
                break;
              }
            }
          }
          player->set_trajectory_round(i, idx);
        }
//                    player->decrease(actions[idx]);
        total += actions[idx];
      }
    }
    return std::make_pair(total, final_round);
  }

  bool reinforcePath(PopContainer &players) {
    for (auto &player : players)
      player->reinforceTrajectory();
    return true;
  }

  bool reinforcePath(PopContainer &players, size_t final_round) {
    for (auto &player : players)
      player->reinforceTrajectory(final_round);
    return true;
  }

  bool printGroup(PopContainer &players) {
    for (auto &player : players) {
      std::cout << *player << std::endl;
    }
    return true;
  }

  bool calcProbabilities(PopContainer &players) {
    for (auto &player : players)
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
    for (auto &player : players)
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
    for (auto &player : players)
      total += player->endowment() - player->payoff();

    return total;
  }

 private:

  // Random generators
  std::mt19937_64 _mt{EGTTools::Random::SeedGenerator::getInstance().getSeed()};
};

template<>
class CRDGame<PopContainer, void> {

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
    double total = 0.0;
    size_t idx;
    for (auto &player : players) {
      player->resetPayoff();
    }
    for (size_t i = 0; i < rounds; i++) {
      for (auto &player : players) {
        idx = player->selectAction(i);
        player->decrease(actions[idx]);
        total += actions[idx];
      }
    }
    return std::make_pair(total, rounds);
  }

  /**
   * @brief stores the data of each round
   *
   * This method must return:
   * a) actions per round of each player
   *
   * @param players : PopContainer with the agents that will play the game
   * @param actions : available actions per round
   * @param rounds : number of rounds
   */
  void playGameVerbose(PopContainer &players, EGTTools::RL::ActionSpace &actions, size_t rounds, Matrix2D &results) {
    size_t action_idx;
    for (auto &player : players) {
      player->resetPayoff();
    }
    for (size_t i = 0; i < rounds; i++) {
      for (size_t j = 0; j < players.size(); ++j) {
        action_idx = players[j].selectAction(i);
        players[j].decrease(actions[action_idx]);
        // now we store this data on results
        results(j, i) = action_idx;
      }
    }
  }

  bool reinforcePath(PopContainer &players) {
    for (auto &player : players)
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
    for (auto &player : players)
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
    for (auto &player : players)
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
    for (auto &player : players)
      total += player->endowment() - player->payoff();

    return total;
  }

 private:

  // Random generators
  std::mt19937_64 _mt{EGTTools::Random::SeedGenerator::getInstance().getSeed()};
};

}

#endif //DYRWIN_RL_CRDGAME_H
