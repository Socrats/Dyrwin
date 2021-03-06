//
// Created by Elias Fernandez on 2019-03-11.
//

#ifndef DYRWIN_RL_CRDGAME_H
#define DYRWIN_RL_CRDGAME_H

#include <cmath>
#include <random>
#include <vector>
#include <tuple>
#include <iostream>
#include <Dyrwin/SeedGenerator.h>
//#include <Dyrwin/RL/Agent.h>
#include <Dyrwin/RL/TimingUncertainty.hpp>
#include <Dyrwin/RL/PopContainer.hpp>
#include <Dyrwin/RL/Utils.h>

namespace EGTTools::RL {
/**
 * @brief Implements the Collective-risk dilemma defined in Milinski et. al 2008.
 *
 * @tparam A. Container for the agents.
 */
template<typename A, typename R, typename G>
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
class CRDGame<A, void, void> {

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

template<typename R, typename G>
class CRDGame<PopContainer, R, G> {

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
  playGame(PopContainer &players, EGTTools::RL::ActionSpace &actions, size_t min_rounds, R &gen_round, G &generator) {
    auto final_round = gen_round.calculateEnd(min_rounds, generator);

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
//          if (idx > 0) {
//            for (size_t n = 0; n < idx; ++n) {
//              if (player->decrease(actions[idx - n - 1])) {
//                idx = idx - n - 1;
//                break;
//              }
//            }
//          }
          // play zero in this case
          idx = 0;
          player->set_trajectory_round(i, idx);
        }
        total += actions[idx];
      }
    }
    return std::make_pair(total, final_round);
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
  std::pair<double, size_t>
  playGameVerbose(PopContainer &players,
                  EGTTools::RL::ActionSpace &actions,
                  size_t min_rounds,
                  R &gen_round,
                  G &generator,
                  Matrix2D &results) {
    auto final_round = gen_round.calculateEnd(min_rounds, generator);

    size_t action_idx;
    double total = 0.0;
    for (auto &player : players) {
      player->resetPayoff();
    }
    for (size_t i = 0; i < final_round; i++) {
      for (size_t j = 0; j < players.size(); ++j) {
        action_idx = players(j)->selectAction(i);
        // In case nothing is left of the player's endowment, then donate 0
        if (!players(j)->decrease(actions[action_idx])) {
          // Select the next best action
//          if (action_idx > 0) {
//            for (size_t n = 0; n < action_idx; ++n) {
//              if (players(j)->decrease(actions[action_idx - n - 1])) {
//                action_idx = action_idx - n - 1;
//                break;
//              }
//            }
//          }
          // select 0 in this case
          action_idx = 0;
          players(j)->set_trajectory_round(i, action_idx);
        }
        results(j, i) = actions[action_idx];
        total += actions[action_idx];
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

  void updatePayoffs(PopContainer &players, double value) {
    for (auto &player: players) {
      player->multiply_by_payoff(value);
    }
  }

  void subtractEndowment(PopContainer &players) {
    for (auto &player: players) {
      player->subtract_endowment_to_payoff();
    }
  }

  double playersContribution(PopContainer &players) {
    double total = 0;
    for (auto &player : players)
      total += player->endowment() - player->payoff();

    return total;
  }
};

template<>
class CRDGame<PopContainer, void, void> {

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
        if (!player->decrease(actions[idx])) {
          // Select the next best action
//          if (idx > 0) {
//            for (size_t n = 0; n < idx; ++n) {
//              if (player->decrease(actions[idx - n - 1])) {
//                idx = idx - n - 1;
//                break;
//              }
//            }
//          }
          // select 0 in this case
          idx = 0;
          player->set_trajectory_round(i, idx);
        }
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
  double playGameVerbose(PopContainer &players, EGTTools::RL::ActionSpace &actions, size_t rounds, Matrix2D &results) {
    size_t action_idx;
    double total = 0.0;
    for (auto &player : players) {
      player->resetPayoff();
    }
    for (size_t i = 0; i < rounds; i++) {
      for (size_t j = 0; j < players.size(); ++j) {
        action_idx = players(j)->selectAction(i);
        // In case nothing is left of the player's endowment, then donate 0
        if (!players(j)->decrease(actions[action_idx])) {
          // Select the next best action
//          if (action_idx > 0) {
//            for (size_t n = 0; n < action_idx; ++n) {
//              if (players(j)->decrease(actions[action_idx - n - 1])) {
//                action_idx = action_idx - n - 1;
//                break;
//              }
//            }
//          }
          // select 0 in this case
          action_idx = 0;
          players(j)->set_trajectory_round(i, action_idx);
        }
        // now we store this data on results
        results(j, i) = actions[action_idx];
        total += actions[action_idx];
      }
    }
    return total;
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

  void setPayoffs(PopContainer &players, double value) {
    for (auto &player: players) {
      player->set_payoff(value);
    }
  }

  void updatePayoffs(PopContainer &players, double value) {
    for (auto &player: players) {
      player->multiply_by_payoff(value);
    }
  }

  void subtractEndowment(PopContainer &players) {
    for (auto &player: players) {
      player->subtract_endowment_to_payoff();
    }
  }

  double playersContribution(PopContainer &players) {
    double total = 0;
    for (auto &player : players)
      total += player->endowment() - player->payoff();

    return total;
  }
};

}

#endif //DYRWIN_RL_CRDGAME_H
