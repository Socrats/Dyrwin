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
template<typename A = Agent, typename R = EGTTools::TimingUncertainty<std::mt19937_64>, typename G = std::mt19937_64>
class CRDConditional {

 public:
  explicit CRDConditional(FlattenState flatten) : _flatten(std::move(flatten)) {
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
   * @return std::pair (donations, rounds)
   */
  std::pair<double, unsigned>
  playGame(std::vector<A> &players, std::vector<size_t> &actions, size_t rounds, R &gen_round) {

    auto final_round = gen_round.calculateEnd(rounds, _mt);

    double total = 0.0, partial = 0.0;
    for (auto &player : players) {
      player.resetPayoff();
    }
    for (size_t i = 0; i < final_round; ++i) {
      _state[0] = i, _state[1] = static_cast<size_t>(partial);
      partial = 0.0;
//#pragma omp parallel for shared(total)
      for (size_t j = 0; j < players.size(); ++j) {
        unsigned idx = players[j].selectAction(i, _flatten.toIndex(_state));
        players[j].decrease(actions[idx]);
        partial += actions[idx];
      }
      total += partial;
    }
    return std::make_pair(total, final_round);
  }

  std::pair<double, unsigned>
  playGame(std::vector<A *> &players, std::vector<size_t> &actions, size_t rounds, R &gen_round) {

    auto final_round = gen_round.calculateEnd(rounds, _mt);

    double total = 0.0, partial = 0.0;
    for (auto &player : players) {
      player->resetPayoff();
    }
    for (size_t i = 0; i < final_round; i++) {
      _state[0] = i, _state[1] = static_cast<size_t>(partial);
      partial = 0.0;
//#pragma omp parallel for shared(total)
      for (size_t j = 0; j < players.size(); ++j) {
        unsigned idx = players[j]->selectAction(i, _flatten.toIndex(_state));
        players[j]->decrease(actions[idx]);
        partial += actions[idx];
      }
      total += partial;
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

  FlattenState &flatten() { return _flatten; }

 private:
  FlattenState _flatten;
  EGTTools::RL::Factors _state;
  // Random generators
  std::mt19937_64 _mt{EGTTools::Random::SeedGenerator::getInstance().getSeed()};
};

template<typename A>
class CRDConditional<A, void, void> {

 public:
  explicit CRDConditional(FlattenState flatten) : _flatten(std::move(flatten)) {
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
   * @return std::pair (donations, rounds)
   */
  std::pair<double, unsigned>
  playGame(std::vector<A> &players, std::vector<size_t> &actions, size_t rounds) {
    double total = 0.0, partial = 0.0;
    for (auto &player : players) {
      player.resetPayoff();
    }
    for (size_t i = 0; i < rounds; ++i) {
      _state[0] = i, _state[1] = static_cast<size_t>(partial);
      partial = 0.0;
      for (auto a : players) {
        unsigned idx = a.selectAction(i, _flatten.toIndex(_state));
        a.decrease(actions[idx]);
        partial += actions[idx];
      }
      total += partial;
    }
    return std::make_pair(total, rounds);
  }

  std::pair<double, unsigned>
  playGame(std::vector<A *> &players, std::vector<size_t> &actions, size_t rounds) {
    double total = 0.0, partial = 0.0;
    for (auto &player : players) {
      player->resetPayoff();
    }
    for (size_t i = 0; i < rounds; ++i) {
      _state[0] = i, _state[1] = static_cast<size_t>(partial);
      partial = 0.0;
      for (auto &a : players) {
        unsigned idx = a->selectAction(i, _flatten.toIndex(_state));
        a->decrease(actions[idx]);
        partial += actions[idx];
      }
      total += partial;
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

  FlattenState &flatten() { return _flatten; }

 private:
  FlattenState _flatten;
  EGTTools::RL::Factors _state;
  // Random generators
  std::mt19937_64 _mt{EGTTools::Random::SeedGenerator::getInstance().getSeed()};
};

template<typename R, typename G>
class CRDConditional<PopContainer, R, G> {

 public:
  explicit CRDConditional(FlattenState flatten) : _flatten(std::move(flatten)) {}

  /**
   * @brief Model of the Collective-Risk dillemma game.
   *
   * This game constitutes an MDP.
   *
   * This function plays the game for a number of rounds
   *
   * Due to how conditional players are implemented, actions must be integers.
   *
   * Each conditional player has a state composed of a tuple (round, sum of the previous round contributions)
   *
   * @param players
   * @param actions
   * @param min_rounds
   * @return std::tuple (donations, rounds)
   */
  std::pair<double, size_t>
  playGame(PopContainer &players, EGTTools::RL::ActionSpace &actions, size_t min_rounds, R &gen_round, G &generator) {
    auto final_round = gen_round.calculateEnd(min_rounds, generator);

    int total = 0, partial = 0;
    size_t action, idx;
    std::vector<size_t> state(2, 0); // creates a vector of size 2 with all members equal to 0
    for (auto &player : players) {
      player->resetPayoff();
    }
    for (size_t i = 0; i < final_round; i++) {
      state[0] = i, state[1] = static_cast<size_t>(partial);
      idx = _flatten.toIndex(state);
      partial = 0.0;
      for (auto &player : players) {
        action = player->selectAction(i, idx);
        if (!player->decrease(actions[action])) {
          // Select the next best action
          if (action > 0) {
            for (size_t n = 0; n < action; ++n) {
              if (player->decrease(actions[action - n - 1])) {
                action = action - n - 1;
                break;
              }
            }
          }
          player->set_trajectory_state(i, idx, action);
        }
        partial += actions[action];
      }
      total += partial;
    }
    return std::make_pair(static_cast<double>(total), final_round);
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
  std::pair<int, size_t>
  playGameVerbose(PopContainer &players,
                  EGTTools::RL::ActionSpace &actions,
                  size_t min_rounds,
                  R &gen_round,
                  G &generator,
                  Matrix2D &results) {
    auto final_round = gen_round.calculateEnd(min_rounds, generator);

    size_t action_idx, state_idx;
    std::vector<size_t> state(2, 0); // creates a vector of size 2 with all members equal to 0
    int partial = 0;
    int total = 0;
    for (auto &player : players) {
      player->resetPayoff();
    }
    for (size_t i = 0; i < final_round; i++) {
      state[0] = i, state[1] = static_cast<size_t>(partial);
      state_idx = _flatten.toIndex(state);
      partial = 0.0;
      for (size_t j = 0; j < players.size(); ++j) {
        action_idx = players(j)->selectAction(i, state_idx);
        if (!players(j)->decrease(actions[action_idx])) {
          // Select the next best action
          if (action_idx > 0) {
            for (size_t n = 0; n < action_idx; ++n) {
              if (players(j)->decrease(actions[action_idx - n - 1])) {
                action_idx = action_idx - n - 1;
                break;
              }
            }
          }
          players(j)->set_trajectory_state(i, state_idx, action_idx);
        }
        partial += actions[action_idx];
        // now we store this data on results
        results(j, i) = actions[action_idx];
      }
      total += partial;
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

  FlattenState &flatten() { return _flatten; }

 private:
  FlattenState _flatten;
};

template<>
class CRDConditional<PopContainer, void, void> {

 public:

  explicit CRDConditional(FlattenState flatten) : _flatten(std::move(flatten)) {}

  /**
   * @brief Model of the Collective-Risk dilemma game.
   *
   * This game constitutes an MDP.
   *
   * This function plays the game for a number of rounds
   *
   * @param players
   * @param actions
   * @param rounds
   * @return std::pair (donations, rounds)
   */
  std::pair<double, size_t>
  playGame(PopContainer &players, EGTTools::RL::ActionSpace &actions, size_t rounds) {
    size_t action_idx, state_idx;
    int total = 0, partial = 0;
    std::vector<size_t> state(2, 0); // creates a vector of size 2 with all members equal to 0
    for (auto &player : players) {
      player->resetPayoff();
    }
    for (size_t i = 0; i < rounds; ++i) {
      state[0] = i, state[1] = static_cast<size_t>(partial);
      state_idx = _flatten.toIndex(state);
      partial = 0.0;
      for (auto &player : players) {
        action_idx = player->selectAction(i, state_idx);
        if (!player->decrease(actions[action_idx])) {
          // Select the next best action
          if (action_idx > 0) {
            for (size_t n = 0; n < action_idx; ++n) {
              if (player->decrease(actions[action_idx - n - 1])) {
                action_idx = action_idx - n - 1;
                break;
              }
            }
          }
          player->set_trajectory_state(i, state_idx, action_idx);
        }
        partial += actions[action_idx];
      }
      total += partial;
    }
    return std::make_pair(static_cast<double>(total), rounds);
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
  int playGameVerbose(PopContainer &players, EGTTools::RL::ActionSpace &actions, size_t rounds, Matrix2D &results) {
    size_t action_idx, state_idx;
    std::vector<size_t> state(2, 0); // creates a vector of size 2 with all members equal to 0
    int partial = 0;
    int total = 0;
    for (auto &player : players) {
      player->resetPayoff();
    }
    for (size_t i = 0; i < rounds; i++) {
      state[0] = i, state[1] = static_cast<size_t>(partial);
      state_idx = _flatten.toIndex(state);
      partial = 0.0;
      for (size_t j = 0; j < players.size(); ++j) {
        action_idx = players(j)->selectAction(i, state_idx);
        if (!players(j)->decrease(actions[action_idx])) {
          // Select the next best action
          if (action_idx > 0) {
            for (size_t n = 0; n < action_idx; ++n) {
              if (players(j)->decrease(actions[action_idx - n - 1])) {
                action_idx = action_idx - n - 1;
                break;
              }
            }
          }
          players(j)->set_trajectory_state(i, state_idx, action_idx);
        }
        partial += actions[action_idx];
        // now we store this data on results
        results(j, i) = actions[action_idx];
      }
      total += partial;
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

  FlattenState &flatten() { return _flatten; }

 private:
  FlattenState _flatten;
};

}

#endif //DYRWIN_RL_CRDGAME_H
