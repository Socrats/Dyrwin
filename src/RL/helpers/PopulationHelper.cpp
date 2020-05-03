//
// Created by Elias Fernandez on 03/05/2020.
//

#include <Dyrwin/RL/helpers/PopulationHelper.h>

bool EGTTools::RL::helpers::reinforcePath(PopContainer &players) {
  for (auto &player : players)
    player->reinforceTrajectory();
  return true;
}

bool EGTTools::RL::helpers::reinforcePath(PopContainer &players, size_t final_round) {
  for (auto &player : players)
    player->reinforceTrajectory(final_round);
  return true;
}

void EGTTools::RL::helpers::printGroup(PopContainer &players) {
  for (auto &player : players) {
    std::cout << *player << std::endl;
  }
}

bool EGTTools::RL::helpers::calcProbabilities(PopContainer &players) {
  for (auto &player : players)
    player->inferPolicy();
  return true;
}

bool EGTTools::RL::helpers::resetEpisode(PopContainer &players) {
  for (auto &player : players) {
    player->resetTrajectory();
  }
  return true;
}

double EGTTools::RL::helpers::playersPayoff(PopContainer &players) {
  double total = 0;
  for (auto &player : players)
    total += player->payoff();

  return total;
}

void EGTTools::RL::helpers::setPayoffs(PopContainer &players, double value) {
  for (auto &player: players) {
    player->set_payoff(value);
  }
}

void EGTTools::RL::helpers::updatePayoffs(PopContainer &players, double value) {
  for (auto &player: players) {
    player->multiply_by_payoff(value);
  }
}

void EGTTools::RL::helpers::subtractEndowment(PopContainer &players) {
  for (auto &player: players) {
    player->subtract_endowment_to_payoff();
  }
}

double EGTTools::RL::helpers::playersContribution(PopContainer &players) {
  double total = 0;
  for (auto &player : players)
    total += player->endowment() - player->payoff();

  return total;
}

void EGTTools::RL::helpers::resetQValues(PopContainer &players) {
  for (auto &player : players) {
    player->resetQValues();
  }
}

void EGTTools::RL::helpers::forgetPropensities(PopContainer &players, double forget_rate) {
  for (auto &player : players) {
    player->forgetQValues(forget_rate);
  }
}