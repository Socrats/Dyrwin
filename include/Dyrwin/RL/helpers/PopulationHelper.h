//
// Created by Elias Fernandez on 03/05/2020.
//

#ifndef DYRWIN_INCLUDE_DYRWIN_RL_HELPERS_POPULATIONHELPER_H_
#define DYRWIN_INCLUDE_DYRWIN_RL_HELPERS_POPULATIONHELPER_H_

#include <Dyrwin/RL/PopContainer.hpp>

namespace EGTTools::RL::helpers {
/**
 * Reinforces the actions of all players in PopContainer given their payoff
 * @param players : container with pointers to each agent
 * @return true if the update was successful.
 */
[[maybe_unused]] bool reinforcePath(PopContainer &players);

/**
 * Reinforces the actions of all players in PopContainer given their payoff
 * @param players : container with pointers to each agent
 * @param final_round : total number of rounds of the game, so that the trajectory can be correctly reinforced.
 * @return true if the update was successful.
 */
[[maybe_unused]] bool reinforcePath(PopContainer &players, size_t final_round);

/**
 * Reinforces the actions of all players in PopContainer with \p reward.
 *
 * This function should be used when the current payoff of a player should be
 * multiplied by a factor before reinforcing.
 *
 * @param players : container with pointers to each agent
 * @param reward : reward used to reinforce the agents
 * @return true if the update was successful
 */
[[maybe_unused]] bool reinforcePath(PopContainer &players, double reward);

/**
 * Reinforces the actions of all players in PopContainer with payoff * \p factor.
 *
 * This function should be used when the current payoff of a player should be
 * multiplied by a factor before reinforcing.
 *
 * @param players : container with pointers to each agent
 * @param final_round : total number of rounds of the game, so that the trajectory can be correctly reinforced.
 * @param reward : reward used to reinforce the agents
 * @return true if the update was successful
 */
[[maybe_unused]] bool reinforcePath(PopContainer &players, size_t final_round, double reward);

/**
 * Prints each player's description
 * @param players : container with pointers to each agent
 */
[[maybe_unused]] void printGroup(PopContainer &players);

/**
 * Updates the behavior profile of each agent.
 * @param players : container with pointers to each agent.
 * @return true if the update was successful.
 */
[[maybe_unused]] bool calcProbabilities(PopContainer &players);

/**
 * Resets the trajectory of each agent
 * @param players : container with pointers to each agent.
 * @return true if the reset was successful.
 */
[[maybe_unused]] bool resetEpisode(PopContainer &players);

/**
 * Updates the behavior profile of each agent and resets its trajectory.
 * @param players : container with pointers to each agent.
 * @return true if successful.
 */
[[maybe_unused]] bool calcProbabilitiesAndResetEpisode(PopContainer &players);

/**
 * Updates the behavior profile of each agent, resets its trajectory, and updates the learning rate.
 * @param players : container with pointers to each agent.
 * @param decay : decay factor.
 * @param min_learning_rate : minimum learning rate.
 * @return true if successful.
 */
[[maybe_unused]] bool calcProbabilitiesResetEpisodeAndUpdateLearningRate(PopContainer &players,
                                                       double decay,
                                                       double min_learning_rate);

/**
 * Gives the sum of payoff of all players in the container.
 * @param players : container with pointers to each agent
 * @return the sum of payoffs of the players in the container.
 */
[[maybe_unused]] double playersPayoff(PopContainer &players);

/**
 * Sets the payoffs of the players in the container to value.
 * @param players : container with pointers to each agent.
 * @param value : value to set the payoff.
 */
[[maybe_unused]] void setPayoffs(PopContainer &players, double value);

/**
 * Updates the payoff of each agent by multiplying it by value.
 * @param players : container with pointers to each agent.
 * @param value : value to multiply by the previous payoff.
 */
[[maybe_unused]] void updatePayoffs(PopContainer &players, double value);

/**
 * Subtracts the endowment to the payoff of each agent.
 * @param players : container with pointers to each agent.
 */
[[maybe_unused]] void subtractEndowment(PopContainer &players);

/**
 * Returns the sum of contributions of all players in the container.
 * @param players : container with pointers to each agent in the population.
 * @return the sum of contributions of all players in the container.
 */
[[maybe_unused]] double playersContribution(PopContainer &players);

/**
 * Reinitialize/set to 0 the QValues of a Q-learning agent.
 * @param players : container with pointers to each agent.
 */
[[maybe_unused]] void resetQValues(PopContainer &players);

/**
 * Vanishes the Q-values of a previous generations by a forget_rate.
 * @param player : container with pointers to each agent.
 * @param forget_rate : indicates which part of the Q values will remain (0, 1].
 */
[[maybe_unused]] void forgetPropensities(PopContainer &players, double forget_rate);

/**
 * Sets the learning rate of all the players in the container.
 * @param players : container with pointers to each agent.
 * @param learning_rate : learning rate to set.
 */
[[maybe_unused]] void setLearningRate(PopContainer &players, double learning_rate);

/**
 * Decreases the learning rate of each agent in the container.
 *
 * This function must only be called for agents that have learning rate,
 * otherwise there will be an exception.
 * @param players : container with pointers to each agent.
 * @param decrease_rate : value by which the learning rate will be multiplied.
 */
[[maybe_unused]] void decreaseLearningRate(PopContainer &players, double decrease_rate);

/**
 * Increases the temperature (the probability the the best action will be selected).
 *
 * This function must only be called for agents that have learning rate,
 * otherwise there will be an exception.
 * @param players : container with pointers to each agent.
 * @param increase_rate : value by which the temperature will be multiplied.
 */
[[maybe_unused]] void increaseTemperature(PopContainer &players, double increase_rate);
}

#endif //DYRWIN_INCLUDE_DYRWIN_RL_HELPERS_POPULATIONHELPER_H_
