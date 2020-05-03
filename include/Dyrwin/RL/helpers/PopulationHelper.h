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
bool reinforcePath(PopContainer &players);

/**
 * Reinforces the actions of all players in PopContainer given their payoff
 * @param players : container with pointers to each agent
 * @param final_round : total number of rounds of the game, so that the trajectory can be correctly reinforced.
 * @return true if the update was successful.
 */
bool reinforcePath(PopContainer &players, size_t final_round);

/**
 * Reinforces the actions of all players in PopContainer with \p reward
 * @param players : container with pointers to each agent
 * @param reward : reward used to reinforce the agents
 * @return true if the update was successful
 */
bool reinforcePath(PopContainer &players, double reward);

/**
 * Reinforces the actions of all players in PopContainer with \p reward
 * @param players : container with pointers to each agent
 * @param final_round : total number of rounds of the game, so that the trajectory can be correctly reinforced.
 * @param reward : reward used to reinforce the agents
 * @return true if the update was successful
 */
bool reinforcePath(PopContainer &players, size_t final_round, double reward);

/**
 * Prints each player's description
 * @param players : container with pointers to each agent
 */
void printGroup(PopContainer &players);

/**
 * Updates the behavior profile of each agent.
 * @param players : container with pointers to each agent.
 * @return true if the update was successful.
 */
bool calcProbabilities(PopContainer &players);

/**
 * Resets the trajectory of each agent
 * @param players : container with pointers to each agent.
 * @return true if the reset was successful.
 */
bool resetEpisode(PopContainer &players);

/**
 * Updates the behavior profile of each agent and resets its trajectory.
 * @param players : container with pointers to each agent.
 * @return true if successful.
 */
bool calcProbabilitiesAndResetEpisode(PopContainer &players);

/**
 * Gives the sum of payoff of all players in the container.
 * @param players : container with pointers to each agent
 * @return the sum of payoffs of the players in the container.
 */
double playersPayoff(PopContainer &players);

/**
 * Sets the payoffs of the players in the container to value.
 * @param players : container with pointers to each agent.
 * @param value : value to set the payoff.
 */
void setPayoffs(PopContainer &players, double value);

/**
 * Updates the payoff of each agent by multiplying it by value.
 * @param players : container with pointers to each agent.
 * @param value : value to multiply by the previous payoff.
 */
void updatePayoffs(PopContainer &players, double value);

/**
 * Subtracts the endowment to the payoff of each agent.
 * @param players : container with pointers to each agent.
 */
void subtractEndowment(PopContainer &players);

/**
 * Returns the sum of contributions of all players in the container.
 * @param players : container with pointers to each agent in the population.
 * @return the sum of contributions of all players in the container.
 */
double playersContribution(PopContainer &players);

/**
 * Reinitialize/set to 0 the QValues of a Q-learning agent.
 * @param players : container with pointers to each agent.
 */
void resetQValues(PopContainer &players);

/**
 * Vanishes the Q-values of a previous generations by a forget_rate.
 * @param player : container with pointers to each agent.
 * @param forget_rate : indicates which part of the Q values will remain (0, 1].
 */
void forgetPropensities(PopContainer &players, double forget_rate);

/**
 * Decreases the learning rate of each agent in the container.
 *
 * This function must only be called for agents that have learning rate,
 * otherwise there will be an exception.
 * @param players : container with pointers to each agent.
 * @param decrease_rate : value by which the learning rate will be multiplied.
 */
void decreaseLearningRate(PopContainer &players, double decrease_rate);

/**
 * Increases the temperature (the probability the the best action will be selected).
 *
 * This function must only be called for agents that have learning rate,
 * otherwise there will be an exception.
 * @param players : container with pointers to each agent.
 * @param increase_rate : value by which the temperature will be multiplied.
 */
void increaseTemperature(PopContainer &players, double increase_rate);
}

#endif //DYRWIN_INCLUDE_DYRWIN_RL_HELPERS_POPULATIONHELPER_H_
