//
// Created by Elias Fernandez on 2019-02-10.
//
#include <vector>
#include <random>
#include <Dyrwin/RL/PopContainer.hpp>
#include <Dyrwin/SeedGenerator.h>

using namespace std;
using namespace EGTTools::RL;

int main() {
  // Instantiate population
  size_t pop_size = 10;
  size_t nb_states = 3;
  size_t episode_size = 3;
  size_t nb_actions = 3;
  size_t endowment = 20;
  size_t group_size = 6;
  std::vector<double> args = {0.03, 5.0};
  std::string agent_type = "BatchQLearning";

  // random generator
  std::mt19937_64 generator(EGTTools::Random::SeedGenerator::getInstance().getSeed());

  // Create a population of conditional agents of size \p pop_size
  PopContainer population(agent_type, pop_size, nb_states, nb_actions, episode_size, endowment, args);

  // Print population
  cout << "--Print population--" << endl;
  for (size_t i = 0; i < pop_size; ++i) {
    population(i)->set_payoff(i);
    cout << population(i)->payoff() << endl;
  }
  cout << "--------" << endl;
  cout << "--Print group--" << endl;

  PopContainer group;
  std::vector<size_t> pop_index(pop_size);
  std::iota(pop_index.begin(), pop_index.end(), 0);
  for (size_t i = 0; i < group_size; ++i) {
    group.push_back(population(i));
    cout << group(i)->payoff() << endl;
  }

  cout << "--------" << endl;
  cout << "--Print random group--" << endl;

  // Get random group
  std::shuffle(pop_index.begin(), pop_index.end(), generator);
  for (size_t i = 0; i < group_size; ++i) {
    group(i) = population(pop_index[i]);
    cout << group(i)->payoff() << endl;
  }

  return 0;
}
