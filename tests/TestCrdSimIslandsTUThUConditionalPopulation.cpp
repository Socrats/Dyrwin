//
// Created by Elias Fernandez on 2019-02-10.
//
#include <iostream>
#include <map>
#include <random>
#include <Dyrwin/RL/simulators/CrdIslands.h>
#include <Dyrwin/OutputHandlers.hpp>

using namespace std;
using namespace EGTTools;

int main() {
  //parameters
  size_t nb_evaluation_games = 1000;
  size_t nb_generations = 100;
  size_t nb_populations = 2;
  size_t population_size = 24;
  size_t group_size = 6;
  size_t nb_actions = 3; //0, 2 and 4
  size_t min_rounds = 8;
  size_t max_rounds = 50;
  size_t mean_rounds = 10;
  double p = 0;
  size_t nb_games = 1000;
  double risk = 0.9;
  double alpha = 0.09;
  double temperature = 5;
  std::string agent_type = "BatchQLearning";

  auto endowment = static_cast<int>(2 * mean_rounds);
  auto target = static_cast<int>(mean_rounds * group_size);
  int delta = 20;
  EGTTools::RL::ActionSpace available_actions = EGTTools::RL::ActionSpace(nb_actions);
  for (size_t i = 0; i < nb_actions; ++i) available_actions[i] = i;

  std::vector<double> args{alpha, temperature};

  try {
    EGTTools::RL::Simulators::CRD::CRDSimIslands sim;
    EGTTools::RL::DataTypes::DataTableCRD results = sim.run_conditional_population_islandsTUThU(nb_evaluation_games,
                                                                                                nb_populations,
                                                                                                population_size,
                                                                                                group_size,
                                                                                                nb_generations,
                                                                                                nb_games,
                                                                                                min_rounds,
                                                                                                mean_rounds,
                                                                                                max_rounds,
                                                                                                p,
                                                                                                target,
                                                                                                delta,
                                                                                                endowment,
                                                                                                risk,
                                                                                                available_actions,
                                                                                                agent_type,
                                                                                                args);
    std::cout << "Finished simulation" << std::endl;
    std::string header =
        std::accumulate(results.header.begin(), results.header.end(), std::string(""),
                        [](string &ss, string &s) {
                          return ss.empty() ? s : ss + "," + s;
                        });
    EGTTools::OutputHandler::writeToCSVFile("testIslandsTUThUConditionalPopulation.csv", header, results.data);

  } catch (std::invalid_argument &e) {
    std::cerr << "\033[1;31m[EXCEPTION] Invalid argument: " << e.what() << "\033[0m" << std::endl;
    return -1;
  }

  return 0;
}
