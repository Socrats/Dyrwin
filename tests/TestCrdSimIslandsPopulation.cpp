//
// Created by Elias Fernandez on 2019-02-10.
//
#include <iostream>
#include <map>
#include <random>
#include <chrono>
#include <Dyrwin/RL/simulators/CrdIslands.h>
#include <Dyrwin/OutputHandlers.hpp>

using namespace std;
using namespace std::chrono;
using namespace EGTTools;

int main() {
  //parameters
  size_t nb_evaluation_games = 1000;
  size_t nb_generations = 1000;
  size_t nb_populations = 1;
  size_t population_size = 50;
  size_t group_size = 6;
  size_t nb_actions = 5; //0, 2 and 4
  size_t nb_rounds = 10;
  size_t nb_games = 1000;
  double risk = 0.7;
  double alpha = 0.09;
  double temperature = 10;
  std::string agent_type = "BatchQLearning";

  auto endowment = static_cast<double>(2 * nb_rounds);
  auto target = static_cast<double>(nb_rounds * group_size);
  EGTTools::RL::ActionSpace available_actions = EGTTools::RL::ActionSpace(nb_actions);
  for (size_t i = 0; i < nb_actions; ++i) available_actions[i] = i;

  std::vector<double> args{alpha, temperature};

  try {
    EGTTools::RL::Simulators::CRD::CRDSimIslands sim;
    sim.set_verbose_level(0);
    sim.set_learning_rate_decay(0.1, 0.001);
    // Calculate execution time
    high_resolution_clock::time_point t1 = high_resolution_clock::now();
    EGTTools::RL::DataTypes::DataTableCRD results = sim.run_population_islands(nb_evaluation_games,
                                                                               nb_populations,
                                                                               population_size,
                                                                               group_size,
                                                                               nb_generations,
                                                                               nb_games,
                                                                               nb_rounds,
                                                                               target,
                                                                               endowment,
                                                                               risk,
                                                                               available_actions,
                                                                               agent_type,
                                                                               args);
    // Print execution time
    high_resolution_clock::time_point t2 = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(t2 - t1).count();
    std::cout << "Finished simulation" << std::endl;
    std::cout << "Execution time: " << duration << std::endl;
    std::string header =
        std::accumulate(results.header.begin(), results.header.end(), std::string(""),
                        [](string &ss, string &s) {
                          return ss.empty() ? s : ss + "," + s;
                        });
    EGTTools::OutputHandler::writeToCSVFile("testIslandsPopulation.csv", header, results.data);

  } catch (std::invalid_argument &e) {
    std::cerr << "\033[1;31m[EXCEPTION] Invalid argument: " << e.what() << "\033[0m" << std::endl;
    return -1;
  }

  return 0;
}
