/**
 * Created by Elias F. Domingos
 * Please refer to the LICENCE agreement if you want to use this code.
 *
 * The objective of this main function is to add an executable that will
 * handle a population of agents, for which at each iteration a group of
 * group_size will be sampled and will play a CRD. At each epoch the accumulated
 * payoffs are used to update the estimation of the Propensity matrices and
 * the policies.
 */

#include <iostream>
#include <vector>
#include <chrono>
#include <Dyrwin/RL/simulators/CrdIslands.h>
#include <Dyrwin/CommandLineParsing.h>

using namespace std::chrono;
using namespace EGTTools::RL::Simulators::CRD;

int main(int argc, char *argv[]) {
  //parameters
  size_t nb_groups;
  size_t group_size;
  size_t actions; //0, 2 and 4
  size_t rounds;
  size_t attempts;
  size_t games;
  double cataclysm;
  double alpha, beta;
  double temperature;
  double lambda;
  double epsilon;
  double threshold;
  double endowment;
  std::string filename;
  std::string agent_type;
  Options options;

  // Setup options
  options.push_back(makeDefaultedOption<size_t>("nbGroups,N", &nb_groups, "set the number of groups", 5));
  options.push_back(makeDefaultedOption<size_t>("groupSize,M", &group_size, "set the group size", 6));
  options.push_back(makeDefaultedOption<size_t>("nbActions,p", &actions, "set the number of actions", 3));
  options.push_back(makeDefaultedOption<size_t>("nbGenerations,E", &attempts, "set the number of epochs", 1000));
  options.push_back(makeDefaultedOption<size_t>("nbGames,G", &games, "set the number of games per epoch", 1000));
  options.push_back(makeDefaultedOption<size_t>("nbRounds,t", &rounds, "set the number of rounds", 10));
  options.push_back(
      makeDefaultedOption<std::string>("agentType,A", &agent_type, "set agent type", "BatchQLearning"));
  options.push_back(makeDefaultedOption<double>("risk,r", &cataclysm, "set the risk probability", 0.9));
  options.push_back(makeDefaultedOption<double>("alpha,a", &alpha, "learning rate", 0.3));
  options.push_back(makeDefaultedOption<double>("beta,b", &beta, "learning rate", 0.03));
  options.push_back(
      makeDefaultedOption<double>("temperature,T", &temperature, "temperature of the boltzman distribution",
                                  10.));
  options.push_back(makeDefaultedOption<double>("lambda,l", &lambda, "discount factor", .01));
  options.push_back(makeDefaultedOption<double>("epsilon,e", &epsilon, "discount factor", .01));

  if (!parseCommandLine(argc, argv, options))
    return 1;

  // Calculate execution time
  high_resolution_clock::time_point t1 = high_resolution_clock::now();

  // Initialize agents depending on command option
  std::vector<double> args;
  if (agent_type == "rothErevLambda") {
    args.push_back(lambda);
    args.push_back(epsilon);
    args.push_back(temperature);
  } else if (agent_type == "QLearning") {
    args.push_back(alpha);
    args.push_back(lambda);
    args.push_back(temperature);
  } else if (agent_type == "HistericQLearning") {
    args.push_back(alpha);
    args.push_back(beta);
    args.push_back(temperature);
  } else if (agent_type == "BatchQLearning") {
    args.push_back(alpha);
    args.push_back(temperature);
  }

  endowment = 2 * rounds;
  threshold = group_size * rounds;
  EGTTools::RL::ActionSpace available_actions = EGTTools::RL::ActionSpace(actions);
  for (size_t i = 0; i < actions; ++i) available_actions[i] = i;

  CRDSimIslands sim;
  sim.set_verbose_level(0);
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
  std::cout << "Execution time: " << duration << std::endl;
  std::string header =
      std::accumulate(results.header.begin(), results.header.end(), std::string(""),
                      [](string &ss, string &s) {
                        return ss.empty() ? s : ss + "," + s;
                      });
  EGTTools::OutputHandler::writeToCSVFile("testIslandsPopulation.csv", header, results.data);

  return 0;
}