//
// Created by Elias Fernandez on 10/02/2020.
//
#include <Dyrwin/RL/simulators/CrdIslands.h>

using namespace EGTTools::RL::Simulators::CRD;

CRDSimIslands::CRDSimIslands() {
  _real_rand = std::uniform_real_distribution<double>(0.0, 1.0);
}
EGTTools::RL::DataTypes::DataTableCRD
CRDSimIslands::run_group_islands(size_t nb_evaluation_games,
                                 size_t nb_groups,
                                 size_t group_size,
                                 size_t nb_generations,
                                 size_t nb_games,
                                 size_t nb_rounds,
                                 int target,
                                 int endowment,
                                 double risk,
                                 EGTTools::RL::ActionSpace &available_actions,
                                 const std::string &agent_type,
                                 const std::vector<double> &args) {
  // First we instantiate a game - for now the game is always an Unconditional CRDGame
  CRDGame <PopContainer> game;

  // Create a vector of groups - nb_actions = available_actions.size()
  std::vector<PopContainer> groups;
  for (size_t i = 0; i < nb_groups; i++)
    groups.emplace_back(agent_type, group_size, nb_rounds, available_actions.size(), nb_rounds, endowment, args);

  // First we let all the groups adapt
#pragma omp parallel for default(none) shared(groups, nb_groups, nb_generations, \
  nb_games, nb_rounds, target, risk, available_actions, game)
  for (size_t group = 0; group < nb_groups; ++group) {
    run_crd_single_group(nb_generations, nb_games, nb_rounds, target,
                         risk, available_actions, groups[group], game);
  }

  // Now we create the data container
  // It will contain a data matrix/table of 10 columns and
  // nb_evaluation_games * group_size * nb_rounds
  std::vector<std::string> headers =
      {"group", "player", "game_index", "round", "action", "group_contributions", "contributions_others",
       "total_contribution", "payoff", "success"};
  std::vector<std::string> column_types = {"int", "int", "int", "int", "int", "int", "int", "int", "int", "int"};
  EGTTools::RL::DataTypes::DataTableCRD
      data(nb_evaluation_games * group_size * nb_rounds, 10,
           headers, column_types, groups);

  // Then we evaluate the agents by creating randomly mixed groups
  evaluate_crd_populations(nb_groups,
                           group_size,
                           group_size,
                           nb_games,
                           nb_rounds,
                           target,
                           risk,
                           available_actions,
                           game,
                           data);

  return data;
}

EGTTools::RL::DataTypes::DataTableCRD CRDSimIslands::run_population_islands(size_t nb_evaluation_games,
                                                                            size_t nb_populations,
                                                                            size_t population_size,
                                                                            size_t group_size,
                                                                            size_t nb_generations,
                                                                            size_t nb_games,
                                                                            size_t nb_rounds,
                                                                            int target,
                                                                            int endowment,
                                                                            double risk,
                                                                            ActionSpace &available_actions,
                                                                            const std::string &agent_type,
                                                                            const std::vector<double> &args) {
  // First we instantiate a game - for now the game is always an Unconditional CRDGame
  CRDGame <PopContainer> game;

  // Create a vector of groups - nb_actions = available_actions.size()
  std::vector<PopContainer> populations;
  for (size_t i = 0; i < nb_populations; i++)
    populations.emplace_back(agent_type,
                             population_size,
                             nb_rounds,
                             available_actions.size(),
                             nb_rounds,
                             endowment,
                             args);

  // First we let all the groups adapt
#pragma omp parallel for default(none) shared(populations, nb_populations, population_size, group_size, nb_generations,\
  nb_games, nb_rounds, target, risk, available_actions, game)
  for (size_t population = 0; population < nb_populations; ++population) {
    run_crd_population(population_size, group_size, nb_generations, nb_games, nb_rounds, target,
                       risk, available_actions, populations[population], game);
  }

  // Now we create the data container
  // It will contain a data matrix/table of 10 columns and
  // nb_evaluation_games * group_size * nb_rounds
  std::vector<std::string> headers =
      {"group", "player", "game_index", "round", "action", "group_contributions", "contributions_others",
       "total_contribution", "payoff", "success"};
  std::vector<std::string> column_types = {"int", "int", "int", "int", "int", "int", "int", "int", "int", "int"};
  EGTTools::RL::DataTypes::DataTableCRD
      data(nb_evaluation_games * group_size * nb_rounds, 10,
           headers, column_types, populations);

  // Then we evaluate the agents by creating randomly mixed groups
  evaluate_crd_populations(nb_populations,
                           population_size,
                           group_size,
                           nb_games,
                           nb_rounds,
                           target,
                           risk,
                           available_actions,
                           game,
                           data);

  return data;
}
