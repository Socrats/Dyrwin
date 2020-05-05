//
// Created by Elias Fernandez on 10/02/2020.
//
#include <Dyrwin/RL/simulators/CrdIslands.h>

using namespace EGTTools::RL::Simulators::CRD;

CRDSimIslands::CRDSimIslands() {
  // verbose level is initialized to highest value, so that all game data is added to the data container
  _verbose_level = 1;
  _decay = 1.;
  _min_learning_rate = 1.;
  _real_rand = std::uniform_real_distribution<double>(0.0, 1.0);
}
void CRDSimIslands::reinforce_population(double &pool,
                                         size_t &success,
                                         double target,
                                         double &risk,
                                         PopContainer &pop,
                                         size_t &final_round) {
  if (pool >= target) {
    success++;
    EGTTools::RL::helpers::reinforcePath(pop, final_round);
  } else {
    EGTTools::RL::helpers::reinforcePath(pop, final_round, 1 - risk);
  }
}
void CRDSimIslands::reinforce_population(double &pool,
                                         size_t &success,
                                         double target,
                                         double &risk,
                                         EGTTools::RL::PopContainer &pop) {
  if (pool >= target) {
    success++;
    EGTTools::RL::helpers::reinforcePath(pop);
  } else {
    EGTTools::RL::helpers::reinforcePath(pop, 1 - risk);
  }
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
  CRDGame<PopContainer, void, void> game;

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

  // Define variables and structures to store the evaluation data
  size_t nb_rows, nb_columns;
  std::vector<std::string> headers;
  std::vector<std::string> column_types;

  if (_verbose_level == 0) {
    headers = {"group", "success", "final_public_account"};
    column_types = {"int", "bool", "float"};
    nb_rows = nb_evaluation_games;
    nb_columns = 3;
  } else {
    headers = {"group", "player", "game_index", "round", "action", "group_contributions", "contributions_others",
               "total_contribution", "payoff", "success", "target", "final_public_account", "final_round"};
    column_types = {"int", "int", "int", "int", "int", "int", "int", "int", "int", "bool", "int", "int", "int"};
    nb_rows = nb_evaluation_games * group_size * nb_rounds;
    nb_columns = 13;
  }
  // Now we create the data container
  EGTTools::RL::DataTypes::DataTableCRD data(nb_rows, nb_columns, headers, column_types, groups);

  // Then we evaluate the agents by creating randomly mixed groups
  evaluate_crd_populations(nb_groups,
                           group_size,
                           group_size,
                           nb_evaluation_games,
                           nb_rounds,
                           target,
                           risk,
                           available_actions,
                           game,
                           data);

  return data;
}
EGTTools::RL::DataTypes::DataTableCRD
CRDSimIslands::run_conditional_group_islands(size_t nb_evaluation_games,
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
  // Instantiate factored state
  // The state is composed of the (current round, count of each of the possible action)
  // However, since there is fixed number of agents, the possible combination of counts
  // are fixed and can be calculated with a binomial.
  FlattenState flatten(Factors{nb_rounds, EGTTools::starsBars(group_size, available_actions.size())});
  // First we instantiate a game - for now the game is always an Unconditional CRDGame
  CRDConditionalCount<void, void> game(flatten);

  // Create a vector of groups - nb_actions = available_actions.size()
  std::vector<PopContainer> groups;
  for (size_t i = 0; i < nb_groups; i++)
    groups.emplace_back(agent_type,
                        group_size,
                        game.flatten().factor_space,
                        available_actions.size(),
                        nb_rounds,
                        endowment,
                        args);

  // First we let all the groups adapt
#pragma omp parallel for default(none) shared(groups, nb_groups, nb_generations, \
  nb_games, nb_rounds, target, risk, available_actions, game)
  for (size_t group = 0; group < nb_groups; ++group) {
    run_crd_single_group(nb_generations, nb_games, nb_rounds, target,
                         risk, available_actions, groups[group], game);
  }

  // Define variables and structures to store the evaluation data
  size_t nb_rows, nb_columns;
  std::vector<std::string> headers;
  std::vector<std::string> column_types;

  if (_verbose_level == 0) {
    headers = {"group", "success", "final_public_account"};
    column_types = {"int", "bool", "float"};
    nb_rows = nb_evaluation_games;
    nb_columns = 3;
  } else {
    headers = {"group", "player", "game_index", "round", "action", "group_contributions", "contributions_others",
               "total_contribution", "payoff", "success", "target", "final_public_account", "final_round"};
    column_types = {"int", "int", "int", "int", "int", "int", "int", "int", "int", "bool", "int", "int", "int"};
    nb_rows = nb_evaluation_games * group_size * nb_rounds;
    nb_columns = 13;
  }
  // Now we create the data container
  EGTTools::RL::DataTypes::DataTableCRD data(nb_rows, nb_columns, headers, column_types, groups);

  // Then we evaluate the agents by creating randomly mixed groups
  evaluate_crd_populations(nb_groups,
                           group_size,
                           group_size,
                           nb_evaluation_games,
                           nb_rounds,
                           target,
                           risk,
                           available_actions,
                           game,
                           data);

  return data;
}
EGTTools::RL::DataTypes::DataTableCRD CRDSimIslands::run_group_islandsTU(size_t nb_evaluation_games,
                                                                         size_t nb_groups,
                                                                         size_t group_size,
                                                                         size_t nb_generations,
                                                                         size_t nb_games,
                                                                         size_t min_rounds,
                                                                         size_t mean_rounds,
                                                                         size_t max_rounds,
                                                                         double p,
                                                                         int target,
                                                                         int endowment,
                                                                         double risk,
                                                                         EGTTools::RL::ActionSpace &available_actions,
                                                                         const std::string &agent_type,
                                                                         const std::vector<double> &args) {
  // First we calculate the probability of the game ending after the minimum number of rounds
  if (mean_rounds > 0) {
    p = 1.0 / static_cast<double>(mean_rounds - min_rounds + 1);
  }
  EGTTools::TimingUncertainty<std::mt19937_64> tu(p, max_rounds);
  // Then we instantiate a game with uncertainty
  CRDGame <PopContainer, EGTTools::TimingUncertainty<std::mt19937_64>, std::mt19937_64> game;

  // Create a vector of groups - nb_actions = available_actions.size()
  std::vector<PopContainer> groups;
  for (size_t i = 0; i < nb_groups; i++)
    groups.emplace_back(agent_type, group_size, max_rounds, available_actions.size(), max_rounds, endowment, args);

  // First we let all the groups adapt
#pragma omp parallel for default(none) shared(groups, tu, nb_groups, nb_generations, \
  nb_games, min_rounds, target, risk, available_actions, game)
  for (size_t group = 0; group < nb_groups; ++group) {
    run_crd_single_groupTU(nb_generations, nb_games, min_rounds, tu, target,
                           risk, available_actions, groups[group], game);
  }

  // Define variables and structures to store the evaluation data
  size_t nb_rows, nb_columns;
  std::vector<std::string> headers;
  std::vector<std::string> column_types;

  if (_verbose_level == 0) {
    headers = {"group", "success", "final_public_account", "final_round"};
    column_types = {"int", "bool", "float", "int"};
    nb_rows = nb_evaluation_games;
    nb_columns = 4;
  } else {
    headers = {"group", "player", "game_index", "round", "action", "group_contributions", "contributions_others",
               "total_contribution", "payoff", "success", "target", "final_public_account", "final_round"};
    column_types = {"int", "int", "int", "int", "int", "int", "int", "int", "int", "bool", "int", "int", "int"};
    nb_rows = nb_evaluation_games * group_size * max_rounds;
    nb_columns = 13;
  }
  // Now we create the data container
  EGTTools::RL::DataTypes::DataTableCRD data(nb_rows, nb_columns, headers, column_types, groups);

  // Then we evaluate the agents by creating randomly mixed groups
  auto total_nb_rounds = evaluate_crd_populationsTU(nb_groups,
                                                    group_size,
                                                    group_size,
                                                    nb_evaluation_games,
                                                    min_rounds,
                                                    tu,
                                                    target,
                                                    risk,
                                                    available_actions,
                                                    game,
                                                    data);

  // Finally we clear the unused rows
  if (_verbose_level > 0) {
    auto total_rows = group_size * total_nb_rounds;
    data.data.conservativeResize(total_rows, 13);
  }

  return data;
}
EGTTools::RL::DataTypes::DataTableCRD CRDSimIslands::run_conditional_group_islandsTU(size_t nb_evaluation_games,
                                                                                     size_t nb_groups,
                                                                                     size_t group_size,
                                                                                     size_t nb_generations,
                                                                                     size_t nb_games,
                                                                                     size_t min_rounds,
                                                                                     size_t mean_rounds,
                                                                                     size_t max_rounds,
                                                                                     double p,
                                                                                     int target,
                                                                                     int endowment,
                                                                                     double risk,
                                                                                     EGTTools::RL::ActionSpace &available_actions,
                                                                                     const std::string &agent_type,
                                                                                     const std::vector<double> &args) {
  // First we calculate the probability of the game ending after the minimum number of rounds
  if (mean_rounds > 0) {
    p = 1.0 / static_cast<double>(mean_rounds - min_rounds + 1);
  }
  EGTTools::TimingUncertainty<std::mt19937_64> tu(p, max_rounds);
  // Instantiate factored state
  // The first dimension indicate the round of the game,
  // and the second the donations of the group in the previous round (maximum group_size * (nb_action - 1) + 1)
//  FlattenState flatten(Factors{max_rounds, (group_size * (available_actions.size() - 1)) + 1});
  // The state is composed of the (current round, nb of times action 0 selected, ..., nb times action n selected)
  // The state space depends on the number of actions
  FlattenState flatten(Factors{max_rounds, EGTTools::starsBars(group_size, available_actions.size())});
  // First we instantiate a game - for now the game is always an Unconditional CRDGame
  CRDConditionalCount <EGTTools::TimingUncertainty<std::mt19937_64>, std::mt19937_64> game(flatten);

  // Create a vector of groups - nb_actions = available_actions.size()
  std::vector<PopContainer> groups;
  for (size_t i = 0; i < nb_groups; i++)
    groups.emplace_back(agent_type,
                        group_size,
                        game.flatten().factor_space,
                        available_actions.size(),
                        max_rounds,
                        endowment,
                        args);

  // First we let all the groups adapt
#pragma omp parallel for default(none) shared(groups, tu, nb_groups, nb_generations, \
  nb_games, min_rounds, target, risk, available_actions, game)
  for (size_t group = 0; group < nb_groups; ++group) {
    run_crd_single_groupTU(nb_generations, nb_games, min_rounds, tu, target,
                           risk, available_actions, groups[group], game);
  }

  // Define variables and structures to store the evaluation data
  size_t nb_rows, nb_columns;
  std::vector<std::string> headers;
  std::vector<std::string> column_types;

  if (_verbose_level == 0) {
    headers = {"group", "success", "final_public_account", "final_round"};
    column_types = {"int", "bool", "float", "int"};
    nb_rows = nb_evaluation_games;
    nb_columns = 4;
  } else {
    headers = {"group", "player", "game_index", "round", "action", "group_contributions", "contributions_others",
               "total_contribution", "payoff", "success", "target", "final_public_account", "final_round"};
    column_types = {"int", "int", "int", "int", "int", "int", "int", "int", "int", "bool", "int", "int", "int"};
    nb_rows = nb_evaluation_games * group_size * max_rounds;
    nb_columns = 13;
  }
  // Now we create the data container
  EGTTools::RL::DataTypes::DataTableCRD data(nb_rows, nb_columns, headers, column_types, groups);

  // Then we evaluate the agents by creating randomly mixed groups
  auto total_nb_rounds = evaluate_crd_populationsTU(nb_groups,
                                                    group_size,
                                                    group_size,
                                                    nb_evaluation_games,
                                                    min_rounds,
                                                    tu,
                                                    target,
                                                    risk,
                                                    available_actions,
                                                    game,
                                                    data);

  // Finally we clear the unused rows
  if (_verbose_level > 0) {
    auto total_rows = group_size * total_nb_rounds;
    data.data.conservativeResize(total_rows, 13);
  }

  return data;
}
EGTTools::RL::DataTypes::DataTableCRD CRDSimIslands::run_group_islandsThU(size_t nb_evaluation_games,
                                                                          size_t nb_groups,
                                                                          size_t group_size,
                                                                          size_t nb_generations,
                                                                          size_t nb_games,
                                                                          size_t nb_rounds,
                                                                          int target,
                                                                          double delta,
                                                                          int endowment,
                                                                          double risk,
                                                                          ActionSpace &available_actions,
                                                                          const std::string &agent_type,
                                                                          const std::vector<double> &args) {
  // First we instantiate a game - for now the game is always an Unconditional CRDGame
  CRDGame<PopContainer, void, void> game;

  // Create a vector of groups - nb_actions = available_actions.size()
  std::vector<PopContainer> groups;
  for (size_t i = 0; i < nb_groups; i++)
    groups.emplace_back(agent_type, group_size, nb_rounds, available_actions.size(), nb_rounds, endowment, args);

  // First we let all the groups adapt
#pragma omp parallel for default(none) shared(groups, nb_groups, nb_generations, \
  nb_games, nb_rounds, target, delta, risk, available_actions, game)
  for (size_t group = 0; group < nb_groups; ++group) {
    run_crd_single_groupThU(nb_generations, nb_games, nb_rounds, target, delta,
                            risk, available_actions, groups[group], game);
  }

  // Define variables and structures to store the evaluation data
  size_t nb_rows, nb_columns;
  std::vector<std::string> headers;
  std::vector<std::string> column_types;

  if (_verbose_level == 0) {
    headers = {"group", "success", "final_public_account", "final_target"};
    column_types = {"int", "bool", "float", "float"};
    nb_rows = nb_evaluation_games;
    nb_columns = 4;
  } else {
    headers = {"group", "player", "game_index", "round", "action", "group_contributions", "contributions_others",
               "total_contribution", "payoff", "success", "target", "final_public_account", "final_round"};
    column_types = {"int", "int", "int", "int", "int", "int", "int", "int", "int", "bool", "int", "int", "int"};
    nb_rows = nb_evaluation_games * group_size * nb_rounds;
    nb_columns = 13;
  }
  // Now we create the data container
  EGTTools::RL::DataTypes::DataTableCRD data(nb_rows, nb_columns, headers, column_types, groups);

  // Then we evaluate the agents by creating randomly mixed groups
  evaluate_crd_populationsThU(nb_groups,
                              group_size,
                              group_size,
                              nb_evaluation_games,
                              nb_rounds,
                              target,
                              delta,
                              risk,
                              available_actions,
                              game,
                              data);

  return data;
}
EGTTools::RL::DataTypes::DataTableCRD CRDSimIslands::run_conditional_group_islandsThU(size_t nb_evaluation_games,
                                                                                      size_t nb_groups,
                                                                                      size_t group_size,
                                                                                      size_t nb_generations,
                                                                                      size_t nb_games,
                                                                                      size_t nb_rounds,
                                                                                      int target,
                                                                                      double delta,
                                                                                      int endowment,
                                                                                      double risk,
                                                                                      ActionSpace &available_actions,
                                                                                      const std::string &agent_type,
                                                                                      const std::vector<double> &args) {
  // Instantiate factored state
  // The state is composed of the (current round, count of each of the possible action)
  // However, since there is fixed number of agents, the possible combination of counts
  // are fixed and can be calculated with a binomial.
  FlattenState flatten(Factors{nb_rounds, EGTTools::starsBars(group_size, available_actions.size())});
  // First we instantiate a game - for now the game is always an Unconditional CRDGame
  CRDConditionalCount<void, void> game(flatten);

  // Create a vector of groups - nb_actions = available_actions.size()
  std::vector<PopContainer> groups;
  for (size_t i = 0; i < nb_groups; i++)
    groups.emplace_back(agent_type,
                        group_size,
                        game.flatten().factor_space,
                        available_actions.size(),
                        nb_rounds,
                        endowment,
                        args);

  // First we let all the groups adapt
#pragma omp parallel for default(none) shared(groups, nb_groups, nb_generations, \
  nb_games, nb_rounds, target, delta, risk, available_actions, game)
  for (size_t group = 0; group < nb_groups; ++group) {
    run_crd_single_groupThU(nb_generations, nb_games, nb_rounds, target, delta,
                            risk, available_actions, groups[group], game);
  }

  // Define variables and structures to store the evaluation data
  size_t nb_rows, nb_columns;
  std::vector<std::string> headers;
  std::vector<std::string> column_types;

  if (_verbose_level == 0) {
    headers = {"group", "success", "final_public_account", "final_target"};
    column_types = {"int", "bool", "float", "float"};
    nb_rows = nb_evaluation_games;
    nb_columns = 4;
  } else {
    headers = {"group", "player", "game_index", "round", "action", "group_contributions", "contributions_others",
               "total_contribution", "payoff", "success", "target", "final_public_account", "final_round"};
    column_types = {"int", "int", "int", "int", "int", "int", "int", "int", "int", "bool", "int", "int", "int"};
    nb_rows = nb_evaluation_games * group_size * nb_rounds;
    nb_columns = 13;
  }
  // Now we create the data container
  EGTTools::RL::DataTypes::DataTableCRD data(nb_rows, nb_columns, headers, column_types, groups);

  // Then we evaluate the agents by creating randomly mixed groups
  evaluate_crd_populationsThU(nb_groups,
                              group_size,
                              group_size,
                              nb_evaluation_games,
                              nb_rounds,
                              target,
                              delta,
                              risk,
                              available_actions,
                              game,
                              data);

  return data;
}
EGTTools::RL::DataTypes::DataTableCRD CRDSimIslands::run_group_islandsTUThU(size_t nb_evaluation_games,
                                                                            size_t nb_groups,
                                                                            size_t group_size,
                                                                            size_t nb_generations,
                                                                            size_t nb_games,
                                                                            size_t min_rounds,
                                                                            size_t mean_rounds,
                                                                            size_t max_rounds,
                                                                            double p,
                                                                            int target,
                                                                            double delta,
                                                                            int endowment,
                                                                            double risk,
                                                                            ActionSpace &available_actions,
                                                                            const std::string &agent_type,
                                                                            const std::vector<double> &args) {
  // First we calculate the probability of the game ending after the minimum number of rounds
  if (mean_rounds > 0) {
    p = 1.0 / static_cast<double>(mean_rounds - min_rounds + 1);
  }
  EGTTools::TimingUncertainty<std::mt19937_64> tu(p, max_rounds);
  // Then we instantiate a game with uncertainty
  CRDGame <PopContainer, EGTTools::TimingUncertainty<std::mt19937_64>, std::mt19937_64> game;

  // Create a vector of groups - nb_actions = available_actions.size()
  std::vector<PopContainer> groups;
  for (size_t i = 0; i < nb_groups; i++)
    groups.emplace_back(agent_type, group_size, max_rounds, available_actions.size(), max_rounds, endowment, args);

  // First we let all the groups adapt
#pragma omp parallel for default(none) shared(groups, tu, nb_groups, nb_generations, \
  nb_games, min_rounds, target, delta, risk, available_actions, game)
  for (size_t group = 0; group < nb_groups; ++group) {
    run_crd_single_groupTUThU(nb_generations, nb_games, min_rounds, tu, target, delta,
                              risk, available_actions, groups[group], game);
  }

  // Define variables and structures to store the evaluation data
  size_t nb_rows, nb_columns;
  std::vector<std::string> headers;
  std::vector<std::string> column_types;

  if (_verbose_level == 0) {
    headers = {"group", "success", "final_public_account", "final_round", "final_target"};
    column_types = {"int", "bool", "float", "int", "float"};
    nb_rows = nb_evaluation_games;
    nb_columns = 5;
  } else {
    headers = {"group", "player", "game_index", "round", "action", "group_contributions", "contributions_others",
               "total_contribution", "payoff", "success", "target", "final_public_account", "final_round"};
    column_types = {"int", "int", "int", "int", "int", "int", "int", "int", "int", "bool", "int", "int", "int"};
    nb_rows = nb_evaluation_games * group_size * max_rounds;
    nb_columns = 13;
  }
  // Now we create the data container
  EGTTools::RL::DataTypes::DataTableCRD data(nb_rows, nb_columns, headers, column_types, groups);

  // Then we evaluate the agents by creating randomly mixed groups
  auto total_nb_rounds = evaluate_crd_populationsTUThU(nb_groups,
                                                       group_size,
                                                       group_size,
                                                       nb_evaluation_games,
                                                       min_rounds,
                                                       tu,
                                                       target,
                                                       delta,
                                                       risk,
                                                       available_actions,
                                                       game,
                                                       data);

  // Finally we clear the unused rows
  if (_verbose_level > 0) {
    auto total_rows = group_size * total_nb_rounds;
    data.data.conservativeResize(total_rows, 13);
  }

  return data;
}
EGTTools::RL::DataTypes::DataTableCRD CRDSimIslands::run_conditional_group_islandsTUThU(size_t nb_evaluation_games,
                                                                                        size_t nb_groups,
                                                                                        size_t group_size,
                                                                                        size_t nb_generations,
                                                                                        size_t nb_games,
                                                                                        size_t min_rounds,
                                                                                        size_t mean_rounds,
                                                                                        size_t max_rounds,
                                                                                        double p,
                                                                                        int target,
                                                                                        double delta,
                                                                                        int endowment,
                                                                                        double risk,
                                                                                        ActionSpace &available_actions,
                                                                                        const std::string &agent_type,
                                                                                        const std::vector<double> &args) {
  // First we calculate the probability of the game ending after the minimum number of rounds
  if (mean_rounds > 0) {
    p = 1.0 / static_cast<double>(mean_rounds - min_rounds + 1);
  }
  EGTTools::TimingUncertainty<std::mt19937_64> tu(p, max_rounds);
  // Instantiate factored state
  // The state is composed of the (current round, nb of times action 0 selected, ..., nb times action n selected)
  // The state space depends on the number of actions
  FlattenState flatten(Factors{max_rounds, EGTTools::starsBars(group_size, available_actions.size())});
  // First we instantiate a game - for now the game is always an Unconditional CRDGame
  CRDConditionalCount <EGTTools::TimingUncertainty<std::mt19937_64>, std::mt19937_64> game(flatten);

  // Create a vector of groups - nb_actions = available_actions.size()
  std::vector<PopContainer> groups;
  for (size_t i = 0; i < nb_groups; i++)
    groups.emplace_back(agent_type,
                        group_size,
                        game.flatten().factor_space,
                        available_actions.size(),
                        max_rounds,
                        endowment,
                        args);

  // First we let all the groups adapt
#pragma omp parallel for default(none) shared(groups, tu, nb_groups, nb_generations, \
  nb_games, min_rounds, target, delta, risk, available_actions, game)
  for (size_t group = 0; group < nb_groups; ++group) {
    run_crd_single_groupTUThU(nb_generations, nb_games, min_rounds, tu, target, delta,
                              risk, available_actions, groups[group], game);
  }

  // Define variables and structures to store the evaluation data
  size_t nb_rows, nb_columns;
  std::vector<std::string> headers;
  std::vector<std::string> column_types;

  if (_verbose_level == 0) {
    headers = {"group", "success", "final_public_account", "final_round", "final_target"};
    column_types = {"int", "bool", "float", "int", "float"};
    nb_rows = nb_evaluation_games;
    nb_columns = 5;
  } else {
    headers = {"group", "player", "game_index", "round", "action", "group_contributions", "contributions_others",
               "total_contribution", "payoff", "success", "target", "final_public_account", "final_round"};
    column_types = {"int", "int", "int", "int", "int", "int", "int", "int", "int", "bool", "int", "int", "int"};
    nb_rows = nb_evaluation_games * group_size * max_rounds;
    nb_columns = 13;
  }
  // Now we create the data container
  EGTTools::RL::DataTypes::DataTableCRD data(nb_rows, nb_columns, headers, column_types, groups);

  // Then we evaluate the agents by creating randomly mixed groups
  auto total_nb_rounds = evaluate_crd_populationsTUThU(nb_groups,
                                                       group_size,
                                                       group_size,
                                                       nb_evaluation_games,
                                                       min_rounds,
                                                       tu,
                                                       target,
                                                       delta,
                                                       risk,
                                                       available_actions,
                                                       game,
                                                       data);

  // Finally we clear the unused rows
  if (_verbose_level > 0) {
    auto total_rows = group_size * total_nb_rounds;
    data.data.conservativeResize(total_rows, 13);
  }

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
  CRDGame<PopContainer, void, void> game;

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
#pragma omp parallel for default(none) shared(populations, nb_populations, population_size, group_size, nb_generations, \
  nb_games, nb_rounds, target, risk, available_actions, game)
  for (size_t population = 0; population < nb_populations; ++population) {
    run_crd_population(population_size, group_size, nb_generations, nb_games, nb_rounds, target,
                       risk, available_actions, populations[population], game);
  }

  // Define variables and structures to store the evaluation data
  size_t nb_rows, nb_columns;
  std::vector<std::string> headers;
  std::vector<std::string> column_types;

  if (_verbose_level == 0) {
    headers = {"group", "success", "final_public_account"};
    column_types = {"int", "bool", "float"};
    nb_rows = nb_evaluation_games;
    nb_columns = 3;
  } else {
    headers = {"group", "player", "game_index", "round", "action", "group_contributions", "contributions_others",
               "total_contribution", "payoff", "success", "target", "final_public_account", "final_round"};
    column_types = {"int", "int", "int", "int", "int", "int", "int", "int", "int", "bool", "int", "int", "int"};
    nb_rows = nb_evaluation_games * group_size * nb_rounds;
    nb_columns = 13;
  }
  // Now we create the data container
  EGTTools::RL::DataTypes::DataTableCRD data(nb_rows, nb_columns, headers, column_types, populations);

  // Then we evaluate the agents by creating randomly mixed groups
  evaluate_crd_populations(nb_populations,
                           population_size,
                           group_size,
                           nb_evaluation_games,
                           nb_rounds,
                           target,
                           risk,
                           available_actions,
                           game,
                           data);

  return data;
}
EGTTools::RL::DataTypes::DataTableCRD CRDSimIslands::run_conditional_population_islands(size_t nb_evaluation_games,
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
  // Instantiate factored state
  // The state is composed of the (current round, count of each of the possible action)
  // However, since there is fixed number of agents, the possible combination of counts
  // are fixed and can be calculated with a binomial.
  FlattenState flatten(Factors{nb_rounds, EGTTools::starsBars(group_size, available_actions.size())});
  // First we instantiate a game - for now the game is always an Unconditional CRDGame
  CRDConditionalCount<void, void> game(flatten);

  // Create a vector of groups - nb_actions = available_actions.size()
  std::vector<PopContainer> populations;
  for (size_t i = 0; i < nb_populations; i++)
    populations.emplace_back(agent_type,
                             population_size,
                             game.flatten().factor_space,
                             available_actions.size(),
                             nb_rounds,
                             endowment,
                             args);

  // First we let all the groups adapt
#pragma omp parallel for default(none) shared(populations, nb_populations, population_size, group_size, nb_generations, \
  nb_games, nb_rounds, target, risk, available_actions, game)
  for (size_t population = 0; population < nb_populations; ++population) {
    run_crd_population(population_size, group_size, nb_generations, nb_games, nb_rounds, target,
                       risk, available_actions, populations[population], game);
  }

  // Define variables and structures to store the evaluation data
  size_t nb_rows, nb_columns;
  std::vector<std::string> headers;
  std::vector<std::string> column_types;

  if (_verbose_level == 0) {
    headers = {"group", "success", "final_public_account"};
    column_types = {"int", "bool", "float"};
    nb_rows = nb_evaluation_games;
    nb_columns = 3;
  } else {
    headers = {"group", "player", "game_index", "round", "action", "group_contributions", "contributions_others",
               "total_contribution", "payoff", "success", "target", "final_public_account", "final_round"};
    column_types = {"int", "int", "int", "int", "int", "int", "int", "int", "int", "bool", "int", "int", "int"};
    nb_rows = nb_evaluation_games * group_size * nb_rounds;
    nb_columns = 13;
  }
  // Now we create the data container
  EGTTools::RL::DataTypes::DataTableCRD data(nb_rows, nb_columns, headers, column_types, populations);

  // Then we evaluate the agents by creating randomly mixed groups
  evaluate_crd_populations(nb_populations,
                           population_size,
                           group_size,
                           nb_evaluation_games,
                           nb_rounds,
                           target,
                           risk,
                           available_actions,
                           game,
                           data);

  return data;
}
EGTTools::RL::DataTypes::DataTableCRD CRDSimIslands::run_population_islandsTU(size_t nb_evaluation_games,
                                                                              size_t nb_populations,
                                                                              size_t population_size,
                                                                              size_t group_size,
                                                                              size_t nb_generations,
                                                                              size_t nb_games,
                                                                              size_t min_rounds,
                                                                              size_t mean_rounds,
                                                                              size_t max_rounds,
                                                                              double p,
                                                                              int target,
                                                                              int endowment,
                                                                              double risk,
                                                                              ActionSpace &available_actions,
                                                                              const std::string &agent_type,
                                                                              const std::vector<double> &args) {
  // First we calculate the probability of the game ending after the minimum number of rounds
  if (mean_rounds > 0) {
    p = 1.0 / static_cast<double>(mean_rounds - min_rounds + 1);
  }
  EGTTools::TimingUncertainty<std::mt19937_64> tu(p, max_rounds);
  // Then we instantiate a game with uncertainty
  CRDGame <PopContainer, EGTTools::TimingUncertainty<std::mt19937_64>, std::mt19937_64> game;

  // Create a vector of groups - nb_actions = available_actions.size()
  std::vector<PopContainer> populations;
  for (size_t i = 0; i < nb_populations; i++)
    populations.emplace_back(agent_type,
                             population_size,
                             max_rounds,
                             available_actions.size(),
                             max_rounds,
                             endowment,
                             args);

  // First we let all the groups adapt
#pragma omp parallel for default(none) shared(populations, tu, nb_populations, population_size, group_size, nb_generations, \
  nb_games, min_rounds, target, risk, available_actions, game)
  for (size_t population = 0; population < nb_populations; ++population) {
    run_crd_populationTU(population_size, group_size, nb_generations, nb_games, min_rounds, tu, target,
                         risk, available_actions, populations[population], game);
  }

  // Define variables and structures to store the evaluation data
  size_t nb_rows, nb_columns;
  std::vector<std::string> headers;
  std::vector<std::string> column_types;

  if (_verbose_level == 0) {
    headers = {"group", "success", "final_public_account", "final_round"};
    column_types = {"int", "bool", "float", "int"};
    nb_rows = nb_evaluation_games;
    nb_columns = 4;
  } else {
    headers = {"group", "player", "game_index", "round", "action", "group_contributions", "contributions_others",
               "total_contribution", "payoff", "success", "target", "final_public_account", "final_round"};
    column_types = {"int", "int", "int", "int", "int", "int", "int", "int", "int", "bool", "int", "int", "int"};
    nb_rows = nb_evaluation_games * group_size * max_rounds;
    nb_columns = 13;
  }
  // Now we create the data container
  EGTTools::RL::DataTypes::DataTableCRD data(nb_rows, nb_columns, headers, column_types, populations);

  // Then we evaluate the agents by creating randomly mixed groups
  auto total_nb_rounds = evaluate_crd_populationsTU(nb_populations,
                                                    population_size,
                                                    group_size,
                                                    nb_evaluation_games,
                                                    min_rounds,
                                                    tu,
                                                    target,
                                                    risk,
                                                    available_actions,
                                                    game,
                                                    data);

  // Finally we clear the unused rows
  if (_verbose_level > 0) {
    auto total_rows = group_size * total_nb_rounds;
    data.data.conservativeResize(total_rows, 13);
  }

  return data;
}
EGTTools::RL::DataTypes::DataTableCRD CRDSimIslands::run_conditional_population_islandsTU(size_t nb_evaluation_games,
                                                                                          size_t nb_populations,
                                                                                          size_t population_size,
                                                                                          size_t group_size,
                                                                                          size_t nb_generations,
                                                                                          size_t nb_games,
                                                                                          size_t min_rounds,
                                                                                          size_t mean_rounds,
                                                                                          size_t max_rounds,
                                                                                          double p,
                                                                                          int target,
                                                                                          int endowment,
                                                                                          double risk,
                                                                                          ActionSpace &available_actions,
                                                                                          const std::string &agent_type,
                                                                                          const std::vector<double> &args) {
  // First we calculate the probability of the game ending after the minimum number of rounds
  if (mean_rounds > 0) {
    p = 1.0 / static_cast<double>(mean_rounds - min_rounds + 1);
  }
  EGTTools::TimingUncertainty<std::mt19937_64> tu(p, max_rounds);
  // Instantiate factored state
  // The state is composed of the (current round, nb of times action 0 selected, ..., nb times action n selected)
  // The state space depends on the number of actions
  FlattenState flatten(Factors{max_rounds, EGTTools::starsBars(group_size, available_actions.size())});
  // First we instantiate a game - for now the game is always an Unconditional CRDGame
  CRDConditionalCount <EGTTools::TimingUncertainty<std::mt19937_64>, std::mt19937_64> game(flatten);

  // Create a vector of groups - nb_actions = available_actions.size()
  std::vector<PopContainer> populations;
  for (size_t i = 0; i < nb_populations; i++)
    populations.emplace_back(agent_type,
                             population_size,
                             game.flatten().factor_space,
                             available_actions.size(),
                             max_rounds,
                             endowment,
                             args);

  // First we let all the groups adapt
#pragma omp parallel for default(none) shared(populations, tu, nb_populations, population_size, group_size, nb_generations, \
  nb_games, min_rounds, target, risk, available_actions, game)
  for (size_t population = 0; population < nb_populations; ++population) {
    run_crd_populationTU(population_size, group_size, nb_generations, nb_games, min_rounds, tu, target,
                         risk, available_actions, populations[population], game);
  }

  // Define variables and structures to store the evaluation data
  size_t nb_rows, nb_columns;
  std::vector<std::string> headers;
  std::vector<std::string> column_types;

  if (_verbose_level == 0) {
    headers = {"group", "success", "final_public_account", "final_round"};
    column_types = {"int", "bool", "float", "int"};
    nb_rows = nb_evaluation_games;
    nb_columns = 4;
  } else {
    headers = {"group", "player", "game_index", "round", "action", "group_contributions", "contributions_others",
               "total_contribution", "payoff", "success", "target", "final_public_account", "final_round"};
    column_types = {"int", "int", "int", "int", "int", "int", "int", "int", "int", "bool", "int", "int", "int"};
    nb_rows = nb_evaluation_games * group_size * max_rounds;
    nb_columns = 13;
  }
  // Now we create the data container
  EGTTools::RL::DataTypes::DataTableCRD data(nb_rows, nb_columns, headers, column_types, populations);

  // Then we evaluate the agents by creating randomly mixed groups
  auto total_nb_rounds = evaluate_crd_populationsTU(nb_populations,
                                                    population_size,
                                                    group_size,
                                                    nb_evaluation_games,
                                                    min_rounds,
                                                    tu,
                                                    target,
                                                    risk,
                                                    available_actions,
                                                    game,
                                                    data);

  // Finally we clear the unused rows
  if (_verbose_level > 0) {
    auto total_rows = group_size * total_nb_rounds;
    data.data.conservativeResize(total_rows, 13);
  }

  return data;
}
EGTTools::RL::DataTypes::DataTableCRD CRDSimIslands::run_population_islandsThU(size_t nb_evaluation_games,
                                                                               size_t nb_populations,
                                                                               size_t population_size,
                                                                               size_t group_size,
                                                                               size_t nb_generations,
                                                                               size_t nb_games,
                                                                               size_t nb_rounds,
                                                                               int target,
                                                                               double delta,
                                                                               int endowment,
                                                                               double risk,
                                                                               ActionSpace &available_actions,
                                                                               const std::string &agent_type,
                                                                               const std::vector<double> &args) {
  // First we instantiate a game - for now the game is always an Unconditional CRDGame
  CRDGame<PopContainer, void, void> game;

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
#pragma omp parallel for default(none) shared(populations, nb_populations, population_size, group_size, nb_generations, \
  nb_games, nb_rounds, target, delta, risk, available_actions, game)
  for (size_t population = 0; population < nb_populations; ++population) {
    run_crd_populationThU(population_size, group_size, nb_generations, nb_games, nb_rounds, target, delta,
                          risk, available_actions, populations[population], game);
  }

  // Define variables and structures to store the evaluation data
  size_t nb_rows, nb_columns;
  std::vector<std::string> headers;
  std::vector<std::string> column_types;

  if (_verbose_level == 0) {
    headers = {"group", "success", "final_public_account", "final_target"};
    column_types = {"int", "bool", "float", "float"};
    nb_rows = nb_evaluation_games;
    nb_columns = 4;
  } else {
    headers = {"group", "player", "game_index", "round", "action", "group_contributions", "contributions_others",
               "total_contribution", "payoff", "success", "target", "final_public_account", "final_round"};
    column_types = {"int", "int", "int", "int", "int", "int", "int", "int", "int", "bool", "int", "int", "int"};
    nb_rows = nb_evaluation_games * group_size * nb_rounds;
    nb_columns = 13;
  }
  // Now we create the data container
  EGTTools::RL::DataTypes::DataTableCRD data(nb_rows, nb_columns, headers, column_types, populations);

  // Then we evaluate the agents by creating randomly mixed groups
  evaluate_crd_populationsThU(nb_populations,
                              population_size,
                              group_size,
                              nb_evaluation_games,
                              nb_rounds,
                              target,
                              delta,
                              risk,
                              available_actions,
                              game,
                              data);

  return data;
}
EGTTools::RL::DataTypes::DataTableCRD CRDSimIslands::run_conditional_population_islandsThU(size_t nb_evaluation_games,
                                                                                           size_t nb_populations,
                                                                                           size_t population_size,
                                                                                           size_t group_size,
                                                                                           size_t nb_generations,
                                                                                           size_t nb_games,
                                                                                           size_t nb_rounds,
                                                                                           int target,
                                                                                           double delta,
                                                                                           int endowment,
                                                                                           double risk,
                                                                                           ActionSpace &available_actions,
                                                                                           const std::string &agent_type,
                                                                                           const std::vector<double> &args) {
  // Instantiate factored state
  // The state is composed of the (current round, count of each of the possible action)
  // However, since there is fixed number of agents, the possible combination of counts
  // are fixed and can be calculated with a binomial.
  FlattenState flatten(Factors{nb_rounds, EGTTools::starsBars(group_size, available_actions.size())});
  // First we instantiate a game - for now the game is always an Unconditional CRDGame
  CRDConditionalCount<void, void> game(flatten);

  // Create a vector of groups - nb_actions = available_actions.size()
  std::vector<PopContainer> populations;
  for (size_t i = 0; i < nb_populations; i++)
    populations.emplace_back(agent_type,
                             population_size,
                             game.flatten().factor_space,
                             available_actions.size(),
                             nb_rounds,
                             endowment,
                             args);

  // First we let all the groups adapt
#pragma omp parallel for default(none) shared(populations, nb_populations, population_size, group_size, nb_generations, \
  nb_games, nb_rounds, target, delta, risk, available_actions, game)
  for (size_t population = 0; population < nb_populations; ++population) {
    run_crd_populationThU(population_size, group_size, nb_generations, nb_games, nb_rounds, target, delta,
                          risk, available_actions, populations[population], game);
  }

  // Define variables and structures to store the evaluation data
  size_t nb_rows, nb_columns;
  std::vector<std::string> headers;
  std::vector<std::string> column_types;

  if (_verbose_level == 0) {
    headers = {"group", "success", "final_public_account", "final_target"};
    column_types = {"int", "bool", "float", "float"};
    nb_rows = nb_evaluation_games;
    nb_columns = 4;
  } else {
    headers = {"group", "player", "game_index", "round", "action", "group_contributions", "contributions_others",
               "total_contribution", "payoff", "success", "target", "final_public_account", "final_round"};
    column_types = {"int", "int", "int", "int", "int", "int", "int", "int", "int", "bool", "int", "int", "int"};
    nb_rows = nb_evaluation_games * group_size * nb_rounds;
    nb_columns = 13;
  }
  // Now we create the data container
  EGTTools::RL::DataTypes::DataTableCRD data(nb_rows, nb_columns, headers, column_types, populations);

  // Then we evaluate the agents by creating randomly mixed groups
  evaluate_crd_populationsThU(nb_populations,
                              population_size,
                              group_size,
                              nb_evaluation_games,
                              nb_rounds,
                              target,
                              delta,
                              risk,
                              available_actions,
                              game,
                              data);

  return data;
}
EGTTools::RL::DataTypes::DataTableCRD CRDSimIslands::run_population_islandsTUThU(size_t nb_evaluation_games,
                                                                                 size_t nb_populations,
                                                                                 size_t population_size,
                                                                                 size_t group_size,
                                                                                 size_t nb_generations,
                                                                                 size_t nb_games,
                                                                                 size_t min_rounds,
                                                                                 size_t mean_rounds,
                                                                                 size_t max_rounds,
                                                                                 double p,
                                                                                 int target,
                                                                                 double delta,
                                                                                 int endowment,
                                                                                 double risk,
                                                                                 ActionSpace &available_actions,
                                                                                 const std::string &agent_type,
                                                                                 const std::vector<double> &args) {
  // First we calculate the probability of the game ending after the minimum number of rounds
  if (mean_rounds > 0) {
    p = 1.0 / static_cast<double>(mean_rounds - min_rounds + 1);
  }
  EGTTools::TimingUncertainty<std::mt19937_64> tu(p, max_rounds);
  // Then we instantiate a game with uncertainty
  CRDGame <PopContainer, EGTTools::TimingUncertainty<std::mt19937_64>, std::mt19937_64> game;

  // Create a vector of groups - nb_actions = available_actions.size()
  std::vector<PopContainer> populations;
  for (size_t i = 0; i < nb_populations; i++)
    populations.emplace_back(agent_type,
                             population_size,
                             max_rounds,
                             available_actions.size(),
                             max_rounds,
                             endowment,
                             args);

  // First we let all the groups adapt
#pragma omp parallel for default(none) shared(populations, tu, nb_populations, population_size, group_size, nb_generations, \
  nb_games, min_rounds, target, delta, risk, available_actions, game)
  for (size_t population = 0; population < nb_populations; ++population) {
    run_crd_populationTUThU(population_size, group_size, nb_generations, nb_games, min_rounds, tu, target, delta,
                            risk, available_actions, populations[population], game);
  }

  // Define variables and structures to store the evaluation data
  size_t nb_rows, nb_columns;
  std::vector<std::string> headers;
  std::vector<std::string> column_types;

  if (_verbose_level == 0) {
    headers = {"group", "success", "final_public_account", "final_round", "final_target"};
    column_types = {"int", "bool", "float", "int", "float"};
    nb_rows = nb_evaluation_games;
    nb_columns = 5;
  } else {
    headers = {"group", "player", "game_index", "round", "action", "group_contributions", "contributions_others",
               "total_contribution", "payoff", "success", "target", "final_public_account", "final_round"};
    column_types = {"int", "int", "int", "int", "int", "int", "int", "int", "int", "bool", "int", "int", "int"};
    nb_rows = nb_evaluation_games * group_size * max_rounds;
    nb_columns = 13;
  }
  // Now we create the data container
  EGTTools::RL::DataTypes::DataTableCRD data(nb_rows, nb_columns, headers, column_types, populations);

  // Then we evaluate the agents by creating randomly mixed groups
  auto total_nb_rounds = evaluate_crd_populationsTUThU(nb_populations,
                                                       population_size,
                                                       group_size,
                                                       nb_evaluation_games,
                                                       min_rounds,
                                                       tu,
                                                       target,
                                                       delta,
                                                       risk,
                                                       available_actions,
                                                       game,
                                                       data);

  // Finally we clear the unused rows
  if (_verbose_level > 0) {
    auto total_rows = group_size * total_nb_rounds;
    data.data.conservativeResize(total_rows, 13);
  }

  return data;
}
EGTTools::RL::DataTypes::DataTableCRD CRDSimIslands::run_conditional_population_islandsTUThU(size_t nb_evaluation_games,
                                                                                             size_t nb_populations,
                                                                                             size_t population_size,
                                                                                             size_t group_size,
                                                                                             size_t nb_generations,
                                                                                             size_t nb_games,
                                                                                             size_t min_rounds,
                                                                                             size_t mean_rounds,
                                                                                             size_t max_rounds,
                                                                                             double p,
                                                                                             int target,
                                                                                             double delta,
                                                                                             int endowment,
                                                                                             double risk,
                                                                                             ActionSpace &available_actions,
                                                                                             const std::string &agent_type,
                                                                                             const std::vector<double> &args) {
  // First we calculate the probability of the game ending after the minimum number of rounds
  if (mean_rounds > 0) {
    p = 1.0 / static_cast<double>(mean_rounds - min_rounds + 1);
  }
  EGTTools::TimingUncertainty<std::mt19937_64> tu(p, max_rounds);
  // Instantiate factored state
  // The state is composed of the (current round, nb of times action 0 selected, ..., nb times action n selected)
  // The state space depends on the number of actions
  FlattenState flatten(Factors{max_rounds, EGTTools::starsBars(group_size, available_actions.size())});
  // First we instantiate a game - for now the game is always an Unconditional CRDGame
  CRDConditionalCount <EGTTools::TimingUncertainty<std::mt19937_64>, std::mt19937_64> game(flatten);

  // Create a vector of groups - nb_actions = available_actions.size()
  std::vector<PopContainer> populations;
  for (size_t i = 0; i < nb_populations; i++)
    populations.emplace_back(agent_type,
                             population_size,
                             game.flatten().factor_space,
                             available_actions.size(),
                             max_rounds,
                             endowment,
                             args);

  // First we let all the groups adapt
#pragma omp parallel for default(none) shared(populations, tu, nb_populations, population_size, group_size, nb_generations, \
  nb_games, min_rounds, target, delta, risk, available_actions, game)
  for (size_t population = 0; population < nb_populations; ++population) {
    run_crd_populationTUThU(population_size, group_size, nb_generations, nb_games, min_rounds, tu, target, delta,
                            risk, available_actions, populations[population], game);
  }

  // Define variables and structures to store the evaluation data
  size_t nb_rows, nb_columns;
  std::vector<std::string> headers;
  std::vector<std::string> column_types;

  if (_verbose_level == 0) {
    headers = {"group", "success", "final_public_account", "final_round", "final_target"};
    column_types = {"int", "bool", "float", "int", "float"};
    nb_rows = nb_evaluation_games;
    nb_columns = 5;
  } else {
    headers = {"group", "player", "game_index", "round", "action", "group_contributions", "contributions_others",
               "total_contribution", "payoff", "success", "target", "final_public_account", "final_round"};
    column_types = {"int", "int", "int", "int", "int", "int", "int", "int", "int", "bool", "int", "int", "int"};
    nb_rows = nb_evaluation_games * group_size * max_rounds;
    nb_columns = 13;
  }
  // Now we create the data container
  EGTTools::RL::DataTypes::DataTableCRD data(nb_rows, nb_columns, headers, column_types, populations);

  // Then we evaluate the agents by creating randomly mixed groups
  auto total_nb_rounds = evaluate_crd_populationsTUThU(nb_populations,
                                                       population_size,
                                                       group_size,
                                                       nb_evaluation_games,
                                                       min_rounds,
                                                       tu,
                                                       target,
                                                       delta,
                                                       risk,
                                                       available_actions,
                                                       game,
                                                       data);

  // Finally we clear the unused rows
  if (_verbose_level > 0) {
    auto total_rows = group_size * total_nb_rounds;
    data.data.conservativeResize(total_rows, 13);
  }

  return data;
}

size_t CRDSimIslands::verbose_level() const { return _verbose_level; }

void CRDSimIslands::set_verbose_level(size_t verbose_level) {
  if (verbose_level > 1) throw std::invalid_argument("Verbose level can only be set to 0, 1.");
  _verbose_level = verbose_level;
}

void CRDSimIslands::set_learning_rate_decay(double decay, double min_lr) {
  _decay = decay;
  _min_learning_rate = min_lr;
}