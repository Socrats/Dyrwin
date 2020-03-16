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

  // Now we create the data container
  // It will contain a data matrix/table of 10 columns and
  // nb_evaluation_games * group_size * nb_rounds
  std::vector<std::string> headers =
      {"group", "player", "game_index", "round", "action", "group_contributions", "contributions_others",
       "total_contribution", "payoff", "success", "target", "final_public_account", "final_round"};
  std::vector<std::string>
      column_types = {"int", "int", "int", "int", "int", "int", "int", "int", "int", "bool", "int", "int", "int"};
  EGTTools::RL::DataTypes::DataTableCRD
      data(nb_evaluation_games * group_size * nb_rounds, 13,
           headers, column_types, groups);

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
  // The first dimension indicate the round of the game,
  // and the second the donations of the group in the previous round (maximum group_size * (nb_action - 1) + 1)
//  FlattenState flatten(Factors{nb_rounds, (group_size * (available_actions.size() - 1)) + 1});
  // The state is composed of the (current round, nb of times action 0 selected, ..., nb times action n selected)
  // The state space depends on the number of actions
  Factors state_space = Factors(available_actions.size() + 1, group_size + 1);
  state_space[0] = nb_rounds;
  FlattenState flatten(state_space);
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

  // Now we create the data container
  // It will contain a data matrix/table of 10 columns and
  // nb_evaluation_games * group_size * nb_rounds
  std::vector<std::string> headers =
      {"group", "player", "game_index", "round", "action", "group_contributions", "contributions_others",
       "total_contribution", "payoff", "success", "target", "final_public_account", "final_round"};
  std::vector<std::string>
      column_types = {"int", "int", "int", "int", "int", "int", "int", "int", "int", "bool", "int", "int", "int"};
  EGTTools::RL::DataTypes::DataTableCRD
      data(nb_evaluation_games * group_size * nb_rounds, 13,
           headers, column_types, groups);

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

  // Now we create the data container
  // It will contain a data matrix/table of 10 columns and
  // nb_evaluation_games * group_size * nb_rounds
  std::vector<std::string> headers =
      {"group", "player", "game_index", "round", "action", "group_contributions", "contributions_others",
       "total_contribution", "payoff", "success", "target", "final_public_account", "final_round"};
  std::vector<std::string>
      column_types = {"int", "int", "int", "int", "int", "int", "int", "int", "int", "bool", "int", "int", "int"};
  EGTTools::RL::DataTypes::DataTableCRD
      data(nb_evaluation_games * group_size * max_rounds, 13,
           headers, column_types, groups);

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
  auto total_rows = group_size * total_nb_rounds;
  data.data.conservativeResize(total_rows, 13);

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
  Factors state_space = Factors(available_actions.size() + 1, group_size + 1);
  state_space[0] = max_rounds;
  FlattenState flatten(state_space);
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

  // Now we create the data container
  // It will contain a data matrix/table of 10 columns and
  // nb_evaluation_games * group_size * nb_rounds
  std::vector<std::string> headers =
      {"group", "player", "game_index", "round", "action", "group_contributions", "contributions_others",
       "total_contribution", "payoff", "success", "target", "final_public_account", "final_round"};
  std::vector<std::string>
      column_types = {"int", "int", "int", "int", "int", "int", "int", "int", "int", "bool", "int", "int", "int"};
  EGTTools::RL::DataTypes::DataTableCRD
      data(nb_evaluation_games * group_size * max_rounds, 13,
           headers, column_types, groups);

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
  auto total_rows = group_size * total_nb_rounds;
  data.data.conservativeResize(total_rows, 13);

  return data;
}
EGTTools::RL::DataTypes::DataTableCRD CRDSimIslands::run_group_islandsThU(size_t nb_evaluation_games,
                                                                          size_t nb_groups,
                                                                          size_t group_size,
                                                                          size_t nb_generations,
                                                                          size_t nb_games,
                                                                          size_t nb_rounds,
                                                                          int target,
                                                                          int delta,
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

  // Now we create the data container
  // It will contain a data matrix/table of 10 columns and
  // nb_evaluation_games * group_size * nb_rounds
  std::vector<std::string> headers =
      {"group", "player", "game_index", "round", "action", "group_contributions", "contributions_others",
       "total_contribution", "payoff", "success", "target", "final_public_account", "final_round"};
  std::vector<std::string>
      column_types = {"int", "int", "int", "int", "int", "int", "int", "int", "int", "bool", "int", "int", "int"};
  EGTTools::RL::DataTypes::DataTableCRD
      data(nb_evaluation_games * group_size * nb_rounds, 13,
           headers, column_types, groups);

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
                                                                                      int delta,
                                                                                      int endowment,
                                                                                      double risk,
                                                                                      ActionSpace &available_actions,
                                                                                      const std::string &agent_type,
                                                                                      const std::vector<double> &args) {
  // Instantiate factored state
  // The first dimension indicate the round of the game,
  // and the second the donations of the group in the previous round (maximum group_size * (nb_action - 1) + 1)
//  FlattenState flatten(Factors{nb_rounds, (group_size * (available_actions.size() - 1)) + 1});
  // The state is composed of the (current round, nb of times action 0 selected, ..., nb times action n selected)
  // The state space depends on the number of actions
  Factors state_space = Factors(available_actions.size() + 1, group_size + 1);
  state_space[0] = nb_rounds;
  FlattenState flatten(state_space);
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

  // Now we create the data container
  // It will contain a data matrix/table of 10 columns and
  // nb_evaluation_games * group_size * nb_rounds
  std::vector<std::string> headers =
      {"group", "player", "game_index", "round", "action", "group_contributions", "contributions_others",
       "total_contribution", "payoff", "success", "target", "final_public_account", "final_round"};
  std::vector<std::string>
      column_types = {"int", "int", "int", "int", "int", "int", "int", "int", "int", "bool", "int", "int", "int"};
  EGTTools::RL::DataTypes::DataTableCRD
      data(nb_evaluation_games * group_size * nb_rounds, 13,
           headers, column_types, groups);

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
                                                                            int delta,
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

  // Now we create the data container
  // It will contain a data matrix/table of 10 columns and
  // nb_evaluation_games * group_size * nb_rounds
  std::vector<std::string> headers =
      {"group", "player", "game_index", "round", "action", "group_contributions", "contributions_others",
       "total_contribution", "payoff", "success", "target", "final_public_account", "final_round"};
  std::vector<std::string>
      column_types = {"int", "int", "int", "int", "int", "int", "int", "int", "int", "bool", "int", "int", "int"};
  EGTTools::RL::DataTypes::DataTableCRD
      data(nb_evaluation_games * group_size * max_rounds, 13,
           headers, column_types, groups);

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
  auto total_rows = group_size * total_nb_rounds;
  data.data.conservativeResize(total_rows, 13);

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
                                                                                        int delta,
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
  // The first dimension indicate the round of the game,
  // and the second the donations of the group in the previous round (maximum group_size * (nb_action - 1) + 1)
//  FlattenState flatten(Factors{max_rounds, (group_size * (available_actions.size() - 1)) + 1});
  // The state is composed of the (current round, nb of times action 0 selected, ..., nb times action n selected)
  // The state space depends on the number of actions
  Factors state_space = Factors(available_actions.size() + 1, group_size + 1);
  state_space[0] = max_rounds;
  FlattenState flatten(state_space);
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

  // Now we create the data container
  // It will contain a data matrix/table of 10 columns and
  // nb_evaluation_games * group_size * nb_rounds
  std::vector<std::string> headers =
      {"group", "player", "game_index", "round", "action", "group_contributions", "contributions_others",
       "total_contribution", "payoff", "success", "target", "final_public_account", "final_round"};
  std::vector<std::string>
      column_types = {"int", "int", "int", "int", "int", "int", "int", "int", "int", "bool", "int", "int", "int"};
  EGTTools::RL::DataTypes::DataTableCRD
      data(nb_evaluation_games * group_size * max_rounds, 13,
           headers, column_types, groups);

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
  auto total_rows = group_size * total_nb_rounds;
  data.data.conservativeResize(total_rows, 13);

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

  // Now we create the data container
  // It will contain a data matrix/table of 10 columns and
  // nb_evaluation_games * group_size * nb_rounds
  std::vector<std::string> headers =
      {"group", "player", "game_index", "round", "action", "group_contributions", "contributions_others",
       "total_contribution", "payoff", "success", "target", "final_public_account", "final_round"};
  std::vector<std::string>
      column_types = {"int", "int", "int", "int", "int", "int", "int", "int", "int", "bool", "int", "int", "int"};
  EGTTools::RL::DataTypes::DataTableCRD
      data(nb_evaluation_games * group_size * nb_rounds, 13,
           headers, column_types, populations);

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
  // The first dimension indicate the round of the game,
  // and the second the donations of the group in the previous round (maximum group_size * (nb_action - 1) + 1)
//  FlattenState flatten(Factors{nb_rounds, (group_size * (available_actions.size() - 1)) + 1});
  // The state is composed of the (current round, nb of times action 0 selected, ..., nb times action n selected)
  // The state space depends on the number of actions
  Factors state_space = Factors(available_actions.size() + 1, group_size + 1);
  state_space[0] = nb_rounds;
  FlattenState flatten(state_space);
  // First we instantiate a game - for now the game is always an Unconditional CRDGame
  CRDConditionalCount<void, void> game(flatten);

  // Create a vector of groups - nb_actions = available_actions.size()
  std::vector <PopContainer> populations;
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

  // Now we create the data container
  // It will contain a data matrix/table of 10 columns and
  // nb_evaluation_games * group_size * nb_rounds
  std::vector <std::string> headers =
      {"group", "player", "game_index", "round", "action", "group_contributions", "contributions_others",
       "total_contribution", "payoff", "success", "target", "final_public_account", "final_round"};
  std::vector <std::string>
      column_types =
      {"int", "int", "int", "int", "float", "float", "float", "float", "float", "bool", "float", "float", "int"};
  EGTTools::RL::DataTypes::DataTableCRD
      data(nb_evaluation_games * group_size * nb_rounds, 13,
           headers, column_types, populations);

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

  // Now we create the data container
  // It will contain a data matrix/table of 10 columns and
  // nb_evaluation_games * group_size * nb_rounds
  std::vector<std::string> headers =
      {"group", "player", "game_index", "round", "action", "group_contributions", "contributions_others",
       "total_contribution", "payoff", "success", "target", "final_public_account", "final_round"};
  std::vector<std::string>
      column_types = {"int", "int", "int", "int", "int", "int", "int", "int", "int", "bool", "int", "int", "int"};
  EGTTools::RL::DataTypes::DataTableCRD
      data(nb_evaluation_games * group_size * max_rounds, 13,
           headers, column_types, populations);

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
  auto total_rows = group_size * total_nb_rounds;
  data.data.conservativeResize(total_rows, 13);

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
  // The first dimension indicate the round of the game,
  // and the second the donations of the group in the previous round (maximum group_size * (nb_action - 1) + 1)
//  FlattenState flatten(Factors{max_rounds, (group_size * (available_actions.size() - 1)) + 1});
  // The state is composed of the (current round, nb of times action 0 selected, ..., nb times action n selected)
  // The state space depends on the number of actions
  Factors state_space = Factors(available_actions.size() + 1, group_size + 1);
  state_space[0] = max_rounds;
  FlattenState flatten(state_space);
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

  // Now we create the data container
  // It will contain a data matrix/table of 10 columns and
  // nb_evaluation_games * group_size * nb_rounds
  std::vector<std::string> headers =
      {"group", "player", "game_index", "round", "action", "group_contributions", "contributions_others",
       "total_contribution", "payoff", "success", "target", "final_public_account", "final_round"};
  std::vector<std::string>
      column_types = {"int", "int", "int", "int", "int", "int", "int", "int", "int", "bool", "int", "int", "int"};
  EGTTools::RL::DataTypes::DataTableCRD
      data(nb_evaluation_games * group_size * max_rounds, 13,
           headers, column_types, populations);

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
  auto total_rows = group_size * total_nb_rounds;
  data.data.conservativeResize(total_rows, 13);

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
                                                                               int delta,
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

  // Now we create the data container
  // It will contain a data matrix/table of 10 columns and
  // nb_evaluation_games * group_size * nb_rounds
  std::vector<std::string> headers =
      {"group", "player", "game_index", "round", "action", "group_contributions", "contributions_others",
       "total_contribution", "payoff", "success", "target", "final_public_account", "final_round"};
  std::vector<std::string>
      column_types = {"int", "int", "int", "int", "int", "int", "int", "int", "int", "bool", "int", "int", "int"};
  EGTTools::RL::DataTypes::DataTableCRD
      data(nb_evaluation_games * group_size * nb_rounds, 13,
           headers, column_types, populations);

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
                                                                                           int delta,
                                                                                           int endowment,
                                                                                           double risk,
                                                                                           ActionSpace &available_actions,
                                                                                           const std::string &agent_type,
                                                                                           const std::vector<double> &args) {
  // Instantiate factored state
  // The first dimension indicate the round of the game,
  // and the second the donations of the group in the previous round (maximum group_size * (nb_action - 1) + 1)
//  FlattenState flatten(Factors{nb_rounds, (group_size * (available_actions.size() - 1)) + 1});
  // The state is composed of the (current round, nb of times action 0 selected, ..., nb times action n selected)
  // The state space depends on the number of actions
  Factors state_space = Factors(available_actions.size() + 1, group_size + 1);
  state_space[0] = nb_rounds;
  FlattenState flatten(state_space);
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

  // Now we create the data container
  // It will contain a data matrix/table of 10 columns and
  // nb_evaluation_games * group_size * nb_rounds
  std::vector<std::string> headers =
      {"group", "player", "game_index", "round", "action", "group_contributions", "contributions_others",
       "total_contribution", "payoff", "success", "target", "final_public_account", "final_round"};
  std::vector<std::string>
      column_types = {"int", "int", "int", "int", "int", "int", "int", "int", "int", "bool", "int", "int", "int"};
  EGTTools::RL::DataTypes::DataTableCRD
      data(nb_evaluation_games * group_size * nb_rounds, 13,
           headers, column_types, populations);

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
                                                                                 int delta,
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

  // Now we create the data container
  // It will contain a data matrix/table of 10 columns and
  // nb_evaluation_games * group_size * nb_rounds
  std::vector<std::string> headers =
      {"group", "player", "game_index", "round", "action", "group_contributions", "contributions_others",
       "total_contribution", "payoff", "success", "target", "final_public_account", "final_round"};
  std::vector<std::string>
      column_types = {"int", "int", "int", "int", "int", "int", "int", "int", "int", "bool", "int", "int", "int"};
  EGTTools::RL::DataTypes::DataTableCRD
      data(nb_evaluation_games * group_size * max_rounds, 13,
           headers, column_types, populations);

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
  auto total_rows = group_size * total_nb_rounds;
  data.data.conservativeResize(total_rows, 13);

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
                                                                                             int delta,
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
  // The first dimension indicate the round of the game,
  // and the second the donations of the group in the previous round (maximum group_size * (nb_action - 1) + 1)
//  FlattenState flatten(Factors{max_rounds, (group_size * (available_actions.size() - 1)) + 1});
  // The state is composed of the (current round, nb of times action 0 selected, ..., nb times action n selected)
  // The state space depends on the number of actions
  Factors state_space = Factors(available_actions.size() + 1, group_size + 1);
  state_space[0] = max_rounds;
  FlattenState flatten(state_space);
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

  // Now we create the data container
  // It will contain a data matrix/table of 10 columns and
  // nb_evaluation_games * group_size * nb_rounds
  std::vector<std::string> headers =
      {"group", "player", "game_index", "round", "action", "group_contributions", "contributions_others",
       "total_contribution", "payoff", "success", "target", "final_public_account", "final_round"};
  std::vector<std::string>
      column_types = {"int", "int", "int", "int", "int", "int", "int", "int", "int", "bool", "int", "int", "int"};
  EGTTools::RL::DataTypes::DataTableCRD
      data(nb_evaluation_games * group_size * max_rounds, 13,
           headers, column_types, populations);

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
  auto total_rows = group_size * total_nb_rounds;
  data.data.conservativeResize(total_rows, 13);

  return data;
}