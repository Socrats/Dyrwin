//
// Created by Elias Fernandez on 10/02/2020.
//
#include <Dyrwin/RL/simulators/CrdIslands.h>

using namespace EGTTools::RL::CRD::Simulators;

CRDSimIslands::CRDSimIslands() {
  _real_rand = std::uniform_real_distribution<double>(0.0, 1.0);
}
template<class G>
void CRDSimIslands::reinforce_population(double &pool,
                                         size_t &success,
                                         double target,
                                         double &risk,
                                         PopContainer &pop,
                                         size_t &final_round,
                                         G &game,
                                         std::mt19937_64 &generator) {
  if (pool >= target)
    success++;
  else if (_real_rand(generator) < risk)
    game.setPayoffs(pop, 0);

  game.reinforcePath(pop, final_round);
}
template<class G>
void CRDSimIslands::reinforce_population(double &pool,
                                         size_t &success,
                                         double threshold,
                                         double &risk,
                                         EGTTools::RL::PopContainer &pop,
                                         G &game,
                                         std::mt19937_64 &generator) {
  if (pool >= threshold)
    success++;
  else if (_real_rand(generator) < risk)
    game.setPayoffs(pop, 0);

  game.reinforcePath(pop);
}

EGTTools::RL::DataTypes::CRDDataIslands
CRDSimIslands::run_group_islands(size_t nb_groups,
                                 size_t group_size,
                                 size_t nb_generations,
                                 size_t nb_games,
                                 size_t nb_rounds,
                                 int target,
                                 int endowment,
                                 double risk,
                                 const EGTTools::RL::ActionSpace &available_actions,
                                 const std::string &agent_type,
                                 const std::vector<double> &args) {
  return EGTTools::RL::DataTypes::CRDDataIslands();
}
EGTTools::RL::DataTypes::CRDDataIslands CRDSimIslands::run_group_islandsTU(size_t nb_groups,
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
                                                                           const EGTTools::RL::ActionSpace &available_actions,
                                                                           const std::string &agent_type,
                                                                           const std::vector<double> &args) {
  return EGTTools::RL::DataTypes::CRDDataIslands();
}
EGTTools::RL::DataTypes::CRDDataIslands
CRDSimIslands::run_group_islandsThU(size_t nb_groups,
                                    size_t group_size,
                                    size_t nb_generations,
                                    size_t nb_games,
                                    size_t nb_rounds,
                                    int target,
                                    int delta,
                                    int endowment,
                                    double risk,
                                    const EGTTools::RL::ActionSpace &available_actions,
                                    const std::string &agent_type,
                                    const std::vector<double> &args) {
  return EGTTools::RL::DataTypes::CRDDataIslands();
}
EGTTools::RL::DataTypes::CRDDataIslands
CRDSimIslands::run_group_islandsTUThU(size_t nb_groups,
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
                                      const EGTTools::RL::ActionSpace &available_actions,
                                      const std::string &agent_type,
                                      const std::vector<double> &args) {
  return EGTTools::RL::DataTypes::CRDDataIslands();
}
EGTTools::RL::DataTypes::CRDDataIslands
CRDSimIslands::run_population_islands(size_t nb_populations,
                                      size_t population_size,
                                      size_t group_size,
                                      size_t nb_generations,
                                      size_t nb_games,
                                      size_t nb_rounds,
                                      int target,
                                      int endowment,
                                      double risk,
                                      const ActionSpace &available_actions,
                                      const std::string &agent_type,
                                      const std::vector<double> &args) {
  return EGTTools::RL::DataTypes::CRDDataIslands();
}
