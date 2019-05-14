//
// Created by Elias Fernandez on 2019-05-14.
//

#include <Dyrwin/RL/CrdSim.hpp>

EGTTools::RL::CRDSim::CRDSim(size_t nb_episodes, size_t nb_games, size_t nb_rounds,
                             size_t nb_actions, size_t group_size,
                             double risk, double endowment,
                             double threshold,
                             const ActionSpace &available_actions,
                             const std::string &agent_type,
                             const std::vector<double> &args) : _nb_episodes(nb_episodes),
                                                                _nb_games(nb_games),
                                                                _nb_rounds(nb_rounds),
                                                                _nb_actions(nb_actions),
                                                                _group_size(group_size),
                                                                _risk(risk),
                                                                _threshold(threshold),
                                                                _endowment(endowment) {
    if (available_actions.size() != _nb_actions)
        throw std::invalid_argument("you can't specify more actions than " + std::to_string(_nb_actions));

    _available_actions = ActionSpace(nb_actions);
    for (size_t i = 0; i < _nb_actions; ++i) _available_actions[i] = available_actions[i];
    try {
        population = PopContainer(agent_type, group_size, nb_rounds, nb_actions, nb_rounds, _endowment, args);
    } catch (std::invalid_argument &e) {
        throw e;
    }
    if (agent_type == "rothErev") _reinforce = &EGTTools::RL::CRDSim::reinforceOnlyPositive;
    else _reinforce = &EGTTools::RL::CRDSim::reinforceAll;
}

EGTTools::Matrix2D EGTTools::RL::CRDSim::run(size_t nb_episodes, size_t nb_games) {
    Matrix2D results = Matrix2D::Zero(2, nb_episodes);

    for (size_t step = 0; step < nb_episodes; ++step) {
        size_t success = 0;
        double avg_contribution = 0.;
        double avg_rounds = 0.;
        for (unsigned int game = 0; game < nb_games; ++game) {
            // First we play the game
            auto[pool, final_round] = Game.playGame(population, _available_actions, _nb_rounds);
            avg_contribution += (Game.playersContribution(population) / double(_group_size));
            (this->*_reinforce)(pool, success);
            avg_rounds += final_round;
        }
        results(0, step) = static_cast<double>(success) / static_cast<double>(nb_games);
        results(1, step) = static_cast<double>(avg_contribution) / static_cast<double>(nb_games);

        Game.calcProbabilities(population);
        Game.resetEpisode(population);
    }

    return results;
}

EGTTools::Matrix2D EGTTools::RL::CRDSim::run(size_t nb_episodes, size_t nb_games, size_t nb_groups) {
    Matrix2D results = Matrix2D::Zero(2, nb_episodes);

    for (size_t group = 0; group < nb_groups; ++group) {
        // First reset population
        population.reset();

        for (size_t step = 0; step < nb_episodes; ++step) {
            size_t success = 0;
            double avg_contribution = 0.;
            double avg_rounds = 0.;
            for (unsigned int game = 0; game < nb_games; ++game) {
                // First we play the game
                auto[pool, final_round] = Game.playGame(population, _available_actions, _nb_rounds);
                avg_contribution += (Game.playersContribution(population) / double(_group_size));
                (this->*_reinforce)(pool, success);
                avg_rounds += final_round;
            }
            results(0, step) += static_cast<double>(success) / static_cast<double>(nb_games);
            results(1, step) += static_cast<double>(avg_contribution) / static_cast<double>(nb_games);

            Game.calcProbabilities(population);
            Game.resetEpisode(population);
        }
    }

    return results / nb_groups;

}

void EGTTools::RL::CRDSim::resetPopulation() { population.reset(); }

void EGTTools::RL::CRDSim::reinforceOnlyPositive(double &pool, size_t &success) {
    if (pool >= _threshold) {
        Game.reinforcePath(population);
        success++;
    } else if (EGTTools::probabilityDistribution(_generator) > _risk) Game.reinforcePath(population);
    else Game.setPayoffs(population, 0);
}

void EGTTools::RL::CRDSim::reinforceAll(double &pool, size_t &success) {

    if (pool >= _threshold) success++;
    else if (EGTTools::probabilityDistribution(_generator) < _risk) Game.setPayoffs(population, 0);

    Game.reinforcePath(population);
}

void EGTTools::RL::CRDSim::reinforceXico(double &pool, size_t &success) {

    if (pool >= _threshold) success++;
    else if (EGTTools::probabilityDistribution(_generator) < _risk) {
        for (auto &player: population) {
            player->set_payoff(-player->payoff());
        }
    }

    Game.reinforcePath(population);
}

size_t EGTTools::RL::CRDSim::nb_games() const { return _nb_games; }

size_t EGTTools::RL::CRDSim::nb_episodes() const { return _nb_episodes; }

size_t EGTTools::RL::CRDSim::nb_rounds() const { return _nb_rounds; }

size_t EGTTools::RL::CRDSim::nb_actions() const { return _nb_actions; }

double EGTTools::RL::CRDSim::endowment() const { return _endowment; }

double EGTTools::RL::CRDSim::risk() const { return _risk; }

double EGTTools::RL::CRDSim::threshold() const { return _threshold; }

const EGTTools::RL::ActionSpace & EGTTools::RL::CRDSim::available_actions() const { return _available_actions; }

void EGTTools::RL::CRDSim::set_nb_games(size_t nb_games) { _nb_games = nb_games; }

void EGTTools::RL::CRDSim::set_nb_episodes(size_t nb_episodes) { _nb_episodes = nb_episodes; }

void EGTTools::RL::CRDSim::set_nb_rounds(size_t nb_rounds) {
    _nb_rounds = nb_rounds;
    for (auto &individual: population) {
        individual->set_nb_states(_nb_rounds);
    }
}

void EGTTools::RL::CRDSim::set_nb_actions(size_t nb_actions) { _nb_actions = nb_actions; }

void EGTTools::RL::CRDSim::set_risk(double risk) { _risk = risk; }

void EGTTools::RL::CRDSim::set_threshold(double threshold) { _threshold = threshold; }

void EGTTools::RL::CRDSim::set_available_actions(const EGTTools::RL::ActionSpace &available_actions) {
    if (available_actions.size() != _nb_actions)
        throw std::invalid_argument("you can't specify more actions than " + std::to_string(_nb_actions));
    _available_actions.resize(available_actions.size());
    for (size_t i = 0; i < available_actions.size(); ++i)
        _available_actions[i] = available_actions[i];
}
