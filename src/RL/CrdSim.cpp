//
// Created by Elias Fernandez on 2019-05-14.
//

#include <Dyrwin/RL/CrdSim.hpp>
//#include <omp.h>

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
                                                                _endowment(endowment),
                                                                _threshold(threshold),
                                                                _agent_type(agent_type) {
    if (available_actions.size() != _nb_actions)
        throw std::invalid_argument("you can't specify more actions than " + std::to_string(_nb_actions));
    if (available_actions[0] != 0)
        throw std::invalid_argument("First action must always be 0! And actions must be in crescent order!");
    for (size_t i = 0; i < _nb_actions; ++i)
        for (size_t j = i + 1; j < _nb_actions - 1; ++j)
            if (available_actions[i] >= available_actions[j])
                throw std::invalid_argument(
                        "Actions must always be monotonically increasing! There can't be two equal actions!");

    _available_actions = ActionSpace(nb_actions);

    for (size_t i = 0; i < _nb_actions; ++i) _available_actions[i] = available_actions[i];
    try {
        population = PopContainer(agent_type, group_size, nb_rounds, nb_actions, nb_rounds, _endowment, args);
    } catch (std::invalid_argument &e) {
        throw e;
    }
    if (agent_type == "rothErev") _reinforce = &EGTTools::RL::CRDSim::reinforceOnlyPositive;
    else _reinforce = &EGTTools::RL::CRDSim::reinforceAll;

    _real_rand = std::uniform_real_distribution<double>(0.0, 1.0);
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
            (this->*_reinforce)(pool, success, _risk, population, Game);
            avg_rounds += final_round;
        }
        results(0, step) = static_cast<double>(success) / static_cast<double>(nb_games);
        results(1, step) = static_cast<double>(avg_contribution) / static_cast<double>(nb_games);

        Game.calcProbabilities(population);
        Game.resetEpisode(population);
    }

    return results;
}

EGTTools::Matrix2D
EGTTools::RL::CRDSim::run(size_t nb_episodes, size_t nb_games, size_t nb_groups, double risk,
                          const std::vector<double> &args) {
    EGTTools::Matrix2D results = Matrix2D::Zero(2, nb_groups);
    EGTTools::Vector group_achievement = Vector::Zero(nb_groups);
    EGTTools::Vector avg_donations = Vector::Zero(nb_groups);
    size_t transient = (nb_episodes > 100) ? nb_episodes - 100 : 0;

    // Create a vector of groups
    std::vector<PopContainer> groups;

    for (size_t i = 0; i < nb_groups; ++i) {
        try {
            groups.emplace_back(_agent_type, _group_size, _nb_rounds, _nb_actions, _nb_rounds, _endowment, args);
        } catch (std::invalid_argument &e) {
            throw e;
        }
    }

#pragma omp parallel for shared(transient, groups) reduction(+:group_achievement, avg_donations)
    for (size_t group = 0; group < nb_groups; ++group) {
        size_t success;
        double avg_contribution;
        double avg_rounds;
        CRDGame<PopContainer> game;

        for (size_t step = 0; step < nb_episodes; ++step) {
            success = 0;
            avg_contribution = 0.;
            avg_rounds = 0.;
            for (unsigned int i = 0; i < nb_games; ++i) {
                // First we play the game
                auto[pool, final_round] = game.playGame(groups[group], _available_actions, _nb_rounds);
                avg_contribution += (game.playersContribution(groups[group]) / double(_group_size));
                (this->*_reinforce)(pool, success, risk, groups[group], game);
                avg_rounds += final_round;
            }
            if (step >= transient) {
                group_achievement(group) += static_cast<double>(success) / static_cast<double>(nb_games);
                avg_donations(group) += avg_contribution / static_cast<double>(nb_games);
            }

            game.calcProbabilities(groups[group]);
            game.resetEpisode(groups[group]);
        }
    }

    results.row(0) = group_achievement;
    results.row(1) = avg_donations;

    results.array() /= static_cast<double>(nb_episodes - transient);

    return results;

}

EGTTools::Matrix2D
EGTTools::RL::CRDSim::runWellMixed(size_t nb_generations, size_t nb_games, size_t nb_groups, size_t group_size,
                                   double risk,
                                   const std::vector<double> &args) {
    EGTTools::Matrix2D results = Matrix2D::Zero(2, nb_generations);
    size_t pop_size = group_size * nb_groups;
    size_t success;
    double avg_contribution;
    double avg_rounds;
    CRDGame<PopContainer> game;

    std::mt19937_64 mt{EGTTools::Random::SeedGenerator::getInstance().getSeed()};

    // Create a population of _group_size * nb_groups
    PopContainer wmPop(_agent_type, pop_size, _nb_rounds, _nb_actions, _nb_rounds, _endowment, args);
    PopContainer group;
    std::vector<size_t> groups(pop_size);
    std::iota(groups.begin(), groups.end(), 0);
    for (size_t i = 0; i < group_size; ++i)
        group.push_back(wmPop(i));

    for (size_t generation = 0; generation < nb_generations; ++generation) {
        // First we select random groups and let them play nb_games
        success = 0;
        avg_contribution = 0.;
        avg_rounds = 0.;
        for (size_t i = 0; i < nb_games; ++i) {
            std::shuffle(groups.begin(), groups.end(), mt);
            for (size_t j = 0; i < group_size; ++i)
                group(j) = wmPop(groups[j]);
            // First we play the game
            auto[pool, final_round] = game.playGame(group, _available_actions, _nb_rounds);
            avg_contribution += (game.playersContribution(group) / double(group_size));
            (this->*_reinforce)(pool, success, risk, group, game);
            avg_rounds += final_round;
        }
        results(0, generation) += static_cast<double>(success) / static_cast<double>(nb_games);
        results(1, generation) += avg_contribution / static_cast<double>(nb_games);

        game.calcProbabilities(wmPop);
        game.resetEpisode(wmPop);
    }

    return results;

}

EGTTools::Matrix2D
EGTTools::RL::CRDSim::runWellMixed(size_t nb_runs, size_t nb_generations, size_t nb_games, size_t nb_groups,
                                   size_t group_size,
                                   double risk, const std::vector<double> &args) {
    EGTTools::Matrix2D results = Matrix2D::Zero(2, nb_runs);
    size_t transient = nb_generations > 100 ? nb_generations - 100 : 0;

#pragma omp parallel for shared(transient, results)
    for (size_t run = 0; run < nb_runs; ++run) {
        EGTTools::Matrix2D tmp = runWellMixed(nb_generations, nb_games, nb_groups, group_size, risk, args);
        if (transient > 0) {
            auto avg = tmp.block<2, 100>(0, transient);

            results(0, run) = avg.row(0).mean();
            results(1, run) = avg.row(1).mean();
        } else {
            results(0, run) = tmp.row(0).mean();
            results(1, run) = tmp.row(1).mean();
        }
    }

    return results;

}

EGTTools::Matrix2D
EGTTools::RL::CRDSim::runTimingUncertainty(size_t nb_episodes, size_t nb_games, size_t min_rounds, size_t mean_rounds,
                                           size_t max_rounds, double p,
                                           double risk,
                                           const std::vector<double> &args,
                                           const std::string &crd_type) {

    // First of all we instantiate the CRD game with ucertianty
    if (mean_rounds > 0) {
        p = 1.0 / ((static_cast<double>(mean_rounds) - static_cast<double>(min_rounds)) + 1);
    }
    EGTTools::TimingUncertainty<std::mt19937_64> tu(p, max_rounds);
    CRDGame<PopContainer, EGTTools::TimingUncertainty<std::mt19937_64>> game;
    Matrix2D results = Matrix2D::Zero(2, nb_episodes);

    // Now create agent pool
    // Create a population of _group_size
    PopContainer group(_agent_type, _group_size, max_rounds, _nb_actions, max_rounds, _endowment, args);

    // Choose function to use
    void
    (EGTTools::RL::CRDSim::* reinforce)(double &, size_t &, double &, PopContainer &, size_t &,
                                        CRDGame<PopContainer, EGTTools::TimingUncertainty<std::mt19937_64>> &);

    if (_agent_type == "rothErev")
        reinforce = &EGTTools::RL::CRDSim::reinforceOnlyPositive<CRDGame<PopContainer, EGTTools::TimingUncertainty<std::mt19937_64>>>;
    else if (crd_type == "milinski")
        reinforce = &EGTTools::RL::CRDSim::reinforceAll<CRDGame<PopContainer, EGTTools::TimingUncertainty<std::mt19937_64>>>;
    else reinforce = &EGTTools::RL::CRDSim::reinforceXico<CRDGame<PopContainer, EGTTools::TimingUncertainty<std::mt19937_64>>>;

    for (size_t step = 0; step < nb_episodes; ++step) {
        size_t success = 0;
        double avg_contribution = 0.;
        double avg_rounds = 0.;
        for (unsigned int i = 0; i < nb_games; ++i) {
            // First we play the game
            auto[pool, final_round] = game.playGame(group, _available_actions, min_rounds, tu);
            avg_contribution += (game.playersContribution(group) / double(_group_size));
            (this->*reinforce)(pool, success, risk, group, final_round, game);
            avg_rounds += final_round;
        }
        results(0, step) = static_cast<double>(success) / static_cast<double>(nb_games);
        results(1, step) = static_cast<double>(avg_contribution) / static_cast<double>(nb_games);

        game.calcProbabilities(group);
        game.resetEpisode(group);
    }

    return results;
}

EGTTools::Matrix2D
EGTTools::RL::CRDSim::runConditionalTimingUncertainty(size_t nb_episodes, size_t nb_games, size_t min_rounds,
                                                      size_t mean_rounds,
                                                      size_t max_rounds, double p,
                                                      double risk,
                                                      const std::vector<double> &args,
                                                      const std::string &crd_type) {

    // First of all we instantiate the CRD game with ucertianty
    if (mean_rounds > 0) {
        p = 1.0 / ((static_cast<double>(mean_rounds) - static_cast<double>(min_rounds)) + 1);
    }
    EGTTools::TimingUncertainty<std::mt19937_64> tu(p, max_rounds);
    Matrix2D results = Matrix2D::Zero(2, nb_episodes);

    size_t success;
    double avg_contribution;
    double avg_rounds;
    ActionSpace available_actions(_nb_actions);
    std::iota(available_actions.begin(), available_actions.end(), 0);
    FlattenState flatten(Factors{max_rounds, (_group_size * _nb_actions) + 1});
    CRDConditional<PopContainer, EGTTools::TimingUncertainty<std::mt19937_64>> game(flatten);

    // Now create agent pool
    // Create a population of _group_size
    PopContainer group(_agent_type, _group_size, game.flatten().factor_space, _nb_actions, max_rounds,
                       _endowment, args);

    // Choose function to use
    void
    (EGTTools::RL::CRDSim::* reinforce)(double &, size_t &, double &, PopContainer &, size_t &,
                                        CRDConditional<PopContainer, EGTTools::TimingUncertainty<std::mt19937_64>> &);

    if (_agent_type == "rothErev")
        reinforce = &EGTTools::RL::CRDSim::reinforceOnlyPositive<CRDConditional<PopContainer, EGTTools::TimingUncertainty<std::mt19937_64>>>;
    else if (crd_type == "milinski")
        reinforce = &EGTTools::RL::CRDSim::reinforceAll<CRDConditional<PopContainer, EGTTools::TimingUncertainty<std::mt19937_64>>>;
    else reinforce = &EGTTools::RL::CRDSim::reinforceXico<CRDConditional<PopContainer, EGTTools::TimingUncertainty<std::mt19937_64>>>;

    for (size_t step = 0; step < nb_episodes; ++step) {
        success = 0;
        avg_contribution = 0.;
        avg_rounds = 0.;
        for (unsigned int i = 0; i < nb_games; ++i) {
            // First we play the game
            auto[pool, final_round] = game.playGame(group, available_actions, min_rounds, tu);
            avg_contribution += (game.playersContribution(group) / double(_group_size));
            (this->*reinforce)(pool, success, risk, group, final_round, game);
            avg_rounds += final_round;
        }
        results(0, step) = static_cast<double>(success) / static_cast<double>(nb_games);
        results(1, step) = static_cast<double>(avg_contribution) / static_cast<double>(nb_games);

        game.calcProbabilities(group);
        game.resetEpisode(group);
    }

    return results;
}


EGTTools::Matrix2D EGTTools::RL::CRDSim::runConditional(size_t nb_episodes, size_t nb_games,
                                                        const std::vector<double> &args, const std::string &crd_type) {
    Matrix2D results = Matrix2D::Zero(2, nb_episodes);
    size_t success;
    double avg_contribution;
    double avg_rounds;
    ActionSpace available_actions(_nb_actions);
    std::iota(available_actions.begin(), available_actions.end(), 0);
    FlattenState flatten(Factors{_nb_rounds, (_group_size * _nb_actions) + 1});
    CRDConditional<PopContainer> game(flatten);

    // Create a population of _group_size * nb_groups
    PopContainer popConditional(_agent_type, _group_size, game.flatten().factor_space, _nb_actions, _nb_rounds,
                                _endowment, args);
    void
    (EGTTools::RL::CRDSim::* reinforce)(double &, size_t &, double &, PopContainer &, CRDConditional<PopContainer> &);

    if (_agent_type == "rothErev")
        reinforce = &EGTTools::RL::CRDSim::reinforceOnlyPositive<CRDConditional<PopContainer>>;
    else if (crd_type == "milinski") reinforce = &EGTTools::RL::CRDSim::reinforceAll<CRDConditional<PopContainer>>;
    else reinforce = &EGTTools::RL::CRDSim::reinforceXico<CRDConditional<PopContainer>>;

    for (size_t step = 0; step < nb_episodes; ++step) {
        success = 0;
        avg_contribution = 0.;
        avg_rounds = 0.;
        for (size_t i = 0; i < nb_games; ++i) {
            // First we play the game
            auto[pool, final_round] = game.playGame(popConditional, available_actions, _nb_rounds);
            avg_contribution += (game.playersContribution(popConditional) / double(_group_size));
            (this->*reinforce)(pool, success, _risk, popConditional, game);
            avg_rounds += final_round;
        }
        results(0, step) = static_cast<double>(success) / static_cast<double>(nb_games);
        results(1, step) = static_cast<double>(avg_contribution) / static_cast<double>(nb_games);

        game.calcProbabilities(popConditional);
        game.resetEpisode(popConditional);
    }

    return results;

}


EGTTools::Matrix2D EGTTools::RL::CRDSim::runConditional(size_t nb_episodes, size_t nb_games, size_t nb_groups,
                                                        double risk, const std::vector<double> &args,
                                                        const std::string &crd_type) {
    EGTTools::Matrix2D results = Matrix2D::Zero(2, nb_groups);
    EGTTools::Vector group_achievement = Vector::Zero(nb_groups);
    EGTTools::Vector avg_donations = Vector::Zero(nb_groups);
    size_t transient = nb_episodes > 100 ? nb_episodes - 100 : 0;

    ActionSpace available_actions(_nb_actions);
    std::iota(available_actions.begin(), available_actions.end(), 0);

    void
    (EGTTools::RL::CRDSim::* reinforce)(double &, size_t &, double &, PopContainer &, CRDConditional<PopContainer> &);

    if (_agent_type == "rothErev")
        reinforce = &EGTTools::RL::CRDSim::reinforceOnlyPositive<CRDConditional<PopContainer>>;
    else if (crd_type == "milinski") reinforce = &EGTTools::RL::CRDSim::reinforceAll<CRDConditional<PopContainer>>;
    else reinforce = &EGTTools::RL::CRDSim::reinforceXico<CRDConditional<PopContainer>>;

    // Create a vector of groups
    std::vector<PopContainer> groups;

    for (size_t i = 0; i < nb_groups; ++i) {
        try {
            groups.emplace_back(_agent_type, _group_size,
                                EGTTools::RL::factorSpace(Factors{_nb_rounds, (_group_size * _nb_actions) + 1}),
                                _nb_actions, _nb_rounds, _endowment, args);
        } catch (std::invalid_argument &e) {
            throw e;
        }
    }

#pragma omp parallel for shared(transient, available_actions, groups) reduction(+:group_achievement, avg_donations)
    for (size_t group = 0; group < nb_groups; ++group) {
        size_t success;
        double avg_contribution;
        double avg_rounds;
        FlattenState flatten(Factors{_nb_rounds, (_group_size * _nb_actions) + 1});
        CRDConditional<PopContainer> game(flatten);

        for (size_t step = 0; step < nb_episodes; ++step) {
            success = 0;
            avg_contribution = 0.;
            avg_rounds = 0.;
            for (unsigned int i = 0; i < nb_games; ++i) {
                // First we play the game
                auto[pool, final_round] = game.playGame(groups[group], available_actions, _nb_rounds);
                avg_contribution += (game.playersContribution(groups[group]) / double(_group_size));
                (this->*reinforce)(pool, success, risk, groups[group], game);
                avg_rounds += final_round;
            }
            if (step >= transient) {
                group_achievement(group) += static_cast<double>(success) / static_cast<double>(nb_games);
                avg_donations(group) += avg_contribution / static_cast<double>(nb_games);
            }

            game.calcProbabilities(groups[group]);
            game.resetEpisode(groups[group]);
        }
    }
    results.row(0) = group_achievement;
    results.row(1) = avg_donations;

    results.array() /= static_cast<double>(nb_episodes - transient);

    return results;

}

void EGTTools::RL::CRDSim::resetPopulation() { population.reset(); }

template<class G>
void EGTTools::RL::CRDSim::reinforceOnlyPositive(double &pool, size_t &success, double &risk, PopContainer &pop,
                                                 G &game) {
    if (pool >= _threshold) {
        game.reinforcePath(pop);
        success++;
    } else if (_real_rand(_generator) > risk) game.reinforcePath(pop);
    else game.setPayoffs(pop, 0);
}

template<class G>
void EGTTools::RL::CRDSim::reinforceOnlyPositive(double &pool, size_t &success, double &risk, PopContainer &pop,
                                                 size_t &final_round, G &game) {
    if (pool >= _threshold) {
        game.reinforcePath(pop, final_round);
        success++;
    } else if (_real_rand(_generator) > risk) game.reinforcePath(pop, final_round);
    else game.setPayoffs(pop, 0);
}

template<class G>
void EGTTools::RL::CRDSim::reinforceAll(double &pool, size_t &success, double &risk, PopContainer &pop,
                                        G &game) {

    if (pool >= _threshold) success++;
    else if (_real_rand(_generator) < risk) game.setPayoffs(pop, 0);

    game.reinforcePath(pop);
}

template<class G>
void EGTTools::RL::CRDSim::reinforceAll(double &pool, size_t &success, double &risk, PopContainer &pop,
                                        size_t &final_round, G &game) {

    if (pool >= _threshold) success++;
    else if (_real_rand(_generator) < risk) game.setPayoffs(pop, 0);

    game.reinforcePath(pop, final_round);
}

template<class G>
void EGTTools::RL::CRDSim::reinforceXico(double &pool, size_t &success, double &risk, PopContainer &pop,
                                         G &game) {

    if (pool >= _threshold) success++;
    else if (_real_rand(_generator) < risk) {
        for (auto &player: pop) {
            player->set_payoff(player->payoff() - player->endowment());
        }
    }

    game.reinforcePath(pop);
}

template<class G>
void EGTTools::RL::CRDSim::reinforceXico(double &pool, size_t &success, double &risk, PopContainer &pop,
                                         size_t &final_round, G &game) {

    if (pool >= _threshold) success++;
    else if (_real_rand(_generator) < risk) {
        for (auto &player: pop) {
            player->set_payoff(player->payoff() - player->endowment());
        }
    }

    game.reinforcePath(pop, final_round);
}

void EGTTools::RL::CRDSim::setGameType(const std::string &crd_type) {
    if (crd_type == "milinski") _reinforce = &EGTTools::RL::CRDSim::reinforceAll;
    else _reinforce = &EGTTools::RL::CRDSim::reinforceXico;
}

size_t EGTTools::RL::CRDSim::nb_games() const { return _nb_games; }

size_t EGTTools::RL::CRDSim::nb_episodes() const { return _nb_episodes; }

size_t EGTTools::RL::CRDSim::nb_rounds() const { return _nb_rounds; }

size_t EGTTools::RL::CRDSim::nb_actions() const { return _nb_actions; }

double EGTTools::RL::CRDSim::endowment() const { return _endowment; }

double EGTTools::RL::CRDSim::risk() const { return _risk; }

double EGTTools::RL::CRDSim::threshold() const { return _threshold; }

const EGTTools::RL::ActionSpace &EGTTools::RL::CRDSim::available_actions() const { return _available_actions; }

const std::string &EGTTools::RL::CRDSim::agent_type() const { return _agent_type; }

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

void EGTTools::RL::CRDSim::set_agent_type(const std::string &agent_type) {
    _agent_type = agent_type;
}
