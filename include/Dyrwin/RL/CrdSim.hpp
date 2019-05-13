#include <utility>

//
// Created by Elias Fernandez on 2019-05-10.
//

#ifndef DYRWIN_CRDSIM_HPP
#define DYRWIN_CRDSIM_HPP

#include <Dyrwin/RL/BatchQLearningAgent.h>
#include <Dyrwin/RL/QLearningAgent.h>
#include <Dyrwin/RL/RothErevAgent.h>
#include <Dyrwin/RL/HistericQLearningAgent.hpp>
#include <Dyrwin/RL/CRDGame.h>
#include <Dyrwin/RL/PopContainer.hpp>
#include <Dyrwin/RL/Utils.h>

namespace EGTTools::RL {
    class CRDSim {
    public:
        CRDSim(size_t nb_runs, size_t nb_games, size_t nb_rounds, size_t nb_actions, size_t group_size,
               const std::string &agent_type, double risk, const std::vector<double> &args) : _nb_runs(nb_runs),
                                                                                              _nb_games(nb_games),
                                                                                              _nb_rounds(nb_rounds),
                                                                                              _nb_actions(nb_actions),
                                                                                              _group_size(group_size),
                                                                                              _risk(risk) {
            _endowment = 2 * _nb_rounds;
            _threshold = static_cast<double>(_nb_rounds * _group_size);
            _available_actions = ActionSpace(nb_actions);
            for (size_t i = 0; i < _nb_actions; ++i) _available_actions[i] = i;
            try {
                population = PopContainer(agent_type, group_size, nb_rounds, nb_actions, nb_rounds, _endowment, args);
            } catch (std::invalid_argument &e) {
                throw e;
            }
            if (agent_type == "rothErev") _reinforce = &EGTTools::RL::CRDSim::reinforceRothErev;
            else _reinforce = &EGTTools::RL::CRDSim::reinforceBatchQLearning;
        }

        Matrix2D run(size_t nb_runs, size_t nb_games) {
            Matrix2D results = Matrix2D::Zero(2, nb_runs);

            for (size_t step = 0; step < nb_runs; ++step) {
                size_t success = 0;
                double avgpayoff = 0.;
                double avg_rounds = 0.;
                for (unsigned int game = 0; game < nb_games; ++game) {
                    // First we play the game
                    auto[pool, final_round] = Game.playGame(population, _available_actions, _nb_rounds);
                    avgpayoff += (Game.playersPayoff(population) / double(_group_size));
                    (this->*_reinforce)(pool, success, EGTTools::probabilityDistribution(_generator), _risk, _threshold);
                    avg_rounds += final_round;
                }
                results(0, step) = success / static_cast<double>(nb_games);
                results(1, step) = avgpayoff / static_cast<double>(nb_games);

                Game.calcProbabilities(population);
                Game.resetEpisode(population);
            }

            return results;
        }

        void resetPopulation() { population.reset(); }

        void reinforceRothErev(double &pool, size_t &success, double rnd_value, double &cataclysm, double &threshold) {
            if (pool >= threshold) {
                Game.reinforcePath(population);
                success++;
            } else if (rnd_value > cataclysm) Game.reinforcePath(population);
            else Game.setPayoffs(population, 0);
        }

        void reinforceBatchQLearning(double &pool, size_t &success, double rnd_value, double &cataclysm, double &threshold) {

            if (pool >= threshold) success++;
            else if (rnd_value < cataclysm) Game.setPayoffs(population, 0);

            Game.reinforcePath(population);
        }

        size_t nb_games() const { return _nb_games; }

        size_t nb_runs() const { return _nb_runs; }

        size_t nb_rounds() const { return _nb_rounds; }

        size_t nb_actions() const { return _nb_actions; }

        double endowment() const { return _endowment; }

        double risk() const { return _risk; }

        double threshold() const { return _threshold; }

        const ActionSpace &available_actions() const { return _available_actions; }

        void set_nb_games(size_t nb_games) { _nb_games = nb_games; }

        void set_nb_runs(size_t nb_runs) { _nb_runs = nb_runs; }

        void set_nb_rounds(size_t nb_rounds) { _nb_rounds = nb_rounds; }

        void set_nb_actions(size_t nb_actions) { _nb_actions = nb_actions; }

        void set_risk(double risk) { _risk = risk; }

        void set_threshold(double threshold) { _threshold = threshold; }

        void set_available_actions(const ActionSpace &available_actions) {
            if (available_actions.size() != _nb_actions)
                throw std::invalid_argument("you can't specify more actions than " + std::to_string(_nb_actions));
            _available_actions.resize(available_actions.size());
            for (size_t i = 0; i < available_actions.size(); ++i)
                _available_actions[i] = available_actions[i];
        }

        CRDGame<PopContainer> Game;
        PopContainer population;

    private:
        size_t _nb_runs, _nb_games, _nb_rounds, _nb_actions, _group_size;
        double _risk, _threshold, _endowment;
        ActionSpace _available_actions;

        void (EGTTools::RL::CRDSim::* _reinforce)(double &, size_t &, double, double &, double &) = nullptr;

        // Random generators
        std::mt19937_64 _generator{EGTTools::Random::SeedGenerator::getInstance().getSeed()};
    };

}


#endif //DYRWIN_CRDSIM_HPP
