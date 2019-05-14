//
// Created by Elias Fernandez on 2019-05-10.
//

#ifndef DYRWIN_RL_POPCONTAINER_HPP
#define DYRWIN_RL_POPCONTAINER_HPP

#include <typeinfo>
#include <Dyrwin/RL/Agent.h>
#include <Dyrwin/RL/RothErevAgent.h>
#include <Dyrwin/RL/QLearningAgent.h>
#include <Dyrwin/RL/BatchQLearningAgent.h>
#include <Dyrwin/RL/HistericQLearningAgent.hpp>
#include <Dyrwin/RL/Utils.h>

namespace EGTTools::RL {
    class PopContainer {
    private:
        RL::Population _agents;
    public:
        PopContainer() = default;

        PopContainer(const std::string &agent_type, size_t nb_agents, size_t nb_states, size_t nb_actions,
                     size_t episode_length, double endowment, std::vector<double> args = {}) {
            if (agent_type == "rothErev") {
                for (unsigned i = 0; i < nb_agents; i++) {
                    _agents.push_back(std::make_shared<Agent>(nb_states, nb_actions, episode_length, endowment));
                }
            } else if (agent_type == "rothErevLambda") {
                if (args.size() < 2)
                    throw std::invalid_argument("You must specify lambda and temperature as arguments");
                for (unsigned i = 0; i < nb_agents; i++) {
                    _agents.push_back(
                            std::make_shared<RothErevAgent>(nb_states, nb_actions, episode_length, endowment, args[0],
                                                            args[1]));
                }
            } else if (agent_type == "QLearning") {
                if (args.size() < 3)
                    throw std::invalid_argument("You must specify alpha, lambda and temperature as arguments");
                for (unsigned i = 0; i < nb_agents; i++) {
                    _agents.push_back(
                            std::make_shared<QLearningAgent>(nb_states, nb_actions, episode_length, endowment, args[0],
                                                             args[1],
                                                             args[2]));
                }
            } else if (agent_type == "HistericQLearning") {
                if (args.size() < 3)
                    throw std::invalid_argument("You must specify alpha, beta and temperature as arguments");
                for (unsigned i = 0; i < nb_agents; i++) {
                    _agents.push_back(
                            std::make_shared<HistericQLearningAgent>(nb_states, nb_actions, episode_length, endowment,
                                                                     args[0], args[1],
                                                                     args[2]));
                }
            } else if (agent_type == "BatchQLearning") {
                if (args.size() < 2) throw std::invalid_argument("You must specify alpha and temperature as arguments");
                for (unsigned i = 0; i < nb_agents; i++) {
                    _agents.push_back(
                            std::make_shared<BatchQLearningAgent>(nb_states, nb_actions, episode_length, endowment,
                                                                  args[0], args[1]));
                }
            } else {
                throw std::invalid_argument("Invalid agent type");
            }
        }

        Agent &operator[](size_t index) {
            return *_agents[index];
        };

        const Agent &operator[](size_t index) const {
            return *_agents[index];
        };

        Individual &operator()(size_t index) {
            return _agents[index];
        }

        const Individual &operator()(size_t index) const {
            return _agents[index];
        }

        typedef RL::Population::iterator iterator;
        typedef RL::Population::const_iterator const_iterator;

        iterator begin() { return _agents.begin(); }

        iterator end() { return _agents.end(); }

        const_iterator begin() const { return _agents.begin(); }

        const_iterator end() const { return _agents.end(); }

        size_t size() const { return _agents.size(); }

        void clear() { _agents.clear(); }

        void reset() {
            for (auto& agent: _agents) agent->reset();
        }

        std::string toString() const {
            std::stringstream ss;
            ss << "[";
            for(size_t i = 0; i < size(); ++i) {
                ss << _agents[i]->type() << ",";
            }
            ss << "]";
            return ss.str();
        }

        friend std::ostream &operator<<(std::ostream &o, PopContainer &r) { return o << r.toString(); }
    };
}

#endif //DYRWIN_RL_POPCONTAINER_HPP
