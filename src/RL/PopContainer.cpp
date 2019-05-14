//
// Created by Elias Fernandez on 2019-05-14.
//

#include <Dyrwin/RL/PopContainer.hpp>

EGTTools::RL::PopContainer::PopContainer(const std::string &agent_type, size_t nb_agents, size_t nb_states, size_t nb_actions,
             size_t episode_length, double endowment, std::vector<double> args) {
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

EGTTools::RL::Agent &EGTTools::RL::PopContainer::operator[](size_t index) {
    return *_agents[index];
}

const EGTTools::RL::Agent &EGTTools::RL::PopContainer::operator[](size_t index) const {
    return *_agents[index];
}

EGTTools::RL::Individual &EGTTools::RL::PopContainer::operator()(size_t index) {
    return _agents[index];
}

const EGTTools::RL::Individual &EGTTools::RL::PopContainer::operator()(size_t index) const {
    return _agents[index];
}

EGTTools::RL::PopContainer::iterator EGTTools::RL::PopContainer::begin() { return _agents.begin(); }

EGTTools::RL::PopContainer::iterator EGTTools::RL::PopContainer::end() { return _agents.end(); }

EGTTools::RL::PopContainer::const_iterator EGTTools::RL::PopContainer::begin() const { return _agents.begin(); }

EGTTools::RL::PopContainer::const_iterator EGTTools::RL::PopContainer::end() const { return _agents.end(); }

size_t EGTTools::RL::PopContainer::size() const { return _agents.size(); }

void EGTTools::RL::PopContainer::clear() { _agents.clear(); }

void EGTTools::RL::PopContainer::reset() {
    for (auto& agent: _agents) agent->reset();
}

std::string EGTTools::RL::PopContainer::toString() const {
    std::stringstream ss;
    ss << "[";
    for(size_t i = 0; i < size(); ++i) {
        ss << _agents[i]->type() << ",";
    }
    ss << "]";
    return ss.str();
}

std::ostream& operator<<(std::ostream &o, EGTTools::RL::PopContainer &r) {
    return o << r.toString();
}