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
#include <Dyrwin/RL/BatchQLearningForgetAgent.h>
#include <Dyrwin/RL/HistericQLearningAgent.hpp>
#include <Dyrwin/RL/DiscountedQLearning.h>
#include <Dyrwin/RL/ESARSA.h>
#include <Dyrwin/RL/Utils.h>

namespace EGTTools::RL {
class PopContainer {
 private:
  RL::Population _agents;
 public:
  PopContainer() = default;

  PopContainer(const std::string &agent_type, size_t nb_agents, size_t nb_states, size_t nb_actions,
               size_t episode_length, double endowment, std::vector<double> args = {});

  Agent &operator[](size_t index);

  const Agent &operator[](size_t index) const;

  Individual &operator()(size_t index);

  const Individual &operator()(size_t index) const;

  typedef RL::Population::iterator iterator;
  typedef RL::Population::const_iterator const_iterator;

  iterator begin();

  iterator end();

  [[nodiscard]] const_iterator begin() const;

  [[nodiscard]] const_iterator end() const;

  [[nodiscard]] size_t size() const;

  void clear();

  void push_back(const RL::Individual &new_individual);

  void reset();

  [[nodiscard]] std::string toString() const;

  friend std::ostream &operator<<(std::ostream &o, PopContainer &r);
};
}

#endif //DYRWIN_RL_POPCONTAINER_HPP
