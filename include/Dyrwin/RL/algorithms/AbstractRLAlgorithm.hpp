//
// Created by Elias Fernandez on 2019-08-12.
//

#ifndef DYRWIN_ABSTRACTAGENT_HPP
#define DYRWIN_ABSTRACTAGENT_HPP

#include <string>
#include <Dyrwin/Types.h>

namespace EGTTools::RL::algorithms {
/**
 * @brief This class defines an interface for the RL algorithms in this library
 */
class AbstractRLAlgorithm {
 public:
  virtual ~AbstractRLAlgorithm() = 0;

  virtual void reinforceTrajectory() = 0;
  virtual void resetTrajectory() = 0;
  virtual bool inferPolicy() = 0;
  virtual size_t selectAction(size_t state) = 0;
  virtual void decreasePayoff(double value) = 0;
  virtual void increasePayoff(double value) = 0;
  virtual void reset() = 0;

  // getters
  virtual std::string toString() const = 0;
  virtual std::string type() const = 0;
  virtual size_t nb_states() const = 0;
  virtual size_t nb_actions() const = 0;
  virtual double payoff() const = 0;
  virtual const VectorXui &trajectoryStates() const = 0;
  virtual const VectorXui &trajectoryActions() const = 0;
  virtual const Matrix2D &policy() const = 0;
  // setters
  virtual void set_nb_states(size_t nb_states) = 0;
  virtual void set_nb_actions(size_t nb_actions) = 0;
  virtual void set_payoff(double payoff) = 0;
  virtual void reset_payoff() = 0;
  virtual void set_policy(const Eigen::Ref<const Matrix2D> &policy) = 0;
};
}

#endif //DYRWIN_ABSTRACTAGENT_HPP
