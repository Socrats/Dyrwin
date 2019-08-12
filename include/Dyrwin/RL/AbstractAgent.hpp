//
// Created by Elias Fernandez on 2019-08-12.
//

#ifndef DYRWIN_ABSTRACTAGENT_HPP
#define DYRWIN_ABSTRACTAGENT_HPP

namespace EGTTools::RL::Agents {
    /**
     * @brief This class defines an interface for the Agents in this library
     */
    class AbstractAgent {
    public:
        virtual ~AbstractAgent() = default;

        virtual std::string type() = 0;
        virtual void decreasePayoff() = 0;
        virtual void increasePayoff(double value) = 0;
        virtual bool inferPolicy() = 0;
        virtual void resetTrajectory() = 0;
        virtual void reinforceTrajectory() = 0;
        virtual size_t selectAction(size_t state) = 0;
        virtual void reset() = 0;
    };
}

#endif //DYRWIN_ABSTRACTAGENT_HPP
