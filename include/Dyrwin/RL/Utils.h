/**
 * Created by Elias Fernandez on 2019-03-11.
 *
 * Some function are taken from https://github.com/Svalorzen/AI-Toolbox/blob/master/src/Factored/Utils/Core.cpp
 */

#ifndef DYRWIN_RL_UTILS_H
#define DYRWIN_RL_UTILS_H

#include <vector>
#include <memory>
#include <Dyrwin/RL/Agent.h>
#include <Dyrwin/RL/RothErevAgent.h>
#include <Dyrwin/RL/QLearningAgent.h>
#include <Dyrwin/RL/BatchQLearningAgent.h>
#include <Dyrwin/RL/HistericQLearningAgent.hpp>

namespace EGTTools::RL {
    using Factors = std::vector<size_t>;
    using ActionSpace = std::vector<double>;
    using ActionSpaceU = std::vector<size_t>;
    using Individual = std::shared_ptr<EGTTools::RL::Agent>;
    using Population = std::vector<Individual>;
    using RothErevPopulation = std::vector<EGTTools::RL::Agent>;
    using RothErevLambdaPopulation = std::vector<EGTTools::RL::RothErevAgent>;
    using QLearningPopulation = std::vector<EGTTools::RL::QLearningAgent>;
    using BatchQLearningPopulation = std::vector<EGTTools::RL::BatchQLearningAgent>;
    using HistericQLearningPopulation = std::vector<EGTTools::RL::HistericQLearningAgent>;

    /**
     * @brief Calculates the size of a flattened multi-dimensional space.
     * @param space Vector of containing the sizes of each dimension of the space.
     * @return size of the multi-dimensional space
     */
    size_t factorSpace(const Factors &space);

    /**
     * @brief Tranforms an index to the flattened space into a multi-dimensional vector.
     *
     * Calculates the multi-demensional vector that refers to the point indicated by
     * @param id in the flattened space.
     *
     * @param space dimentions of the space
     * @param id index to the flattened space
     * @param out pointer for a vector that will contain the multi-dimensional pointer
     */
    void toFactors(const Factors &space, size_t id, Factors *out);

    /**
     *
     * @param space
     * @param id
     * @return
     */
    Factors toFactors(const Factors &space, size_t id);

    /**
     *
     * @param space
     * @param f
     * @return
     */
    size_t toIndex(const Factors &space, const Factors &f);

    struct FlattenState {
        /**
         *
         * @param space
         */
        explicit FlattenState(const Factors& space);

        Factors space;
        size_t factor_space;

        Factors toFactors(size_t id);

        void toFactors(size_t id, Factors *out);

        size_t toIndex(const Factors &f);
    };
}

#endif //DYRWIN_RL_UTILS_H
