//
// Created by Elias Fernandez on 2019-04-25.
//

#ifndef DYRWIN_SED_MLS_HPP
#define DYRWIN_SED_MLS_HPP

#include <random>
#include <algorithm>
#include <cmath>
#include <Dyrwin/SeedGenerator.h>
#include <Dyrwin/Types.h>
#include <Dyrwin/SED/structure/Group.hpp>
#include <Dyrwin/SED/Utils.hpp>
#include <Dyrwin/OpenMPUtils.hpp>

namespace EGTTools::SED {
    template<typename S = Group>
    class MLS {
    public:
        /**
         * @brief This class implements the Multi Level selection process introduced in Arne et al.
         *
         * This class implements selection on the level of groups.
         *
         * A population with m groups, which all have a maximum size n. Therefore, the maximum population
         * size N = nm. Each group must contain at least one individual. The minimum population size is m
         * (each group must have at least one individual). In each time step, an individual is chosen from
         * a population with a probability proportional to its fitness. The individual produces an
         * identical offspring that is added to the same group. If the group size is greater than n after
         * this step, then either a randomly chosen individual from the group is eliminated (with probability 1-q)
         * or the group splits into two groups (with probability q). Each individual of the splitting
         * group has probability 1/2 to end up in each of the daughter groups. One daughter group remains
         * empty with probability 2^(1-n). In this case, the repeating process is repeated to avoid empty
         * groups. In order to keep the number of groups constant, a randomly chosen group is eliminated
         * whenever a group splits in two.
         *
         * @param generations : maximum number of generations
         * @param nb_strategies : number of strategies in the population
         * @param group_size : group size (n)
         * @param nb_groups : number of groups (m)
         * @param w : intensity of selection
         * @param strategy_freq : frequency of each strategy in the population
         * @param payoff_matrix : payoff matrix
         */
        MLS(size_t generations, size_t nb_strategies, size_t group_size, size_t nb_groups, double w,
            const Eigen::Ref<const Vector> &strategies_freq, const Eigen::Ref<const Matrix2D> &payoff_matrix);

        Vector evolve(double q, double w, const Eigen::Ref<const VectorXui> &init_state);
//
//        Vector evolve(size_t runs, double w);

        double fixationProbability(size_t invader, size_t resident, size_t runs,
                                   double q, double w);

        double fixationProbability(size_t invader, size_t resident, size_t runs,
                                   double q, double lambda, double w);

        Vector fixationProbability(size_t invader, const Eigen::Ref<const VectorXui> &init_state, size_t runs,
                                   double q, double w);

//        double fixationProbability(size_t invader, size_t resident, size_t runs,
//                                   size_t t0, double q, double lambda, double w, double mu);

        Vector
        gradientOfSelection(size_t invader, size_t resident, size_t runs, double w, double q = 0.0);

        Vector
        gradientOfSelection(size_t invader, size_t reduce, const Eigen::Ref<const VectorXui> &init_state, size_t runs,
                            double w, double q = 0.0);

        // To avoid memory explosion, we limit the call to this function for a maximum of 3 strategies
//        SparseMatrix2D transitionMatrix(size_t runs, size_t t0, double q, double lambda, double w);
//
//        SparseMatrix2D
//        transitionMatrix(size_t invader, size_t resident, size_t runs, size_t t0, double q, double lambda, double w);


        // Getters
        size_t generations() { return _generations; }

        size_t nb_strategies() { return _nb_strategies; }

        size_t max_pop_size() { return _pop_size; }

        size_t group_size() { return _group_size; }

        size_t nb_groups() { return _nb_groups; }

        double selection_intensity() { return _w; }

        Vector init_strategy_freq() { return _strategies.cast<double>() / _pop_size; }

        Vector &strategy_freq() { return _strategy_freq; }

        VectorXui &init_strategy_count() { return _strategies; }

        Matrix2D &payoff_matrix() { return _payoff_matrix; }

        // Setters
        void set_generations(size_t generations) { _generations = generations; }

        void set_pop_size(size_t pop_size) { _pop_size = pop_size; }

        void set_group_size(size_t group_size) {
            _group_size = group_size;
            _pop_size = _nb_groups * _group_size;
        }

        void set_nb_groups(size_t nb_groups) {
            _nb_groups = nb_groups;
            _uint_rand.param(std::uniform_int_distribution<size_t>::param_type(0, _nb_groups - 1));
            _pop_size = _nb_groups * _group_size;
        }

        void set_selection_intensity(double w) { _w = w; }

        void set_strategy_freq(const Eigen::Ref<const Vector> &strategy_freq) {
            if (strategy_freq.sum() != 1.0) throw std::invalid_argument("Frequencies must sum to 1");
            _strategy_freq.array() = strategy_freq;
            // Recompute strategies
            size_t tmp = 0;
            for (size_t i = 0; i < (_nb_strategies - 1); ++i) {
                _strategies(i) = (size_t) floor(_strategy_freq(i) * _pop_size);
                tmp += _strategies(i);
            }
            _strategies(_nb_strategies - 1) = _pop_size - tmp;
        }

        void set_strategy_count(const Eigen::Ref<const VectorXui> &strategies) {
            if (strategies.sum() != _pop_size)
                throw std::invalid_argument("The sum of all individuals must be equal to the population size!");
            _strategies.array() = strategies;
            // Recompute strategy frequencies
            _strategy_freq.array() = _strategies.cast<double>() / _pop_size;
        }

        void set_payoff_matrix(const Eigen::Ref<const Matrix2D> &payoff_matrix) {
            if (payoff_matrix.rows() != payoff_matrix.cols())
                throw std::invalid_argument("Payoff matrix must be a square Matrix (n,n)");
            _nb_strategies = payoff_matrix.rows();
            _uint_rand_strategy.param(std::uniform_int_distribution<size_t>::param_type(0, _nb_strategies - 1));
            _payoff_matrix.array() = payoff_matrix;
        }

        std::string toString() const {
            Eigen::IOFormat CleanFmt(4, 0, ", ", "\n", "[", "]");
            std::stringstream ss;
            ss << _payoff_matrix.format(CleanFmt);
            return "Z = " + std::to_string(_pop_size) +
                   "\nm = " + std::to_string(_nb_groups) +
                   "\nn = " + std::to_string(_group_size) +
                   "\nnb_strategies = " + std::to_string(_nb_strategies) +
                   "\npayoff_matrix = " + ss.str();
        }

        friend std::ostream &operator<<(std::ostream &o, MLS &r) { return o << r.toString(); }

    private:
        size_t _generations, _nb_strategies, _group_size, _nb_groups, _pop_size;
        double _w;

        Vector _strategy_freq; // frequency of each strategy in the population
        VectorXui _strategies; //nb of players of each strategy
        Matrix2D _payoff_matrix; // stores the payoff matrix of the game

        // Uniform random distribution
        std::uniform_int_distribution<size_t> _uint_rand;
        std::uniform_int_distribution<size_t> _uint_rand_strategy;
        std::uniform_real_distribution<double> _real_rand; // uniform random distribution

        // Random generators
        std::mt19937_64 _mt{EGTTools::Random::SeedGenerator::getInstance().getSeed()};

        inline void _update(double q, std::vector<S> &groups, VectorXui &strategies);

        inline void _update(double q, double lambda, std::vector<S> &groups, VectorXui &strategies);

        inline void _update(double q, double lambda, double mu, std::vector<S> &groups, VectorXui &strategies);

        inline void _speedUpdate(double q, std::vector<S> &groups, VectorXui &strategies);

        inline void _speedUpdate(double q, double lambda, std::vector<S> &groups, VectorXui &strategies);

        inline void _speedUpdate(double q, double lambda, double mu, std::vector<S> &groups, VectorXui &strategies);

        inline void _createMutant(size_t invader, size_t resident, std::vector<S> &groups);

        inline void _createRandomMutant(size_t invader, std::vector<S> &groups, VectorXui &strategies);

        inline void _updateFullPopulationFrequencies(size_t increase, size_t decrease, VectorXui &strategies);

        void _reproduce(std::vector<S> &groups, VectorXui &strategies);

        void _reproduce(std::vector<S> &groups, VectorXui &strategies, double q);

        void _migrate(double q, std::vector<S> &groups, VectorXui &strategies);

        void _mutate(std::vector<S> &groups, VectorXui &strategies);

        void _splitGroup(size_t parent_group, std::vector<S> &groups, VectorXui &strategies);

        size_t _payoffProportionalSelection(std::vector<S> &groups);

        size_t _sizeProportionalSelection(std::vector<S> &groups);

        bool _pseudoStationary(std::vector<S> &groups);

        void _setFullHomogeneousState(size_t strategy, std::vector<S> &groups);

        inline void _setState(std::vector<S> &groups, std::vector<size_t> &container);

        inline size_t _current_pop_size(std::vector<S> &groups);

    };


    template<typename S>
    MLS<S>::MLS(size_t generations, size_t nb_strategies,
                size_t group_size, size_t nb_groups, double w,
                const Eigen::Ref<const EGTTools::Vector> &strategies_freq,
                const Eigen::Ref<const EGTTools::Matrix2D> &payoff_matrix) : _generations(generations),
                                                                             _nb_strategies(nb_strategies),
                                                                             _group_size(group_size),
                                                                             _nb_groups(nb_groups),
                                                                             _w(w),
                                                                             _strategy_freq(strategies_freq),
                                                                             _payoff_matrix(payoff_matrix) {
        if (static_cast<size_t>(_payoff_matrix.rows() * _payoff_matrix.cols()) != (_nb_strategies * _nb_strategies))
            throw std::invalid_argument(
                    "Payoff matrix has wrong dimensions it must have shape (nb_strategies, nb_strategies)");
        _pop_size = _nb_groups * _group_size;
        // calculate the frequencies of each strategy in the population
        _strategies = VectorXui::Zero(_nb_strategies);
        // Calculate the number of individuals belonging to each strategy from the initial frequencies
        size_t tmp = 0;
        for (size_t i = 0; i < (_nb_strategies - 1); ++i) {
            _strategies(i) = (size_t) floor(_strategy_freq(i) * _pop_size);
            tmp += _strategies(i);
        }
        _strategies(_nb_strategies - 1) = (size_t) _pop_size - tmp;

        // Initialize random uniform distribution
        _uint_rand = std::uniform_int_distribution<size_t>(0, _nb_groups - 1);
        _uint_rand_strategy = std::uniform_int_distribution<size_t>(0, _nb_strategies - 1);
        _real_rand = std::uniform_real_distribution<double>(0.0, 1.0);
    }

    /**
     * Runs the moran process with multi-level selection for a given number of generations
     * or until it reaches a monomorphic state.
     *
     * @tparam S : container for a group
     * @param q : splitting probability
     * @param w : intensity of selection
     * @param init_state : vector with the initial state of the population
     * @return a vector with the final state of the population
     */
    template<typename S>
    Vector MLS<S>::evolve(double q, double w, const Eigen::Ref<const VectorXui> &init_state) {
        if ((_nb_groups == 1) && q != 0.)
            throw std::invalid_argument(
                    "The splitting probability must be zero when there is only 1 group in the population");
        if (static_cast<size_t>(init_state.size()) != _nb_strategies)
            throw std::invalid_argument(
                    "you must specify the number of individuals of each " + std::to_string(_nb_strategies) +
                    " strategies");
        if (init_state.sum() != _pop_size)
            throw std::invalid_argument(
                    "the sum of individuals in the initial state must be equal to " + std::to_string(_pop_size));

        _strategies.array() = init_state;
        // Initialize population with initial state
        VectorXui group_strategies = VectorXui::Zero(_nb_strategies);
        Group group(_nb_strategies, _group_size, w, group_strategies, _payoff_matrix);
        group.set_group_size(_group_size);
        std::vector<size_t> pop_container(_pop_size);
        // initialize container
        size_t z = 0, sum = 1;
        for (size_t i = 0; i < _nb_strategies; ++i) {
            for (size_t j = 0; j < init_state(i); ++j) {
                pop_container[z++] = i;
            }
        }
        std::vector<Group> groups(_nb_groups, group);
        _setState(groups, pop_container);


        // Then we run the Moran Process
        for (size_t t = 0; t < _generations; ++t) {
            _speedUpdate(q, groups, _strategies);
            sum = _strategies.sum();
            if ((_strategies.array() == sum).any()) break;
        } // end Moran process loop
        return _strategies.cast<double>() / static_cast<double>(sum);

    }

/**
 * @brief estimates the fixation probability of the invading strategy over the resident strategy.
 *
 * This function will estimate numerically (by running simulations) the fixation probability of
 * a certain strategy in the population of 1 resident strategy.
 *
 * This implementation specializes on the EGTTools::SED::Group class
 *
 * @tparam S : container for the structure of the population
 * @param invader : index of the invading strategy
 * @param resident : index of the resident strategy
 * @param runs : number of runs (used to average the number of times the invading strategy has fixated)
 * @param q : splitting probability
 * @param w : intensity of selection
 * @return a real number (double) indicating the fixation probability
 */
    template<typename S>
    double
    MLS<S>::fixationProbability(size_t invader, size_t resident, size_t runs, double q, double w) {
        if (invader > _nb_strategies || resident > _nb_strategies)
            throw std::invalid_argument(
                    "you must specify a valid index for invader and resident [0, " + std::to_string(_nb_strategies) +
                    ")");
        if ((_nb_groups == 1) && q != 0.)
            throw std::invalid_argument(
                    "The splitting probability must be zero when there is only 1 group in the population");

        double r2m = 0; // resident to mutant count
        double r2r = 0; // resident to resident count
        VectorXui group_strategies = VectorXui::Zero(_nb_strategies);
        group_strategies(resident) = _group_size;

        // This loop can be done in parallel
#pragma omp parallel for shared(group_strategies) reduction(+:r2m, r2r)
        for (size_t i = 0; i < runs; ++i) {
            // First we initialize a homogeneous population with the resident strategy
            SED::Group group(_nb_strategies, _group_size, w, group_strategies, _payoff_matrix);
            std::vector<SED::Group> groups(_nb_groups, group);
            VectorXui strategies = VectorXui::Zero(_nb_strategies);
            strategies(resident) = _pop_size;

            // Then we create a mutant of the invading strategy
            _createMutant(invader, resident, groups);
            // Update full population frequencies
            _updateFullPopulationFrequencies(invader, resident, strategies);

            // Then we run the Moran Process
            for (size_t t = 0; t < _generations; ++t) {
                _speedUpdate(q, groups, strategies);

                if (strategies(invader) == 0) {
                    r2r += 1.0;
                    break;
                } else if (strategies(resident) == 0) {
                    r2m += 1.0;
                    break;
                }
            } // end Moran process loop
        } // end runs loop
        if ((r2m == 0.0) && (r2r == 0.0)) return 0.0;
        else return r2m / (r2m + r2r);
    }

/**
 * @brief estimates the fixation probability of the invading strategy over the resident strategy.
 *
 * This function will estimate numerically (by running simulations) the fixation probability of
 * a certain strategy in the population of 1 resident strategy.
 *
 * This implementation specializes on the EGTTools::SED::Group class
 *
 * @tparam S : container for the structure of the population
 * @param invader : index of the invading strategy
 * @param resident : index of the resident strategy
 * @param runs : number of runs (used to average the number of times the invading strategy has fixated)
 * @param q : splitting probability
 * @param lambda : migration probability
 * @param w : intensity of selection
 * @return a real number (double) indicating the fixation probability
 */
    template<typename S>
    double
    MLS<S>::fixationProbability(size_t invader, size_t resident, size_t runs,
                                double q, double lambda, double w) {
        if (invader > _nb_strategies || resident > _nb_strategies)
            throw std::invalid_argument(
                    "you must specify a valid index for invader and resident [0, " + std::to_string(_nb_strategies) +
                    ")");
        if ((_nb_groups == 1) && q != 0.)
            throw std::invalid_argument(
                    "The splitting probability must be zero when there is only 1 group in the population");

        double r2m = 0; // resident to mutant count
        double r2r = 0; // resident to resident count
        VectorXui group_strategies = VectorXui::Zero(_nb_strategies);
        group_strategies(resident) = _group_size;

        // This loop can be done in parallel
#pragma omp parallel for shared(group_strategies) reduction(+:r2m, r2r)
        for (size_t i = 0; i < runs; ++i) {
            // First we initialize a homogeneous population with the resident strategy
            SED::Group group(_nb_strategies, _group_size, w, group_strategies, _payoff_matrix);
            std::vector<SED::Group> groups(_nb_groups, group);
            VectorXui strategies = VectorXui::Zero(_nb_strategies);
            strategies(resident) = _pop_size;

            // Then we create a mutant of the invading strategy
            _createMutant(invader, resident, groups);
            // Update full population frequencies
            _updateFullPopulationFrequencies(invader, resident, strategies);

            // Then we run the Moran Process
            for (size_t t = 0; t < _generations; ++t) {
                _speedUpdate(q, lambda, groups, strategies);

                if (strategies(invader) == 0) {
                    r2r += 1.0;
                    break;
                } else if (strategies(resident) == 0) {
                    r2m += 1.0;
                    break;
                }
            } // end Moran process loop
        } // end runs loop

        if ((r2m == 0.0) && (r2r == 0.0)) return 0.0;
        else return r2m / (r2m + r2r);
    }

    /**
     * @brief estimates the fixation probability of the invading strategy over the resident strategy.
    *
     * This function will estimate numerically (by running simulations) the fixation probability of
     * a certain strategy in the population of 1 resident strategy.
     *
     * This implementation specializes on the EGTTools::SED::Group class
     * @tparam S : container for the structure of the population (group)
     * @param invader : index of the invading strategy
     * @param init_state : vector containing the initial state of the population (number of individuals of each strategy)
     * @param runs : number of runs (used to average the number of times the invading strategy has fixated)
     * @param q : splitting probability
     * @param w : intensity of selection
     * @return a vector of doubles indicating the probability that each strategy fixates from the initial state
     */
    template<typename S>
    Vector MLS<S>::fixationProbability(size_t invader, const Eigen::Ref<const VectorXui> &init_state, size_t runs,
                                       double q, double w) {
        if (invader > _nb_strategies)
            throw std::invalid_argument(
                    "you must specify a valid index for invader and resident [0, " + std::to_string(_nb_strategies) +
                    ")");
        if ((_nb_groups == 1) && q != 0.)
            throw std::invalid_argument(
                    "The splitting probability must be zero when there is only 1 group in the population");
        if (static_cast<size_t>(init_state.size()) != _nb_strategies)
            throw std::invalid_argument(
                    "you must specify the number of individuals of each " + std::to_string(_nb_strategies) +
                    " strategies");
        if (init_state.sum() != _pop_size)
            throw std::invalid_argument(
                    "the sum of individuals in the initial state must be equal to " + std::to_string(_pop_size));

        Vector fixations = Vector::Zero(_nb_strategies);

        // Initialize population with initial state
        VectorXui group_strategies = VectorXui::Zero(_nb_strategies);
        Group group(_nb_strategies, _group_size, w, group_strategies, _payoff_matrix);
        group.set_group_size(_group_size);
        std::vector<size_t> pop_container(_pop_size);
        // initialize container
        size_t z = 0;
        for (size_t i = 0; i < _nb_strategies; ++i) {
            for (size_t j = 0; j < init_state(i); ++j) {
                pop_container[z++] = i;
            }
        }

        // This loop can be done in parallel
#pragma omp parallel for shared(group, pop_container) reduction(+:fixations)
        for (size_t i = 0; i < runs; ++i) {
            // First we initialize a homogeneous population with the resident strategy
            bool fixated = false;
            std::vector<Group> groups(_nb_groups, group);
            VectorXui strategies = init_state;
            std::vector<size_t> container(pop_container);
            _setState(groups, container);

            // Then we create a mutant of the invading strategy
            _createRandomMutant(invader, groups, strategies);

            // Then we run the Moran Process
            for (size_t t = 0; t < _generations; ++t) {
                _speedUpdate(q, groups, strategies);
                size_t sum = strategies.sum();
                for (size_t s = 0; s < _nb_strategies; ++s)
                    if (strategies(s) == sum) {
                        fixations(s) += 1.0;
                        fixated = true;
                        break;
                    }
                if (fixated) break;
            } // end Moran process loop
        } // end runs loop

        double tmp = fixations.sum();

        if (tmp > 0.0)
            return fixations.array() / tmp;

        return fixations.array();
    }

/**
 * @brief calculates the gradient of selection between 2 strategies.
 *
 * Will return the difference between T+ and T- for each possible population configuration
 * when the is conformed only by the resident and the invading strategy.
 *
 * To estimate T+ - T- (the probability that the number of invaders increase/decrease in the population)
 * we run the simulation for population with k invaders and Z - k residents for @param run
 * times and average how many times did the number of invadors increase and decrease.
 *
 * @tparam S : group container
 * @param invader : index of the invading strategy
 * @param resident : index of the resident strategy
 * @param runs : number of runs (to average the results)
 * @param w : intensity of selection
 * @param q : splitting probability
 * @return : an Eigen vector with the gradient of selection for each k/Z where k is the number of invaders.
 */
    template<typename S>
    Vector
    MLS<S>::gradientOfSelection(size_t invader, size_t resident, size_t runs, double w, double q) {
        if (invader > _nb_strategies || resident > _nb_strategies)
            throw std::invalid_argument(
                    "you must specify a valid index for invader and resident [0, " + std::to_string(_nb_strategies) +
                    ")");
        if ((_nb_groups == 1) && q != 0.)
            throw std::invalid_argument(
                    "The splitting probability must be zero when there is only 1 group in the population");


        Vector gradient = Vector::Zero(_pop_size + 1);

        // This loop can be done in parallel
#pragma omp parallel for shared(gradient)
        for (size_t k = 1; k < _pop_size; ++k) { // Loops over all population configurations
            VectorXui strategies = VectorXui::Zero(_nb_strategies);
            Group group(_nb_strategies, _group_size, w, strategies, _payoff_matrix);
            group.set_group_size(_group_size);
            std::vector<Group> groups(_nb_groups, group);
            std::vector<size_t> pop_container(_pop_size);
            size_t t_plus = 0; // resident to mutant count
            size_t t_minus = 0; // resident to resident count
            strategies(resident) = _pop_size - k;
            strategies(invader) = k;
            // initialize container
            for (size_t i = 0; i < k; ++i) pop_container[i] = invader;
            for (size_t i = k; i < _pop_size; ++i) pop_container[i] = resident;

            // Calculate T+ and T-
            for (size_t i = 0; i < runs; ++i) {
                // First we initialize a homogeneous population with the resident strategy
                _setState(groups, pop_container);
                _update(q, groups, strategies);
                auto sum = static_cast<double>(strategies(invader) + strategies(resident));
                if (static_cast<double>(strategies(invader)) / sum >
                    static_cast<double>(k) / static_cast<double>(_pop_size)) {
                    ++t_plus;
                } else if (static_cast<double>(strategies(invader)) / sum <
                           static_cast<double>(k) / static_cast<double>(_pop_size)) {
                    ++t_minus;
                }
                strategies(resident) = _pop_size - k;
                strategies(invader) = k;
            }
            // Calculate gradient
            gradient(k) = (static_cast<double>(t_plus) - static_cast<double>(t_minus)) / static_cast<double>(runs);
        }

        return gradient;
    }

    /**
    * @brief calculates the gradient of selection for an invading strategy and any initial state.
    *
    * Will return the difference between T+ and T- for each possible population configuration
    * when the is conformed only by the resident and the invading strategy.
    *
    * To estimate T+ - T- (the probability that the number of invaders increase/decrease in the population)
    * we run the simulation for population with k invaders and Z - k residents for @param run
    * times and average how many times did the number of invadors increase and decrease.
    *
    * @tparam S : group container
    * @param invader : index of the invading strategy
    * @param init_state : vector indicating the initial state of the population (how many individuals of each strategy)
    * @param runs : number of runs (to average the results)
    * @param w : intensity of selection
    * @param q : splitting probability
    * @return : an Eigen vector with the gradient of selection for each k/Z where k is the number of invaders.
    */
    template<typename S>
    Vector
    MLS<S>::gradientOfSelection(size_t invader, size_t reduce, const Eigen::Ref<const VectorXui> &init_state,
                                size_t runs, double w, double q) {
        if (invader > _nb_strategies)
            throw std::invalid_argument(
                    "you must specify a valid index for invader and resident [0, " + std::to_string(_nb_strategies) +
                    ")");
        if (_nb_groups == 1 && q > 0.0)
            throw std::invalid_argument(
                    "The splitting probability must be zero when there is only 1 group in the population");

        if (init_state.sum() != _pop_size)
            throw std::invalid_argument(
                    "the sum of individuals in the initial state must be equal to " + std::to_string(_pop_size));

        Vector gradient = Vector::Zero(init_state(reduce) + 1);

        // This loop can be done in parallel
#pragma omp parallel for shared(gradient)
        for (size_t k = 0; k <= init_state(reduce); ++k) { // Loops over all population configurations
            VectorXui strategies = VectorXui::Zero(_nb_strategies);
            Group group(_nb_strategies, _group_size, w, strategies, _payoff_matrix);
            group.set_group_size(_group_size);
            std::vector<Group> groups(_nb_groups, group);
            std::vector<size_t> pop_container(_pop_size);
            size_t t_plus = 0; // resident to mutant count
            size_t t_minus = 0; // resident to resident count
            // initialize container
            size_t z = 0;
            for (size_t i = 0; i < _nb_strategies; ++i) {
                if (i == invader) strategies(i) = k;
                else if (i == reduce) strategies(i) = init_state(i) - k;
                else strategies(i) = init_state(i);
                for (size_t j = 0; j < strategies(i); ++j) {
                    pop_container[z++] = i;
                }
            }

            // Calculate T+ and T-
            for (size_t i = 0; i < runs; ++i) {
                // First we initialize a homogeneous population with the resident strategy
                _setState(groups, pop_container);
                _update(q, groups, strategies);
                auto sum = static_cast<double>(strategies.sum());
                if (strategies(invader) / sum > k / static_cast<double>(_pop_size)) {
                    ++t_plus;
                } else if (strategies(invader) / sum < k / static_cast<double>(_pop_size)) {
                    ++t_minus;
                }
                strategies.array() = init_state;
                strategies(invader) = k;
                strategies(reduce) -= k;
            }
            // Calculate gradient
            gradient(k) = (static_cast<double>(t_plus) - static_cast<double>(t_minus)) / static_cast<double>(runs);
        }

        return gradient;
    }

    template<typename S>
    void MLS<S>::_update(double q, std::vector<S> &groups, VectorXui &strategies) {
        _reproduce(groups, strategies, q);
    }

    template<typename S>
    void MLS<S>::_update(double q, double lambda, std::vector<S> &groups, VectorXui &strategies) {
        _reproduce(groups, strategies, q);
        if (_real_rand(_mt) < lambda) _migrate(q, groups, strategies);
    }

    template<typename S>
    void
    MLS<S>::_update(double q, double lambda, double mu, std::vector<S> &groups, VectorXui &strategies) {
        _reproduce(groups, strategies, q);
        if (_real_rand(_mt) < lambda) _migrate(q, groups, strategies);
        if (_real_rand(_mt) < mu) _mutate(groups, strategies);
    }

    template<typename S>
    void MLS<S>::_speedUpdate(double q, std::vector<S> &groups, VectorXui &strategies) {
        if (!_pseudoStationary(groups)) {
            _reproduce(groups, strategies, q);
        } else { // If the groups have reached maximum size and the population is monomorphic
            _reproduce(groups, strategies);
        }
    }

    template<typename S>
    void MLS<S>::_speedUpdate(double q, double lambda, std::vector<S> &groups, VectorXui &strategies) {
        if (!_pseudoStationary(groups)) {
            _reproduce(groups, strategies, q);
            if (_real_rand(_mt) < lambda) _migrate(q, groups, strategies);
        } else { // If the groups have reached maximum size and the population is monomorphic
            if ((_real_rand(_mt) * (q + lambda)) < q) _reproduce(groups, strategies);
            else _migrate(q, groups, strategies);
        }
    }

    template<typename S>
    void
    MLS<S>::_speedUpdate(double q, double lambda, double mu, std::vector<S> &groups,
                         VectorXui &strategies) {
        if (!_pseudoStationary(groups)) {
            _reproduce(groups, strategies, q);
            if (_real_rand(_mt) < lambda) _migrate(q, groups, strategies);
            if (_real_rand(_mt) < mu) _mutate(groups, strategies);
        } else { // If the groups have reached maximum size and the population is monomorphic
            double p = _real_rand(_mt) * (q + lambda + mu);
            if (p <= q) _reproduce(groups, strategies);
            else if (p <= (q + lambda)) _migrate(q, groups, strategies);
            else _mutate(groups, strategies);
        }
    }

    template<typename S>
    void MLS<S>::_createMutant(size_t invader, size_t resident, std::vector<S> &groups) {
        auto mutate_group = _uint_rand(_mt);
        groups[mutate_group].createMutant(invader, resident);
    }

    /**
     * @brief Adds a mutant of strategy invader to the population
     *
     * Eliminates a randmo strategy from the population and adds a mutant of strategy invader.
     *
     * @tparam S : container for population structur
     * @param invader : index of the invader strategy
     * @param groups : vector of groups
     * @param strategies : vector of strategies
     */
    template<typename S>
    void MLS<S>::_createRandomMutant(size_t invader, std::vector<S> &groups, EGTTools::VectorXui &strategies) {
        auto mutate_group = _uint_rand(_mt);
        size_t mutating_strategy = groups[mutate_group].deleteMember(_mt);
        groups[mutate_group].addMember(invader);
        --strategies(mutating_strategy);
        ++strategies(invader);
    }

    template<typename S>
    void MLS<S>::_updateFullPopulationFrequencies(size_t increase, size_t decrease,
                                                  EGTTools::VectorXui &strategies) {
        ++strategies(increase);
        --strategies(decrease);
    }

/**
 * @brief internal reproduction function.
 *
 * This function always splits the group
 *
 * @tparam S : group container
 */
    template<typename S>
    void MLS<S>::_reproduce(std::vector<S> &groups, VectorXui &strategies) {
        auto parent_group = _payoffProportionalSelection(groups);
        auto[split, new_strategy] = groups[parent_group].createOffspring(_mt);
        ++strategies(new_strategy);
        _splitGroup(parent_group, groups, strategies);
    }

/**
 * @brief internal reproduction function.
 *
 * This functions will split depending a splitting probability q.
 *
 * @tparam S : group container
 * @param groups : vector of groups
 * @param strategies : vector of the current proportions of each strategy in the population
 * @param q : split probability
 */
    template<typename S>
    void MLS<S>::_reproduce(std::vector<S> &groups, VectorXui &strategies, double q) {
        auto parent_group = _payoffProportionalSelection(groups);
        auto[split, new_strategy] = groups[parent_group].createOffspring(_mt);
        ++strategies(new_strategy);
        if (split) {
            if (_real_rand(_mt) < q) { // split group
                _splitGroup(parent_group, groups, strategies);
            } else { // remove individual
                size_t deleted_strategy = groups[parent_group].deleteMember(_mt);
                --strategies(deleted_strategy);
            }
        }
    }

/**
 * @brief Migrates an individual from a group to another
 *
 * @tparam S : group container
 * @param q : splitting probability
 * @param groups : reference to a vector of groups
 */
    template<typename S>
    void MLS<S>::_migrate(double q, std::vector<S> &groups, VectorXui &strategies) {
        size_t parent_group, child_group, migrating_strategy;

        parent_group = _sizeProportionalSelection(groups);
        while (groups[parent_group].group_size() < 2) parent_group = _uint_rand(_mt);
        child_group = _uint_rand(_mt);
        // First we delete a random member from the parent group
        migrating_strategy = groups[parent_group].deleteMember(_mt);
        // Then add the member to the child group
        if (groups[child_group].addMember(migrating_strategy)) {
            if (_real_rand(_mt) < q) _splitGroup(child_group, groups, strategies);
            else { // in case we delete a random member, that strategy will diminish in the population
                migrating_strategy = groups[child_group].deleteMember(_mt);
                --strategies(migrating_strategy);
            }
        }
    }

/**
 * @brief Mutates an individual from the population
 *
 * @tparam S : group container
 * @param mu
 * @param groups : reference to a vector of groups
 */
    template<typename S>
    void MLS<S>::_mutate(std::vector<S> &groups, VectorXui &strategies) {
        size_t parent_group, mutating_strategy, new_strategy;

        parent_group = _sizeProportionalSelection(groups);
        mutating_strategy = groups[parent_group].deleteMember(_mt);
        new_strategy = _uint_rand_strategy(_mt);
        while (mutating_strategy == new_strategy) new_strategy = _uint_rand_strategy(_mt);
        groups[parent_group].addMember(new_strategy);
        --strategies(mutating_strategy);
        ++strategies(new_strategy);
    }

/**
 * @brief splits a group in two
 *
 * This method creates a new group. There is a 0.5 probability that each
 * member of the former group will be part of the new group. Also, since
 * the number of groups is kept constant, a random group is chosen to die.
 *
 * @tparam S : group container
 * @param parent_group : index to the group to split
 * @param groups : reference to a vector of groups
 */
    template<typename S>
    void MLS<S>::_splitGroup(size_t parent_group, std::vector<S> &groups, VectorXui &strategies) {
        // First choose a group to die
        size_t child_group = _uint_rand(_mt);
        while (child_group == parent_group) child_group = _uint_rand(_mt);
        // Now we split the group
        VectorXui &strategies_parent = groups[parent_group].strategies();
        VectorXui &strategies_child = groups[child_group].strategies();

        // update strategies with the eliminated strategies from the child group
        strategies -= strategies_child;
        strategies_child.setZero();
        // vector of binomial distributions for each strategy (this will be used to select the members
        // that go to the child group
        std::binomial_distribution<size_t> binomial(_group_size, 0.5);
        size_t sum = 0;
        while ((sum == 0) || (sum > _group_size)) {
            sum = 0;
            for (size_t i = 0; i < _nb_strategies; ++i) {
                if (strategies_parent(i) > 0) {
                    binomial.param(std::binomial_distribution<size_t>::param_type(strategies_parent(i), 0.5));
                    strategies_child(i) = binomial(_mt);
                    sum += strategies_child(i);
                }
            }
        }
        // reset group size
        groups[child_group].set_group_size(sum);
        groups[parent_group].set_group_size(groups[parent_group].group_size() - sum);
        // reset parent group strategies
        strategies_parent -= strategies_child;
    }

/**
 * @brief selects a group proportional to its total payoff.
 *
 * @tparam S : group container
 * @param groups : reference to the population groups
 * @return : index of the parent group
 */
    template<typename S>
    size_t MLS<S>::_payoffProportionalSelection(std::vector<S> &groups) {
        double total_fitness = 0.0, tmp = 0.0;
        // Calculate total fitness
        for (auto &group: groups) total_fitness += group.totalPayoff();
        total_fitness *= _real_rand(_mt);
        size_t parent_group = 0;
        for (parent_group = 0; parent_group < _nb_groups; ++parent_group) {
            tmp += groups[parent_group].group_fitness();
            if (tmp > total_fitness) return parent_group;
        }

        return 0;
    }

/**
 * @brief selects a group proportional to its size.
 *
 * @tparam S : group container
 * @param groups : reference to the population groups
 * @return : index of the parent group
 */
    template<typename S>
    size_t MLS<S>::_sizeProportionalSelection(std::vector<S> &groups) {
        size_t pop_size = _current_pop_size(groups), tmp = 0;
        std::uniform_int_distribution<size_t> dist(0, pop_size - 1);
        // Calculate total fitness
        size_t p = dist(_mt);
        size_t parent_group = 0;
        for (parent_group = 0; parent_group < _nb_groups; ++parent_group) {
            tmp += groups[parent_group].group_size();
            if (tmp > p) return parent_group;
        }

        return 0;
    }

/**
 * @brief Checks whether a pseudo stationary state has been reached.
 *
 * @tparam S : group container
 * @param groups : reference to a vector of groups
 * @return true if reached a pseudo stationary state, otherwise false
 */
    template<typename S>
    bool MLS<S>::_pseudoStationary(std::vector<S> &groups) {
        if (_current_pop_size(groups) < _pop_size) return false;
        for (auto &group: groups)
            if (!group.isPopulationMonomorphic())
                return false;

        return true;
    }

/**
 * @brief sets the all individuals of one strategy
 *
 * Sets all individuals in the population of one strategy and sets all groups at maximum capacity.
 *
 * @tparam S : group container
 * @param strategy : resident strategy
 * @param groups : reference to a vector of groups
 */
    template<typename S>
    void MLS<S>::_setFullHomogeneousState(size_t strategy, std::vector<S> &groups) {
        for (auto &group: groups)
            group.setPopulationHomogeneous(strategy);
    }

/**
 * @brief returns the total population size
 * @tparam S : group container
 * @param groups : reference to a vector of groups
 * @return the sum of the sizes of all the groups
 */
    template<typename S>
    size_t MLS<S>::_current_pop_size(std::vector<S> &groups) {
        size_t size = 0;
        for (auto &group: groups) size += group.group_size();

        return size;
    }

    /**
     * @brief Sets randomly the state of the population given a vector which contains the strategie sin the population.
     *
     * This method shuffles a vector containing the population of strategies and then assigns each _group_size
     * of strategies to a group.
     *
     * @tparam S : container for the groups
     * @param groups : vector of groups
     * @param container : vector of strategies
     */
    template<typename S>
    void MLS<S>::_setState(std::vector<S> &groups, std::vector<size_t> &container) {
        // Then we shuffle it randomly the contianer
        std::shuffle(container.begin(), container.end(), _mt);

        // Now we randomly initialize the groups with the population configuration from strategies
        for (size_t i = 0; i < _nb_groups; ++i) {
            groups[i].set_group_size(_group_size);
            VectorXui &group_strategies = groups[i].strategies();
            group_strategies.setZero();
            for (size_t j = 0; j < _group_size; ++j) {
                ++group_strategies(container[j + (i * _group_size)]);
            }
        }
    }
}

#endif //DYRWIN_SED_MLS_HPP
