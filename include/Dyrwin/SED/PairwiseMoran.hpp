//
// Created by Elias Fernandez on 2019-06-25.
//

#ifndef DYRWIN_SED_PAIRWISEMORAN_HPP
#define DYRWIN_SED_PAIRWISEMORAN_HPP

#include <Dyrwin/SeedGenerator.h>
#include <Dyrwin/Types.h>
#include <Dyrwin/LruCache.hpp>
#include <Dyrwin/SED/Utils.hpp>
#include <Dyrwin/OpenMPUtils.hpp>

namespace EGTTools::SED {
    /**
     * This class caches the results according to the specified class in the template
     * parameter.
     *
     * @tparam Cache
     */
    template<class Cache=EGTTools::Utils::LRUCache<std::string, double>>
    class PairwiseMoran {
    public:
        /**
         * @brief Implements the Pairwise comparison Moran process.
         *
         * The selection dynamics implemented in this class are as follows:
         * At each generation 2 players are selected at random from the whole population.
         * Their fitness is compared according to the fermi function which returns a probability
         * that defines the likelihood that the first player will imitate the second.
         *
         * This process may include mutation.
         *
         * This class uses a cache to accelerate the computations.
         *
         * @param nb_strategies
         * @param pop_size
         * @param group_size : size of the group
         * @param game : pointer to the game class (it must be a child of AbstractGame)
         */
        PairwiseMoran(size_t pop_size, EGTTools::SED::AbstractGame &game);

        /**
         * Runs the moran process for a given number of generations or until it reaches a monomorphic state
         *
         * @param nb_generations : maximum number of generations
         * @param beta : intensity of selection
         * @param mu: mutation probability
         * @param init_state : initial state of the population
         * @return a vector with the final state of the population
         */
        VectorXui
        evolve(size_t nb_generations, double beta, double mu, const Eigen::Ref<const VectorXui> &init_state);

        /**
         * @brief Estimates the fixation probability of the invading strategy over the resident strategy.
         *
         * This function will estimate numerically the fixation probability of an @param invader strategy
         * in a population of @param resident strategies.
         *
         * @param invader : index of the invading strategy
         * @param resident : index of the resident strategy
         * @param runs : number of independent runs (the estimation improves with the number of runs)
         * @param nb_generations : maximum number of generations per run
         * @param beta : intensity of selection
         * @return the fixation probability of the invader strategy
         */
        double fixationProbability(size_t invader, size_t resident, size_t runs, size_t nb_generations, double beta,
                                   double mu);

        /**
         * @brief Estimates the stationary distribution of the population of strategies in the game.
         *
         * The estimation of the stationary distribution is done by calculating averaging the fraction of
         * the population of each strategy at the end of each trial over all trials.
         *
         * @param nb_runs : number of trials used to estimate the stationary distribution
         * @param nb_generations : number of generations per trial
         * @param beta : intensity of selection
         * @param mu : mutation probability
         * @return the stationary distribution
         */
        Vector stationaryDistribution(size_t nb_runs, size_t nb_generations, double beta, double mu);

        // Getters
        size_t nb_strategies() const;

        size_t population_size() const;

        std::string game_type() const;

        const GroupPayoffs &payoffs() const;

        // Setters
        void set_population_size(size_t pop_size);

        void change_game(EGTTools::SED::AbstractGame &game);


    private:
        size_t _nb_strategies, _pop_size;
        EGTTools::SED::AbstractGame &_game;

        // Random distributions
        std::uniform_int_distribution<size_t> _pop_sampler;
        std::uniform_int_distribution<size_t> _strategy_sampler;
        std::uniform_real_distribution<double> _real_rand;

        // Random generators
        std::mt19937_64 _mt{EGTTools::Random::SeedGenerator::getInstance().getSeed()};

        inline std::pair<size_t, size_t> _sample_players();

        inline double
        _calculate_fitness(const size_t &player_type, VectorXui &strategies, Cache &cache);

        inline void _initialise_population(const VectorXui &strategies, std::vector<size_t> &population);

        inline std::pair<bool, size_t> _is_homogeneous(VectorXui &strategies);
    };

    template<class Cache>
    PairwiseMoran<Cache>::PairwiseMoran(size_t pop_size,
                                        EGTTools::SED::AbstractGame &game) : _nb_strategies(game.nb_strategies()),
                                                                             _pop_size(pop_size),
                                                                             _game(game) {
        // Initialize random uniform distribution
        _pop_sampler = std::uniform_int_distribution<size_t>(0, _pop_size - 1);
        _strategy_sampler = std::uniform_int_distribution<size_t>(0, _nb_strategies - 1);
        _real_rand = std::uniform_real_distribution<double>(0.0, 1.0);
    }

    template<class Cache>
    void
    PairwiseMoran<Cache>::_initialise_population(const VectorXui &strategies, std::vector<size_t> &population) {
        size_t z = 0;
        for (unsigned int i = 0; i < _nb_strategies; ++i) {
            for (size_t j = 0; j < strategies(i); ++j) {
                population[z++] = i;
            }
        }

        // Then we shuffle it randomly
        std::shuffle(population.begin(), population.end(), _mt);
    }

    template<class Cache>
    VectorXui
    PairwiseMoran<Cache>::evolve(size_t nb_generations, double beta, double mu,
                                 const Eigen::Ref<const VectorXui> &init_state) {
        size_t die, birth;
        std::vector<size_t> population(_pop_size, 0);
        VectorXui strategies(_nb_strategies);
        // Initialise strategies from init_state
        strategies.array() = init_state;

        // Avg. number of rounds for a mutation to happen
        size_t nb_generations_for_mutation = std::floor(1 / mu);
        auto[homogeneous, idx_homo] = _is_homogeneous(strategies);

        // Creates a cache for the fitness data
        EGTTools::Utils::LRUCache<std::string, double> cache(10000000);

        // initialise population
        _initialise_population(strategies, population);

        // Now we start the imitation process
        for (size_t i = 0; i < nb_generations; ++i) {
            // First we pick 2 players randomly
            auto[player1, player2] = _sample_players();

            if (homogeneous) {
                if (mu > 0) {
                    i += nb_generations_for_mutation;
                    // mutate
                    birth = _strategy_sampler(_mt);
                    // If population still homogeneous we wait for another mutation
                    while (birth == idx_homo) {
                        i += nb_generations_for_mutation;
                        birth = _strategy_sampler(_mt);
                    }
                    if (i < nb_generations) {
                        population[player1] = birth;
                        strategies(birth) += 1;
                        strategies(idx_homo) -= 1;
                        homogeneous = false;
                    }
                    continue;
                } else break;
            }

            // Check if player mutates
            if (_real_rand(_mt) < mu) {
                die = population[player1];
                birth = _strategy_sampler(_mt);
                population[player1] = birth;
            } else { // If no mutation, player imitates
                // Then we let them play to calculate their payoffs
                auto fitness_p1 = _calculate_fitness(population[player1], strategies, cache);
                auto fitness_p2 = _calculate_fitness(population[player2], strategies, cache);

                // Then we apply the moran process with mutation
                if (_real_rand(_mt) < EGTTools::SED::fermi(beta, fitness_p1, fitness_p2)) {
                    // player 1 copies player 2
                    die = population[player1];
                    birth = population[player2];
                    population[player1] = birth;
                } else {
                    // player 2 copies player 1
                    die = population[player2];
                    birth = population[player1];
                    population[player2] = birth;
                }
            }
            strategies(birth) += 1;
            strategies(die) -= 1;
            // Check if population is homogeneous
            if (strategies(birth) == _pop_size) {
                homogeneous = true;
                idx_homo = birth;
            }
        }

        return strategies;
    }

    template<class Cache>
    double
    PairwiseMoran<Cache>::fixationProbability(size_t invader, size_t resident, size_t runs, size_t nb_generations,
                                              double beta, double mu) {
        if (invader > _nb_strategies || resident > _nb_strategies)
            throw std::invalid_argument(
                    "you must specify a valid index for invader and resident [0, " + std::to_string(_nb_strategies) +
                    ")");

        double r2m = 0; // resident to mutant count
        double r2r = 0; // resident to resident count

        // This loop can be done in parallel
#pragma omp parallel for reduction(+:r2m, r2r)
        for (size_t i = 0; i < runs; ++i) {
            // First we initialize a homogeneous population with the resident strategy
            VectorXui strategies = VectorXui::Zero(_nb_strategies);
            strategies(resident) = _pop_size - 1;
            strategies(invader) = 1;

            // Then we run the Moran Process
            auto final_state = evolve(nb_generations, beta, mu, strategies);

            if (final_state(invader) == 0) {
                r2r += 1.0;
            } else if (final_state(resident) == 0) {
                r2m += 1.0;
            }
        } // end runs loop
        if ((r2m == 0.0) && (r2r == 0.0)) return 0.0;
        else return r2m / (r2m + r2r);
    }

    template<class Cache>
    Vector PairwiseMoran<Cache>::stationaryDistribution(size_t nb_runs, size_t nb_generations, double beta,
                                                        double mu) {
        // First we initialise the container for the stationary distribution
        auto total_nb_states = EGTTools::binomialCoeff(_nb_strategies + _pop_size - 1, _pop_size);
        auto sampler = std::uniform_int_distribution<size_t>(0, total_nb_states - 1);
        VectorXui sdist = VectorXui::Zero(_nb_strategies);

//#pragma omp parallel for reduction(+:sdist)
        for (size_t i = 0; i < nb_runs; ++i) {
            // Then we sample a random population state
            VectorXui init_state = VectorXui::Zero(_nb_strategies);
            // Sample state
            EGTTools::SED::sample_simplex(sampler(_mt), _pop_size, _nb_strategies, init_state);
//            assert(init_state.sum() == _pop_size);

            // Finally we call the evolve function
            sdist += evolve(nb_generations, beta, mu, init_state);
        }
        return sdist.cast<double>() / nb_runs;
    }

    template<class Cache>
    std::pair<size_t, size_t> PairwiseMoran<Cache>::_sample_players() {
        auto player1 = _pop_sampler(_mt);
        auto player2 = _pop_sampler(_mt);
        while (player2 == player1) player2 = _pop_sampler(_mt);
        return std::make_pair(player1, player2);
    }

    template<class Cache>
    double
    PairwiseMoran<Cache>::_calculate_fitness(const size_t &player_type, VectorXui &strategies, Cache &cache) {
        double fitness;
        std::stringstream result;
        result << strategies;

        std::string key = std::to_string(player_type) + result.str();

        // First we check if fitness value is in the lookup table
        if (!cache.exists(key)) {
            strategies(player_type) -= 1;
            fitness = _game.calculate_fitness(player_type, _pop_size, strategies);
            strategies(player_type) += 1;

            // Finally we store the new fitness in the Cache. We also keep a Cache for the payoff given each group combination
            cache.insert(key, fitness);
        } else {
            fitness = cache.get(key);
        }

        return fitness;
    }

    template<class Cache>
    std::pair<bool, size_t> PairwiseMoran<Cache>::_is_homogeneous(VectorXui &strategies) {
        for (size_t i = 0; i < _nb_strategies; ++i) {
            if (strategies(i) == _pop_size) return std::make_pair(true, i);
        }
        return std::make_pair(false, -1);
    }

    template<class Cache>
    size_t PairwiseMoran<Cache>::nb_strategies() const {
        return _nb_strategies;
    }

    template<class Cache>
    size_t PairwiseMoran<Cache>::population_size() const {
        return _pop_size;
    }

    template<class Cache>
    std::string PairwiseMoran<Cache>::game_type() const {
        return _game.type();
    }

    template<class Cache>
    const GroupPayoffs &PairwiseMoran<Cache>::payoffs() const {
        return _game.payoffs();
    }

    template<class Cache>
    void PairwiseMoran<Cache>::set_population_size(size_t pop_size) {
        _pop_size = pop_size;
    }

    template<class Cache>
    void PairwiseMoran<Cache>::change_game(EGTTools::SED::AbstractGame &game) {
        _game = game;
    }
}

#endif //DYRWIN_PAIRWISEMORAN_HPP
