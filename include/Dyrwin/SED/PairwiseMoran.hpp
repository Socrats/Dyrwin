//
// Created by Elias Fernandez on 2019-06-25.
//

#ifndef DYRWIN_SED_PAIRWISEMORAN_HPP
#define DYRWIN_SED_PAIRWISEMORAN_HPP

#include <Dyrwin/SeedGenerator.h>
#include <Dyrwin/Types.h>
#include <Dyrwin/LruCache.hpp>
#include <Dyrwin/SED/games/AbstractGame.hpp>
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
         * @param cache_size : maximum number of elements in the cache
         */
        PairwiseMoran(size_t pop_size, EGTTools::SED::AbstractGame &game, size_t cache_size = 1000000);

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
         * Runs the moran process for a given number of generations or until it reaches a monomorphic state
         *
         * @param nb_generations : maximum number of generations
         * @param beta : intensity of selection
         * @param strategies : reference vector with the initial state of the population
         * @param generator : random engine
         */
        void evolve(size_t nb_generations, double beta, VectorXui &strategies, std::mt19937_64 &generator);

        /**
         * Runs the moran process for a given number of generations or until it reaches a monomorphic state
         *
         * @param nb_generations : maximum number of generations
         * @param beta : intensity of selection
         * @param mu: mutation probability
         * @param init_state : initial state of the population
         * @param generator : random engine
         * @return a vector with the final state of the population
         */
        VectorXui
        evolve(size_t nb_generations, double beta, double mu, const Eigen::Ref<const VectorXui> &init_state,
               std::mt19937_64 &generator);

        /**
         * @brief Runs a moran process with social imitation
         *
         * Runs the moran process for a given number of generations and returns
         * all the states the simulation went through.
         *
         * @param nb_generations : maximum number of generations
         * @param beta : intensity of selection
         * @param mu: mutation probability
         * @param init_state : initial state of the population
         * @return a matrix with all the states the system went through during the simulation
         */
        MatrixXui2D run(size_t nb_generations, double beta, double mu, const Eigen::Ref<const VectorXui> &init_state);

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
         * @param mu : mutation probability
         * @return the fixation probability of the invader strategy
         */
        double fixationProbability(size_t invader, size_t resident, size_t runs, size_t nb_generations, double beta);

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

        size_t cache_size() const;

        std::string game_type() const;

        const GroupPayoffs &payoffs() const;

        // Setters
        void set_population_size(size_t pop_size);

        void set_cache_size(size_t cache_size);

        void change_game(EGTTools::SED::AbstractGame &game);


    private:
        size_t _nb_strategies, _pop_size, _cache_size;
        EGTTools::SED::AbstractGame &_game;

        // Random distributions
        std::uniform_int_distribution<size_t> _pop_sampler;
        std::uniform_int_distribution<size_t> _strategy_sampler;
        std::uniform_real_distribution<double> _real_rand;

        // Random generators
        std::mt19937_64 _mt{EGTTools::Random::SeedGenerator::getInstance().getSeed()};

        /**
         * @brief updates the population of strategies one step
         * @param s1 : index of strategy 1
         * @param s2 : index of strategy 2
         * @param beta : intensity of selection
         * @param birth : container for the index of the birth strategy
         * @param die : container for the index of the die strategy
         * @param strategies : vector of strategy counts
         * @param cache : reference to cache container
         * @param generator : random generator
         */
        inline void
        _update_step(size_t s1, size_t s2, double beta, size_t &birth, size_t &die, VectorXui &strategies,
                     Cache &cache,
                     std::mt19937_64 &generator);

        /**
         * @brief updates the population of strategies and return the number of steps
         * @param s1 : index of strategy 1
         * @param s2 : index of strategy 2
         * @param beta : intensity of selection
         * @param mu : mutation probability
         * @param nb_generations : maximum number of generations
         * @param birth : container for the index of the birth strategy
         * @param die : container for the index of the die strategy
         * @param homogeneous : container indicating whether the population is homogeneous
         * @param idx_homo : container indicating the index of the homogeneous strategy
         * @param strategies : vector of strategy counts
         * @param cache : reference to cache container
         * @param geometric : geometric distribution of steps for a mutation to occur
         * @param generator : random generator
         * @return the number of steps that the update takes.
         */
        inline size_t
        _update_multi_step(size_t s1, size_t s2, double beta, double mu, size_t nb_generations,
                           size_t &birth, size_t &die, bool &homogeneous, size_t &idx_homo,
                           VectorXui &strategies,
                           Cache &cache, std::geometric_distribution<size_t> &geometric,
                           std::mt19937_64 &generator);

        inline std::pair<size_t, size_t> _sample_players();

        inline std::pair<size_t, size_t> _sample_players(std::mt19937_64 &generator);

        /**
         * @brief samples 2 players from the population of strategies and updates references @param s1 and s2.
         * @param s1 : reference container for strategy 1
         * @param s2 : reference container for strategy 2
         * @param strategies : vector of strategy counts
         * @param generator : random generator
         * @return true if the sampled strategies are equal, otherwise false
         */
        inline bool _sample_players(size_t &s1, size_t &s2, VectorXui &strategies, std::mt19937_64 &generator);

        inline double
        _calculate_fitness(const size_t &player_type, VectorXui &strategies, Cache &cache);

        inline void _initialise_population(const VectorXui &strategies, std::vector<size_t> &population);

        inline std::pair<bool, size_t> _is_homogeneous(VectorXui &strategies);
    };

    template<class Cache>
    PairwiseMoran<Cache>::PairwiseMoran(size_t pop_size,
                                        EGTTools::SED::AbstractGame &game,
                                        size_t cache_size) : _pop_size(pop_size),
                                                             _cache_size(cache_size),
                                                             _game(game) {
        // Initialize random uniform distribution
        _nb_strategies = game.nb_strategies();
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
        std::geometric_distribution<size_t> geometric(1 - mu);
        auto[homogeneous, idx_homo] = _is_homogeneous(strategies);

        // Creates a cache for the fitness data
        Cache cache(_cache_size);

        // initialise population
        _initialise_population(strategies, population);

        // Now we start the imitation process
        for (size_t i = 0; i < nb_generations; ++i) {
            // First we pick 2 players randomly
            auto[player1, player2] = _sample_players();

            if (homogeneous) {
                if (mu > 0) {
                    i += geometric(_mt);
                    // mutate
                    birth = _strategy_sampler(_mt);
                    // If population still homogeneous we wait for another mutation
                    while (birth == idx_homo) {
                        i += geometric(_mt);
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
    void
    PairwiseMoran<Cache>::evolve(size_t nb_generations, double beta, VectorXui &strategies,
                                 std::mt19937_64 &generator) {
        // This method runs a Moran process with pairwise comparison
        // using the fermi rule and no mutation

        size_t die, birth, strategy_p1 = 0, strategy_p2 = 0;

        // Check if initial state is already homogeneous, in which case return
        if ((strategies.array() == _pop_size).any()) return;

        // Creates a cache for the fitness data
        Cache cache(_cache_size);

        // Now we start the imitation process
        for (size_t i = 0; i < nb_generations; ++i) {
            // First we pick 2 players randomly
            // If the strategies are the same, there will be no change in the population
            if (_sample_players(strategy_p1, strategy_p2, strategies, generator)) continue;

            _update_step(strategy_p1, strategy_p2, beta, birth, die, strategies, cache, generator);

            // Check if population is homogeneous
            if (strategies(birth) == _pop_size) break;
        }
    }

    template<class Cache>
    VectorXui
    PairwiseMoran<Cache>::evolve(size_t nb_generations, double beta, double mu,
                                 const Eigen::Ref<const VectorXui> &init_state, std::mt19937_64 &generator) {
        size_t die, birth;
        std::vector<size_t> population(_pop_size, 0);
        VectorXui strategies(_nb_strategies);
        // Initialise strategies from init_state
        strategies.array() = init_state;

        // Avg. number of rounds for a mutation to happen
        std::geometric_distribution<size_t> geometric(1 - mu);
        auto[homogeneous, idx_homo] = _is_homogeneous(strategies);

        // Creates a cache for the fitness data
        Cache cache(_cache_size);

        // initialise population
        _initialise_population(strategies, population);

        // Now we start the imitation process
        for (size_t i = 0; i < nb_generations; ++i) {
            // First we pick 2 players randomly
            auto[player1, player2] = _sample_players(generator);

            if (homogeneous) {
                if (mu > 0) {
                    i += geometric(generator);
                    // mutate
                    birth = _strategy_sampler(generator);
                    // If population still homogeneous we wait for another mutation
                    while (birth == idx_homo) {
                        i += geometric(generator);
                        birth = _strategy_sampler(generator);
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
            if (_real_rand(generator) < mu) {
                die = population[player1];
                birth = _strategy_sampler(generator);
                population[player1] = birth;
            } else { // If no mutation, player imitates
                // Then we let them play to calculate their payoffs
                auto fitness_p1 = _calculate_fitness(population[player1], strategies, cache);
                auto fitness_p2 = _calculate_fitness(population[player2], strategies, cache);

                // Then we apply the moran process with mutation
                if (_real_rand(generator) < EGTTools::SED::fermi(beta, fitness_p1, fitness_p2)) {
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
    MatrixXui2D PairwiseMoran<Cache>::run(size_t nb_generations, double beta, double mu,
                                          const Eigen::Ref<const EGTTools::VectorXui> &init_state) {
        size_t die, birth, strategy_p1 = 0, strategy_p2 = 0, current_generation = 1;
        std::vector<size_t> population(_pop_size, 0);
        MatrixXui2D states = MatrixXui2D::Zero(nb_generations, _nb_strategies);
        VectorXui strategies(_nb_strategies);
        // initialise initial state
        states.row(0) = init_state;
        strategies = init_state;

        // Distribution number of generations for a mutation to happen
        std::geometric_distribution<size_t> geometric(mu);

        // Check if state is homogeneous
        auto[homogeneous, idx_homo] = _is_homogeneous(strategies);

        // If it is we add a random mutant
        if (homogeneous) {
            current_generation += geometric(_mt);
            // mutate
            die = idx_homo;
            birth = _strategy_sampler(_mt);
            // If population still homogeneous we wait for another mutation
            while (birth == die) {
                current_generation += geometric(_mt);
                birth = _strategy_sampler(_mt);
            }
            if (current_generation < nb_generations) {
                strategies(birth) += 1;
                strategies(die) -= 1;
                homogeneous = false;
                states.block(1, 0, current_generation, _nb_strategies) = strategies;
            }
        }

        // Creates a cache for the fitness data
        Cache cache(_cache_size);

        for (size_t j = current_generation; j < nb_generations; ++j) {
            // First we pick 2 players randomly
            // If the strategies are the same, there will be no change in the population
            if (_sample_players(strategy_p1, strategy_p2, strategies, _mt)) {
                continue;
            }

            // Update with mutation and return how many steps should be added to the current
            // generation if the only change in the population could have been a mutation
            auto k = _update_multi_step(strategy_p1, strategy_p2, beta, mu, nb_generations,
                                        birth, die, homogeneous, idx_homo,
                                        strategies, cache,
                                        geometric, _mt);

            // Update state count by k steps
            j += k;
            // update all states until k + 1
            if (k > 0) states.block(j, 0, k, _nb_strategies) = strategies;
            else states.row(j) = strategies;
        }
        return states;
    }

    template<class Cache>
    double
    PairwiseMoran<Cache>::fixationProbability(size_t invader, size_t resident, size_t runs, size_t nb_generations,
                                              double beta) {
        if (invader >= _nb_strategies || resident >= _nb_strategies)
            throw std::invalid_argument(
                    "you must specify a valid index for invader and resident [0, " + std::to_string(_nb_strategies) +
                    ")");
        if (invader == resident) throw std::invalid_argument("mutant must be different from resident");

        long int r2m = 0; // resident to mutant count
        long int r2r = 0; // resident to resident count

        // This loop can be done in parallel
#pragma omp parallel for reduction(+:r2m, r2r)
        for (size_t i = 0; i < runs; ++i) {
            // Random generators - each thread should have its own generator
            std::mt19937_64 generator(EGTTools::Random::SeedGenerator::getInstance().getSeed());

            // First we initialize a homogeneous population with the resident strategy
            VectorXui strategies = VectorXui::Zero(_nb_strategies);
            strategies(resident) = _pop_size - 1;
            strategies(invader) = 1;

            // Then we run the Moran Process
            evolve(nb_generations, beta, strategies, generator);

            if (strategies(invader) == 0) {
                ++r2r;
            } else if (strategies(resident) == 0) {
                ++r2m;
            }
        } // end runs loop
        if ((r2m == 0) && (r2r == 0)) return 0.0;
        else return static_cast<double>(r2m) / (r2m + r2r);
    }

    template<class Cache>
    Vector PairwiseMoran<Cache>::stationaryDistribution(size_t nb_runs, size_t nb_generations, double beta, double mu) {
        // First we initialise the container for the stationary distribution
        auto total_nb_states = EGTTools::starsBars(_pop_size, _nb_strategies);
        auto sampler = std::uniform_int_distribution<size_t>(0, total_nb_states - 1);
        VectorXui avg_stationary_distribution = VectorXui::Zero(total_nb_states);
        // Distribution number of generations for a mutation to happen
        std::geometric_distribution<size_t> geometric(mu);

#pragma omp parallel for reduction(+:avg_stationary_distribution)
        for (size_t i = 0; i < nb_runs; ++i) {
            VectorXui sdist = VectorXui::Zero(total_nb_states);
            // Random generators - each thread should have its own generator
            std::mt19937_64 generator{EGTTools::Random::SeedGenerator::getInstance().getSeed()};

            // Then we sample a random population state
            VectorXui strategies = VectorXui::Zero(_nb_strategies);
            // Sample state
            auto current_state = sampler(generator);
            EGTTools::SED::sample_simplex(current_state, _pop_size, _nb_strategies, strategies);
            ++sdist(current_state);

            size_t die, birth, strategy_p1 = 0, strategy_p2 = 0, current_generation = 0;
            // Check if state is homogeneous
            auto[homogeneous, idx_homo] = _is_homogeneous(strategies);

            // If it is we add a random mutant
            if (homogeneous) {
                // mutate
                birth = _strategy_sampler(_mt);
                // If population still homogeneous we wait for another mutation
                while (birth == idx_homo) birth = _strategy_sampler(_mt);
                strategies(birth) += 1;
                strategies(idx_homo) -= 1;
                homogeneous = false;
                current_state = EGTTools::SED::calculate_state(_pop_size, strategies);
            }

            // Creates a cache for the fitness data
            Cache cache(_cache_size);

            for (size_t j = current_generation; j < nb_generations; ++j) {
                // First we pick 2 players randomly
                // If the strategies are the same, there will be no change in the population
                if (_sample_players(strategy_p1, strategy_p2, strategies, generator)) {
                    ++sdist(current_state);
                    continue;
                }

                // Update with mutation and return how many steps should be added to the current
                // generation if the only change in the population could have been a mutation
                auto k = _update_multi_step(strategy_p1, strategy_p2, beta, mu, nb_generations,
                                            birth, die, homogeneous, idx_homo,
                                            strategies, cache,
                                            geometric, generator);

                // Update state count by k steps
                current_state = EGTTools::SED::calculate_state(_pop_size, strategies);
                sdist(current_state) += k;
                j += k;
            }
            avg_stationary_distribution += sdist;
        }
        return avg_stationary_distribution.cast<double>() / (nb_runs * nb_generations);
    }

    template<class Cache>
    void PairwiseMoran<Cache>::_update_step(size_t s1, size_t s2, double beta, size_t &birth, size_t &die,
                                            VectorXui &strategies,
                                            Cache &cache,
                                            std::mt19937_64 &generator) {
        // Then we let them play to calculate their payoffs
        auto fitness_p1 = _calculate_fitness(s1, strategies, cache);
        auto fitness_p2 = _calculate_fitness(s2, strategies, cache);

        // Then we apply the moran process with mutation
        if (_real_rand(generator) < EGTTools::SED::fermi(beta, fitness_p1, fitness_p2)) {
            // player 1 copies player 2
            die = s1;
            birth = s2;
        } else {
            // player 2 copies player 1
            die = s2;
            birth = s1;
        }

        strategies(birth) += 1;
        strategies(die) -= 1;
    }

    template<class Cache>
    size_t
    PairwiseMoran<Cache>::_update_multi_step(size_t s1, size_t s2, double beta, double mu, size_t nb_generations,
                                             size_t &birth, size_t &die,
                                             bool &homogeneous, size_t &idx_homo,
                                             VectorXui &strategies,
                                             Cache &cache,
                                             std::geometric_distribution<size_t> &geometric,
                                             std::mt19937_64 &generator) {

        size_t k = 0;

        if (homogeneous) {
            k += geometric(generator);
            // mutate
            die = idx_homo;
            birth = _strategy_sampler(generator);
            // If population still homogeneous we wait for another mutation
            while (birth == die) {
                k += geometric(_mt);
                birth = _strategy_sampler(generator);
            }
            if (k < nb_generations) {
                strategies(birth) += 1;
                strategies(die) -= 1;
                homogeneous = false;
            }
        } else {
            // Check if player mutates
            if (_real_rand(generator) < mu) {
                die = s1;
                birth = _strategy_sampler(generator);
            } else { // If no mutation, player imitates

                // Then we let them play to calculate their payoffs
                auto fitness_p1 = _calculate_fitness(s1, strategies, cache);
                auto fitness_p2 = _calculate_fitness(s2, strategies, cache);

                // Then we apply the moran process with mutation
                if (_real_rand(generator) < EGTTools::SED::fermi(beta, fitness_p1, fitness_p2)) {
                    // player 1 copies player 2
                    die = s1;
                    birth = s2;
                } else {
                    // player 2 copies player 1
                    die = s2;
                    birth = s1;
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
        return k;
    }

    template<class Cache>
    std::pair<size_t, size_t> PairwiseMoran<Cache>::_sample_players() {
        auto player1 = _pop_sampler(_mt);
        auto player2 = _pop_sampler(_mt);
        while (player2 == player1) player2 = _pop_sampler(_mt);
        return std::make_pair(player1, player2);
    }

    template<class Cache>
    std::pair<size_t, size_t> PairwiseMoran<Cache>::_sample_players(std::mt19937_64 &generator) {
        auto player1 = _pop_sampler(generator);
        auto player2 = _pop_sampler(generator);
        while (player2 == player1) player2 = _pop_sampler(generator);
        return std::make_pair(player1, player2);
    }

    template<class Cache>
    bool
    PairwiseMoran<Cache>::_sample_players(size_t &s1, size_t &s2, VectorXui &strategies, std::mt19937_64 &generator) {
        // sample 2 strategies from the pool
        auto player1 = _pop_sampler(generator);
        auto player2 = _pop_sampler(generator);
        while (player2 == player1) player2 = _pop_sampler(generator);

        size_t tmp = 0;
        s1 = 0;
        s2 = 0;
        bool unset_p1 = true, unset_p2 = true;

        for (size_t i = 0; i < _nb_strategies; ++i) {
            tmp += strategies(i);
            if (tmp > player1 && unset_p1) {
                s1 = i;
                unset_p1 = false;
            }
            if (tmp > player2 && unset_p2) {
                s2 = i;
                unset_p2 = false;
            }
            if (!unset_p1 && !unset_p2) break;
        }
        return s1 == s2;
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
    size_t PairwiseMoran<Cache>::cache_size() const {
        return _cache_size;
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
    void PairwiseMoran<Cache>::set_cache_size(size_t cache_size) {
        _cache_size = cache_size;
    }

    template<class Cache>
    void PairwiseMoran<Cache>::change_game(EGTTools::SED::AbstractGame &game) {
        _game = game;
    }
}

#endif //DYRWIN_PAIRWISEMORAN_HPP
