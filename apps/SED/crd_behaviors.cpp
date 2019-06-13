//
// Created by Elias Fernandez on 2019-05-31.
//
#include <cmath>
#include <unordered_map>
#include <fstream>

#include <Dyrwin/SED/behaviors/CrdBehaviors.hpp>
#include <Dyrwin/Types.h>
#include <Dyrwin/SED/Utils.hpp>
#include <Dyrwin/Distributions.h>
#include <Dyrwin/LruCache.hpp>
#include <Dyrwin/RL/Utils.h>
#include <Dyrwin/SeedGenerator.h>
#include <Dyrwin/CommandLineParsing.h>
#include <Dyrwin/Utils.h>
#include <Dyrwin/SED/games/AbstractGame.hpp>
#include <Dyrwin/SED/games/CrdGame.hpp>
#include <Dyrwin/SED/games/CrdGameTU.hpp>

//using GroupPayoffs = std::unordered_map<size_t, double>;
using GroupPayoffs = EGTTools::Matrix2D;
using StrategyCounts = std::vector<size_t>;


size_t starsBars(size_t stars, size_t bins) {
    return EGTTools::binomialCoeff(stars + bins - 1, stars);
}

/**
 * @brief This function converts a vector containing counts into an index.
 *
 * This method was copies from @ebargiac
 *
 * @param data The vector to convert.
 * @param history The sum of the values contained in data.
 *
 * @return The unique index in [0, starsBars(history, data.size() - 1)) representing data.
 */
size_t
calculate_state(const size_t &group_size, const size_t &nb_states, const EGTTools::RL::Factors &current_group) {
    size_t retval = 0;
    auto remaining = group_size;

    // In order to find the index for the input combination, we are basically
    // counting the number of combinations we have 'behind us", and we're going
    // to be the next. So for example if we have 10 combinations behind us,
    // we're going to be number 11.
    //
    // We do this recursively, element by element. For each element we count
    // the number of combinations we left behind. If data[i] is the highest
    // possible (i.e. it accounts for all remaining points), then it is the
    // first and we didn't miss anything.
    //
    // Otherwise we count how many combinations we'd have had with the max (1),
    // add it to the number, and decrease the h. Then we try again: are we
    // still lower? If so, count again how many combinations we'd have had with
    // this number (size() - 1). And so on, until we match the number we have.
    //
    // Then we go to the next element, considering the subarray of one element
    // less (thus the size() - i), and we keep going.
    //
    // Note that by using this algorithm the last element in the array is never
    // needed (since it is determined by the others), and additionally when we
    // have no remaining elements to parse we can just break.
    for (size_t i = 0; i < current_group.size() - 1; ++i) {
        auto h = remaining;
        while (h > current_group[current_group.size() - i - 2]) {
            retval += starsBars(remaining - h, current_group.size() - i - 1);
            --h;
        }
        if (remaining == current_group[current_group.size() - i - 2])
            break;
        remaining -= current_group[current_group.size() - i - 2];
    }

    return nb_states - retval - 1;
}

GroupPayoffs calculate_payoffs(size_t group_size, size_t nb_strategies, std::uniform_real_distribution<double> &urand,
                               std::mt19937_64 &generator, EGTTools::SED::AbstractGame<std::mt19937_64> *game) {
    // number of possible group combinations
    auto nb_states = EGTTools::binomialCoeff(nb_strategies + group_size - 1, group_size);
    GroupPayoffs payoffs = GroupPayoffs::Zero(nb_strategies, nb_states);
    EGTTools::RL::Factors group_composition(nb_strategies, 0);
    std::vector<double> game_payoffs(nb_strategies, 0);
    size_t sum = 0;

    // For every possible group composition run the game and store the payoff of each strategy
    for (size_t i = 0; i < nb_states; ++i) {
        group_composition.back() = group_size - sum;
        game->play(group_composition, game_payoffs, urand, generator);

        // Fill payoff table
        for (size_t j = 0; j < nb_strategies; ++j) payoffs(j, i) = game_payoffs[j];

        // update group composition
        for (size_t k = 0; k < nb_strategies - 1; ++k) {
            if (sum < group_size) {
                if (group_composition[k] < group_size) {
                    group_composition[k] += 1;
                    sum += 1;
                    break;
                } else {
                    sum -= group_composition[k];
                    group_composition[k] = 0;
                }
            } else {
                sum -= group_composition[k];
                group_composition[k] = 0;
            }
        }
    }

    return payoffs;
}


inline double
calculate_fitness(const size_t &player_type, const size_t &nb_strategies, const size_t &group_size,
                  const size_t &pop_size,
                  std::vector<size_t> &strategies,
                  const GroupPayoffs &payoffs,
                  EGTTools::Utils::LRUCache<std::string, double> &cache) {
    double fitness, payoff;
    size_t sum;
    std::stringstream result;
    std::copy(strategies.begin(), strategies.end(), std::ostream_iterator<int>(result, ""));

    std::string key = std::to_string(player_type) + result.str();

    // First we check if fitness value is in the lookup table
    if (!cache.exists(key)) {
        fitness = 0.0;
        sum = 0;
        strategies[player_type] -= 1;
        std::vector<size_t> sample_counts(nb_strategies, 0);

        // number of possible group combinations
        auto total_nb_states = EGTTools::binomialCoeff(nb_strategies + group_size - 1, group_size);
        size_t nb_states = EGTTools::binomialCoeff(nb_strategies + group_size - 2, group_size - 1);

        // If it isn't, then we must calculate the fitness for every possible group combination
        for (size_t i = 0; i < nb_states; ++i) {
            sample_counts[player_type] += 1;
            sample_counts.back() = group_size - 1 - sum;

            // First update sample_counts with new group composition
            payoff = payoffs(player_type, calculate_state(group_size, total_nb_states, sample_counts));
            sample_counts[player_type] -= 1;

            auto prob = EGTTools::multivariateHypergeometricPDF(pop_size - 1, nb_strategies, group_size - 1,
                                                                sample_counts,
                                                                strategies);

            fitness += payoff * prob;

            // update group composition
            for (size_t k = 0; k < nb_strategies - 1; ++k) {
                if (sum < group_size - 1) {
                    if (sample_counts[k] < group_size - 1) {
                        sample_counts[k] += 1;
                        sum += 1;
                        break;
                    } else {
                        sum -= sample_counts[k];
                        sample_counts[k] = 0;
                    }
                } else {
                    sum -= sample_counts[k];
                    sample_counts[k] = 0;
                }
            }

        }

        strategies[player_type] += 1;

        // Finally we store the new fitness in the Cache. We also keep a Cache for the payoff given each group combination
        cache.insert(key, fitness);
    } else {
        fitness = cache.get(key);
    }

    return fitness;
}

inline void initialize_population(const size_t &nb_strategies, const std::vector<size_t> &strategies,
                                  std::vector<size_t> &population, std::mt19937_64 &generator) {
    size_t z = 0;
    for (unsigned int i = 0; i < nb_strategies; ++i) {
        for (size_t j = 0; j < strategies[i]; ++j) {
            population[z++] = i;
        }
    }

    // Then we shuffle it randomly
    std::shuffle(population.begin(), population.end(), generator);
}

template<typename G>
inline std::pair<size_t, size_t>
samplePlayers(std::uniform_int_distribution<size_t> &dist, G &generator) {
    auto player1 = dist(generator);
    auto player2 = dist(generator);
    while (player2 == player1) player2 = dist(generator);
    return std::make_pair(player1, player2);
}

inline size_t mutate(size_t &player, const double &mu, std::uniform_int_distribution<size_t> &irand,
                     std::uniform_real_distribution<double> &urand, std::mt19937_64 &generator) {
    auto new_player = player;
    if (urand(generator) < mu) while (new_player == player) new_player = irand(generator);
    return new_player;
}

inline std::pair<bool, size_t> is_homogeneous(size_t &pop_size, StrategyCounts &strategies) {
    for (size_t i = 0; i < strategies.size(); ++i) {
        if (strategies[i] == pop_size) return std::make_pair(true, i);
    }
    return std::make_pair(false, -1);
}

int main(int argc, char *argv[]) {

    // First we define a vector of possible behaviors
    size_t nb_strategies = EGTTools::SED::CRD::nb_strategies;
    size_t pop_size;
    size_t nb_generations;
    size_t group_size;
    size_t die, birth;
    size_t nb_rounds, min_rounds, endowment, threshold;
    double beta;
    double mu;
    double risk;
    bool timing_uncertainty;
    double p = 0.0;
    StrategyCounts strategies(EGTTools::SED::CRD::nb_strategies);
    Options options;

    options.push_back(
            makeDefaultedOption<size_t>("generations,g", &nb_generations, "set the number of generations", 1000u));
    options.push_back(makeDefaultedOption<size_t>("popSize,Z", &pop_size, "set the size of the population", 100u));
    options.push_back(
            makeRequiredOption<std::vector<size_t>>("strategies,s", &strategies, "the counts of each strategy"));
    options.push_back(makeDefaultedOption<double>("mu,u", &mu, "set mutation rate", 0.05));
    options.push_back(makeDefaultedOption<double>("beta,b", &beta, "set intensity of selection", 0.001));
    options.push_back(makeDefaultedOption<size_t>("groupSize,n", &group_size, "group size", 6));
    options.push_back(makeDefaultedOption<size_t>("nbRounds,t", &nb_rounds, "number of rounds", 10));
    options.push_back(makeDefaultedOption<size_t>("minRounds,m", &min_rounds, "minimum number of rounds", 8));
    options.push_back(makeDefaultedOption<size_t>("endowment,e", &endowment, "endowment", 40));
    options.push_back(makeDefaultedOption<size_t>("target,p", &threshold, "threshold", 120));
    options.push_back(makeDefaultedOption<double>("risk,r", &risk, "risk", 0.9));
    options.push_back(makeDefaultedOption<bool>("tu,d", &timing_uncertainty, "timing uncertainty", false));
    if (!parseCommandLine(argc, argv, options))
        return 1;

    std::vector<size_t> population(pop_size, 2);
    std::mt19937_64 generator(EGTTools::Random::SeedGenerator::getInstance().getSeed());
    std::uniform_real_distribution<double> urand(0.0, 1.0);
    std::uniform_int_distribution<size_t> irand(0, nb_strategies - 1);
    std::uniform_int_distribution<size_t> pop_rand(0, pop_size - 1);
    // Avg. number of rounds for a mutation to happen
    size_t nb_generations_for_mutation = std::floor(1 / mu);
    auto[homogeneous, idx_homo] = is_homogeneous(pop_size, strategies);

    EGTTools::SED::AbstractGame<std::mt19937_64> *game;

    if (timing_uncertainty) {
        p = 1 / static_cast<double>(nb_rounds - min_rounds + 1);
        EGTTools::TimingUncertainty<std::mt19937_64> tu(p);
        game = new EGTTools::SED::CRD::CrdGameTU(endowment, threshold, min_rounds, group_size, risk, tu);
    } else {
        game = new EGTTools::SED::CRD::CrdGame(endowment, threshold, nb_rounds, group_size, risk);
    }

    // Creates a cache for the fitness data
    EGTTools::Utils::LRUCache<std::string, double> cache(10000000);
    GroupPayoffs payoffs = calculate_payoffs(group_size, nb_strategies, urand, generator, game);

    // Save payoffs
    std::ofstream file("payoffs.txt", std::ios::out | std::ios::trunc);
    if (file.is_open()) {
        file << "Payoffs for each type of player and each possible state:" << std::endl;
        file << "rows: cooperator, defector, altruist, reciprocal, compensator" << std::endl;
        file << "cols: all possible group compositions starting at (0, 0, 0, 0, group_size)" << std::endl;
        file << payoffs << std::endl;
        file << "group_size = " << group_size << std::endl;
        file << "population_size = " << pop_size << std::endl;
        file << "beta = " << beta << std::endl;
        file << "mu = " << mu << std::endl;
        file << "nb_generations = " << nb_generations << std::endl;
        file << "timing_uncertainty = " << timing_uncertainty << std::endl;
        if (timing_uncertainty) {
            file << "min_rounds = " << min_rounds << std::endl;
            file << "mean_rounds = " << nb_rounds << std::endl;
            file << "p = " << p << std::endl;
        } else file << "nb_rounds = " << nb_rounds << std::endl;
        file << "risk = " << risk << std::endl;
        file << "endowment = " << endowment << std::endl;
        file << "threshold = " << threshold << std::endl;
        file << "initial state = (";
        for (size_t i = 0; i < nb_strategies; ++i)
            file << strategies[i] << ", ";
        file << ")" << std::endl;
    }

    // initialise population
    initialize_population(nb_strategies, strategies, population, generator);

    std::cout << "initial state: (";
    for (size_t i = 0; i < nb_strategies; ++i)
        std::cout << strategies[i] << ", ";
    std::cout << ")" << std::endl;

    // Now we start the imitation process
    for (size_t i = 0; i < nb_generations; ++i) {
        // First we pick 2 players randomly
        auto[player1, player2] = samplePlayers(pop_rand, generator);

        if (homogeneous) {
            if (mu > 0) {
                i += nb_generations_for_mutation;
                // mutate
                birth = irand(generator);
                // If population still homogeneous we wait for another mutation
                while (birth == idx_homo) {
                    i += nb_generations_for_mutation;
                    birth = irand(generator);
                }
                if (i < nb_generations) {
                    population[player1] = birth;
                    strategies[birth] += 1;
                    strategies[idx_homo] -= 1;
                    homogeneous = false;
                }
                continue;
            } else break;
        }

        // Check if player mutates
        if (urand(generator) < mu) {
            die = population[player1];
            birth = irand(generator);
            population[player1] = birth;
        } else { // If no mutation, player imitates
            // Then we let them play to calculate their payoffs
            auto fitness_p1 = calculate_fitness(population[player1], nb_strategies, group_size, pop_size, strategies,
                                                payoffs, cache);
            auto fitness_p2 = calculate_fitness(population[player2], nb_strategies, group_size, pop_size, strategies,
                                                payoffs, cache);

            // Then we apply the moran process with mutation
            if (urand(generator) < EGTTools::SED::fermi(beta, fitness_p1, fitness_p2)) {
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
        strategies[birth] += 1;
        strategies[die] -= 1;
        // Check if population is homogeneous
        if (strategies[birth] == pop_size) {
            homogeneous = true;
            idx_homo = birth;
        }
    }

    std::cout << "final state: (";
    for (size_t i = 0; i < nb_strategies; ++i)
        std::cout << strategies[i] << ", ";
    std::cout << ")" << std::endl;

    if (file.is_open()) {
        file << "final state: (";
        for (size_t i = 0; i < nb_strategies; ++i)
            file << strategies[i] << ", ";
        file << ")" << std::endl;
        file.close();
    }
}
