//
// Created by Elias Fernandez on 2019-04-18.
//

#include <Dyrwin/PyMoran/MoranProcess.hpp>

using namespace EGTTools;

MoranProcess::MoranProcess(size_t generations, size_t nb_strategies, size_t group_size, double beta,
                           Eigen::Ref<const Vector> strategy_freq,
                           Eigen::Ref<const Matrix2D> payoff_matrix) : MoranProcess(generations,
                                                                                    nb_strategies, group_size,
                                                                                    1, beta, 0, 0,
                                                                                    strategy_freq,
                                                                                    payoff_matrix) {

}

MoranProcess::MoranProcess(size_t generations, size_t nb_strategies, size_t group_size, size_t nb_groups, double beta,
                           Eigen::Ref<const Vector> strategy_freq,
                           Eigen::Ref<const Matrix2D> payoff_matrix) : MoranProcess(generations,
                                                                                    nb_strategies, group_size,
                                                                                    nb_groups, beta, 0, 0,
                                                                                    strategy_freq,
                                                                                    payoff_matrix) {

}


MoranProcess::MoranProcess(size_t generations, size_t nb_strategies, size_t group_size, size_t nb_groups, double beta,
                           double mu, Eigen::Ref<const Vector> strategy_freq,
                           Eigen::Ref<const Matrix2D> payoff_matrix) : MoranProcess(generations,
                                                                                    nb_strategies, group_size,
                                                                                    nb_groups, beta, mu, 0,
                                                                                    strategy_freq,
                                                                                    payoff_matrix) {

}


MoranProcess::MoranProcess(size_t generations, size_t nb_strategies, size_t group_size,
                           size_t nb_groups, double beta, double mu, double split_prob,
                           Eigen::Ref<const Vector> strategy_freq,
                           Eigen::Ref<const Matrix2D> payoff_matrix) : _generations(generations),
                                                                       _nb_strategies(nb_strategies),
                                                                       _group_size(group_size),
                                                                       _nb_groups(nb_groups),
                                                                       _beta(beta),
                                                                       _mu(mu),
                                                                       _split_prob(split_prob),
                                                                       _strategy_freq(strategy_freq),
                                                                       _payoff_matrix(payoff_matrix) {
    if (static_cast<size_t>(_payoff_matrix.rows() * _payoff_matrix.cols()) != (_nb_strategies * _nb_strategies))
        throw std::invalid_argument(
                "Payoff matrix has wrong dimensions it must have shapre (nb_strategies, nb_strategies)");
    _pop_size = _nb_groups * _group_size;
    _strategies = VectorXui::Zero(_nb_strategies);
    // Calculate the number of individuals belonging to each strategy from the initial frequencies
    size_t tmp = 0;
    for (size_t i = 0; i < (_nb_strategies - 1); ++i) {
        _strategies(i) = (size_t) floor(_strategy_freq(i) * _pop_size);
        tmp += _strategies(i);
    }
    _strategies(_nb_strategies - 1) = (size_t) _pop_size - tmp;
    _final_strategy_freq = Vector::Zero(_nb_strategies);
}

void EGTTools::MoranProcess::initialize_population(std::vector<unsigned int> &population) {
    size_t z = 0;
    for (unsigned int i = 0; i < _nb_strategies; ++i) {
        for (size_t j = 0; j < _strategies(i); ++j) {
            population[z++] = i;
        }
    }

    // Then we shuffle it randomly
    std::shuffle(population.begin(), population.end(), _mt);
}

void EGTTools::MoranProcess::initialize_group_coop(Matrix2D &group_coop,
                                                   std::vector<unsigned int> &population) {
    unsigned int i;
    // Calculate the number of cooperators in each group
    for (i = 0; i < _pop_size; i++) {
        ++group_coop(floor(i / _group_size), population[i]);
    }
}

double EGTTools::MoranProcess::fermifunc(double beta, double a, double b) {
    return 1 / (1 + exp(beta * (a - b)));
}

/**
 * @brief Runs the moran process once
 *
 * This function will run the moran process for @param generations steps.
 * If the mutation rate is set to zero, the simulation will stop once one
 * of the strategies invades the whole population, even if the number of
 * steps is below the number of generations
 *
 * @param beta selection strength
 * @return a Vector containing the frequencies of each strategy in the population
 */
Vector EGTTools::MoranProcess::evolve(double beta) {
    size_t i, j, buff;
    unsigned int p1, p2;
    double fitness1, fitness2;
    std::uniform_int_distribution<unsigned int> dist(0, _pop_size - 1);
    std::uniform_real_distribution<double> _uniform_real_dist(0.0, 1.0);
    std::vector<unsigned int> population = std::vector<unsigned int>(_pop_size, 0);
    Matrix2D group_coop = Matrix2D::Zero(_nb_groups, _nb_strategies);
    Vector freq1 = Vector::Zero(_nb_strategies), freq2 = Vector::Zero(_nb_strategies);
    VectorXui final_strategies = VectorXui::Zero(_nb_strategies);

    initialize_population(population);
    initialize_group_coop(group_coop, population);

    // Reinitialize the proportions of each strategy
    for (i = 0; i < _nb_strategies; ++i) final_strategies(i) = _strategies(i);

    // Now run a Moran Process loop
    if (_mu == 0.) { // Without mutation
        for (i = 0; i < _generations; ++i) {
            _moran_step(p1, p2, freq1, freq2, fitness1, fitness2, beta, group_coop, final_strategies, population, dist,
                        _uniform_real_dist);
            // Check if one of the strategies has dominated the population
            buff = 0;
            for (j = 0; j < _nb_strategies; ++j) {
                if (final_strategies(j) == _pop_size) {
                    buff = 1;
                    break;
                }
            }
            if (buff) break;
        }
    }
    return final_strategies.cast<double>() / _pop_size;
}

Vector EGTTools::MoranProcess::evolve(size_t runs, double beta) {
    Vector freq_buff = Vector::Zero(_nb_strategies);
    // Run loop
#pragma omp parallel for shared(freq_buff)
    for (unsigned int j = 0; j < runs; j++) {
        freq_buff += evolve(beta);
    }
    _final_strategy_freq.array() = freq_buff / runs;
    return _final_strategy_freq;
}

/**
 * @brief Multi-level selection with exponential mapping of fitness described in Traulsen & Nowak, 2006.
 *
 * In each time step, a random individual from the entire population is chosen for reproduction
 * proportional to fitness. The offspring is added to the same group, If the new groups size is
 * less than or equal to n, nothing else happens. If the group size exceeds n, then with probability
 * q, the group splits into two. In this case, a random group is eliminated in order to maintain a
 * constant number of groups. With probability 1-q, however, teh groups does not divide, but instead
 * a random individual from that group is eliminated.
 *
 * @param p1 : index of player 1 in the population
 * @param p2 : index of player 2 in the population
 * @param gradient : gradient of selection (whether the number of cooperator, increases, decreases or is maintained)
 * @param freq1 : frequency of cooperators for player 1
 * @param freq2 : frequency of cooperators for player 2
 * @param fitness1 : fitness of player 1
 * @param fitness2 : fitness of player 2
 * @param beta : intensity of selection
 * @param group_coop : reference to vector where the group compositions are stored
 * @param population : reference to the vector where the population is stored
 * @param dist : random integer generator
 * @param _uniform_real_dist random real number generator
 */
void EGTTools::MoranProcess::_moran_step(unsigned int &p1, unsigned int &p2,
                                         Vector &freq1, Vector &freq2,
                                         double &fitness1, double &fitness2,
                                         double &beta,
                                         Matrix2D &group_coop, VectorXui &final_strategies,
                                         std::vector<unsigned int> &population,
                                         std::uniform_int_distribution<unsigned int> &dist,
                                         std::uniform_real_distribution<double> &_uniform_real_dist) {
    unsigned int g1, g2;
    // Randomly select 2 individuals from the population
    p1 = dist(_mt);
    p2 = dist(_mt);
    while (population[p2] == population[p1]) p2 = dist(_mt);
    // Group index
    g1 = p1 / _group_size;
    g2 = p2 / _group_size;

    // Calculate frequencies
    freq1.array() = group_coop.row(g1);
    --freq1(population[p1]);
    freq2.array() = group_coop.row(g2);
    --freq2(population[p2]);

    // Calculate fitness
    fitness1 = (freq1 * _payoff_matrix.row(population[p1])).array().sum() / static_cast<double>(_group_size - 1);
    fitness2 = (freq2 * _payoff_matrix.row(population[p2])).array().sum() / static_cast<double>(_group_size - 1);

    // Select according to fermi function
    if (_uniform_real_dist(_mt) < fermifunc(beta, fitness1, fitness2)) {
        --final_strategies(population[p1]);
        ++final_strategies(population[p2]);
        --group_coop(g1, population[p1]);
        ++group_coop(g1, population[p2]);
        population[p1] = population[p2];
    } else {
        ++final_strategies(population[p1]);
        --final_strategies(population[p2]);
        ++group_coop(g2, population[p1]);
        --group_coop(g2, population[p2]);
        population[p2] = population[p1];
    }
}