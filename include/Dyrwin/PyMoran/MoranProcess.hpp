//
// Created by Elias Fernandez on 2019-04-18.
//

#ifndef DYRWIN_MORANPROCESS_HPP
#define DYRWIN_MORANPROCESS_HPP

#include <random>
#include <algorithm>
#include <cmath>
#include <Dyrwin/SeedGenerator.h>
#include <Dyrwin/Types.h>


namespace EGTTools {
    class MoranProcess {
    public:
        MoranProcess(size_t generations, size_t nb_strategies, size_t group_size, double beta,
                     Eigen::Ref<const Vector> strategy_freq, Eigen::Ref<const Matrix2D> payoff_matrix);

        MoranProcess(size_t generations, size_t nb_strategies, size_t group_size, size_t nb_groups, double beta,
                     Eigen::Ref<const Vector> strategy_freq, Eigen::Ref<const Matrix2D> payoff_matrix);

        MoranProcess(size_t generations, size_t nb_strategies, size_t group_size, size_t nb_groups, double beta,
                     double mu, Eigen::Ref<const Vector> strategy_freq,
                     Eigen::Ref<const Matrix2D> payoff_matrix);

        MoranProcess(size_t generations, size_t nb_strategies, size_t group_size, size_t nb_groups, double beta,
                     double mu, double split_prob, Eigen::Ref<const Vector> strategy_freq,
                     Eigen::Ref<const Matrix2D> payoff_matrix);


        double fermifunc(double beta, double a, double b);

        Vector evolve(double beta);

        Vector evolve(size_t runs, double beta);

//        Matrix2D evolve(std::vector<double> betas);

//        Matrix2D evolve(size_t runs, std::vector<double> betas);

        inline void initialize_population(std::vector<unsigned int> &population);

        inline void initialize_group_coop(Matrix2D &group_coop, std::vector<unsigned int> &population);


        // Getters
        size_t generations() { return _generations; }

        size_t nb_strategies() { return _nb_strategies; }

        unsigned int pop_size() { return _pop_size; }

        unsigned int group_size() { return _group_size; }

        unsigned int nb_groups() { return _nb_groups; }

        double mu() { return _mu; }

        double beta() { return _beta; }

        double split_prob() { return _split_prob; }

        Vector &init_strategy_freq() { return _strategy_freq; }

        Vector &strategy_freq() { return _final_strategy_freq; }

        VectorXui &init_strategy_count() { return _strategies; }

        Matrix2D &payoff_matrix() { return _payoff_matrix; }

        // Setters
        void set_generations(size_t generations) { _generations = generations; }

        void set_pop_size(unsigned int pop_size) { _pop_size = pop_size; }

        void set_group_size(unsigned group_size) {
            _group_size = group_size;
            _pop_size = _nb_groups * _group_size;
        }

        void set_nb_groups(unsigned nb_groups) {
            _nb_groups = nb_groups;
            _pop_size = _nb_groups * _group_size;
        }

        void set_beta(double beta) { _beta = beta; }

        void set_mu(double mu) { _mu = mu; }

        void set_split_prob(double split_prob) { _split_prob = split_prob; }

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
            _payoff_matrix.array() = payoff_matrix;
//            Eigen::Map<Matrix2D>(_payoff_matrix.data(), _payoff_matrix.rows(), _payoff_matrix.cols()) = payoff_matrix;
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

        friend std::ostream &operator<<(std::ostream &o, MoranProcess &r) { return o << r.toString(); }

    private:
        size_t _generations, _nb_strategies, _group_size, _nb_groups, _pop_size;
        double _beta, _mu, _split_prob;

        Vector _strategy_freq, _final_strategy_freq; // frequency of each strategy in the population
        VectorXui _strategies; //nb of players of each strategy
        Matrix2D _payoff_matrix; // stores the payoff matrix of the game

        // Random generators
        std::mt19937_64 _mt{EGTTools::Random::SeedGenerator::getInstance().getSeed()};

        inline void _moran_step(unsigned int &p1, unsigned int &p2,
                                Vector &freq1, Vector &freq2, double &fitness1, double &fitness2,
                                double &beta,
                                Matrix2D &group_coop, VectorXui &final_strategies,
                                std::vector<unsigned int> &population,
                                std::uniform_int_distribution<unsigned int> &dist,
                                std::uniform_real_distribution<double> &_uniform_real_dist);
    };
}

#endif //DYRWIN_MORANPROCESS_HPP
