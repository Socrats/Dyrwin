//
// Created by Elias Fernandez on 2019-04-18.
//

#ifndef DYRWIN_SED_MORANPROCESS_HPP
#define DYRWIN_SED_MORANPROCESS_HPP

#include <random>
#include <algorithm>
#include <cmath>
#include <Dyrwin/SeedGenerator.h>
#include <Dyrwin/Types.h>


namespace EGTTools {
    class MoranProcessNPlayer : public MoranProcess {
    public:
        MoranProcessNPlayer(size_t generations, size_t nb_strategies, size_t group_size, double beta,
                     Eigen::Ref<const Vector> strategy_freq, Eigen::Ref<const Matrix2D> payoff_matrix);

        MoranProcessNPlayer(size_t generations, size_t nb_strategies, size_t group_size, size_t nb_groups, double beta,
                     Eigen::Ref<const Vector> strategy_freq, Eigen::Ref<const Matrix2D> payoff_matrix);

        MoranProcessNPlayer(size_t generations, size_t nb_strategies, size_t group_size, size_t nb_groups, double beta,
                     double mu, Eigen::Ref<const Vector> strategy_freq,
                     Eigen::Ref<const Matrix2D> payoff_matrix);

        MoranProcessNPlayer(size_t generations, size_t nb_strategies, size_t group_size, size_t nb_groups, double beta,
                     double mu, double split_prob, Eigen::Ref<const Vector> strategy_freq,
                     Eigen::Ref<const Matrix2D> payoff_matrix);



    private:
        size_t group_size;

        inline void _moran_step(unsigned int &p1, unsigned int &p2,
                                Vector &freq1, Vector &freq2, double &fitness1, double &fitness2,
                                double &beta,
                                Matrix2D &group_coop, VectorXui &final_strategies,
                                std::vector<unsigned int> &population,
                                std::uniform_int_distribution<unsigned int> &dist,
                                std::uniform_real_distribution<double> &_uniform_real_dist);
    };
}

#endif //DYRWIN_SED_MORANPROCESS_HPP
