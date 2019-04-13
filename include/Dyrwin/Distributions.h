//
// Created by Elias Fernandez on 2019-03-19.
//

#ifndef DYRWIN_DISTRIBUTIONS_H
#define DYRWIN_DISTRIBUTIONS_H

#include <random>
#include <algorithm>
#include <Eigen/Dense>

namespace egt_tools {
    static std::uniform_real_distribution<double> probabilityDistribution(0.0, 1.0);

    template<typename G>
    size_t choice(const size_t d, const Eigen::MatrixXd::RowXpr& in, G& generator) {
        double p = probabilityDistribution(generator);

        for ( Eigen::MatrixXd::RowXpr::InnerIterator i(in, 0); ; ++i ) {
            if ( i.value() > p ) return static_cast<size_t>(i.col());
            p -= i.value();
        }
        return d-1;
    }
}

#endif //DYRWIN_DISTRIBUTIONS_H
