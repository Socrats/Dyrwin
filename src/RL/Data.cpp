//
// Created by Elias Fernandez on 2019-08-12.
//

#include <Dyrwin/RL/Data.hpp>

EGTTools::RL::DataTypes::CRDData::CRDData(size_t length, EGTTools::RL::PopContainer &container) : population(
        container) {
    eta = Vector::Zero(length);
    avg_contribution = Vector::Zero(length);
}