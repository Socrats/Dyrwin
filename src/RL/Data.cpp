//
// Created by Elias Fernandez on 2019-08-12.
//

#include <Dyrwin/RL/Data.hpp>
#include <utility>

EGTTools::RL::DataTypes::CRDData::CRDData(size_t length, EGTTools::RL::PopContainer &container) : population(
    std::move(container)) {
  eta = Vector::Zero(length);
  avg_contribution = Vector::Zero(length);
}
EGTTools::RL::DataTypes::CRDVerboseData::CRDVerboseData(size_t length,
                                                        EGTTools::RL::PopContainer &container) : population(std::move(
    container)) {
  eta = Vector::Zero(length);
  avg_contribution = Vector::Zero(length);
  polarization = Matrix2D::Zero(4, length);
  contribution_time = Matrix2D::Zero(2, length);
  behavioral_frequency = Matrix2D::Zero(3, length);
}
EGTTools::RL::DataTypes::CRDVerboseStationaryData::CRDVerboseStationaryData(size_t nb_behaviors,
                                                                            double eta,
                                                                            double avg_contribution,
                                                                            EGTTools::RL::PopContainer &container)
    : eta(eta),
      avg_contribution(avg_contribution),
      population(std::move(container)) {
  polarization = Vector::Zero(4);
  contribution_time = Vector::Zero(2);
  behavioral_frequency = Vector::Zero(nb_behaviors);
}
EGTTools::RL::DataTypes::DataTableCRD::DataTableCRD(size_t nb_rows,
                                                    size_t nb_columns,
                                                    std::vector<std::string> &headers,
                                                    std::vector<std::string> &column_types,
                                                    std::vector<EGTTools::RL::PopContainer> &container)
    : header(std::move(headers)),
      column_types(std::move(column_types)) {
  // We first instantiate the data container
  data = Matrix2D::Zero(nb_rows, nb_columns);
}
