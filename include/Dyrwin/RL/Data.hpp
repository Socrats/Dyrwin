//
// Created by Elias Fernandez on 2019-05-18.
//

#ifndef DYRWIN_RL_DATA_HPP
#define DYRWIN_RL_DATA_HPP

#include <string>
#include <vector>
#include <sstream>
#include <Dyrwin/Types.h>
#include <Dyrwin/RL/PopContainer.hpp>

namespace EGTTools::RL::DataTypes {
//    struct CRDSimData {
//        int generation;
//        double avg_fitness;
//        double avg_contributions;
//        double eta;
//
//        void update(int gen, double fitness, double contributions, double group_achievement) {
//            this->generation = gen;
//            this->avg_fitness = fitness;
//            this->avg_contributions = contributions;
//            this->eta = group_achievement;
//        }
//
//        std::string getCSVHeader() {
//            return "generation,avg_fitness,avg_contributions,eta\n";
//        }
//
//        std::string getCSVData() {
//            std::stringstream data;
//            data << generation << "," << avg_fitness << "," << avg_contributions << "," << eta << "\n";
//            return data.str();
//        }
//    };

struct CRDData {
  Vector eta; // group achievement
  Vector avg_contribution;
  EGTTools::RL::PopContainer population;

  CRDData(size_t length, EGTTools::RL::PopContainer &container);
};

struct CRDDataIslands {
  Vector eta;
  Vector avg_contribution;
  std::vector<EGTTools::RL::PopContainer> groups;

  CRDDataIslands() = default;

  CRDDataIslands(Vector &group_achievement, Vector &avg_donations,
                 std::vector<EGTTools::RL::PopContainer> &container) : eta(std::move(group_achievement)),
                                                                       avg_contribution(
                                                                           std::move(avg_donations)),
                                                                       groups(std::move(container)) {};
};

/**
 * This structure stores verbose data over all generations of a CRD simulation.
 */
struct CRDVerboseData {
  Vector eta; // group achievement
  Vector avg_contribution;
  // percentage of players donating C = 0, C < F, C = F, C > F in each generation
  Matrix2D polarization;
  // stores the percentage of contributions in the first and second half of the
  // game
  Matrix2D contribution_time;
  // 3 most frequent behaviors at each generation
  Matrix2D behavioral_frequency;
  // Final population
  EGTTools::RL::PopContainer population;

  CRDVerboseData(size_t length, EGTTools::RL::PopContainer &container);
};

/**
 * This data structure stores the data only after learning (the last X generations).
 */
struct CRDVerboseStationaryData {
  double eta; // average group achievement
  double avg_contribution; // average contribution
  Vector polarization; // final distribution of contributions over players
  Vector contribution_time; // final distribution of contributions over the rounds
  Vector behavioral_frequency; // frequency of each learned behavior
  // Final population
  EGTTools::RL::PopContainer population;

  CRDVerboseStationaryData(size_t nb_behaviors,
                           double eta,
                           double avg_contribution,
                           EGTTools::RL::PopContainer &container);
};

/**
 * This data structure stores the data only after learning (the last X generations).
 */
struct DataTableCRD {
  Matrix2D data;
  std::vector<std::string> header;
  std::vector<std::string> column_types;
  // Final population
  std::vector<EGTTools::RL::PopContainer> populations;

  DataTableCRD(size_t nb_rows,
               size_t nb_columns,
               std::vector<std::string> &headers,
               std::vector<std::string> &column_types,
               std::vector<EGTTools::RL::PopContainer> &container);
};
}

#endif //DYRWIN_RL_DATA_HPP
