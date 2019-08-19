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
}

#endif //DYRWIN_RL_DATA_HPP
