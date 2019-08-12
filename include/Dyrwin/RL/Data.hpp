//
// Created by Elias Fernandez on 2019-05-18.
//

#ifndef DYRWIN_RL_DATA_HPP
#define DYRWIN_RL_DATA_HPP


#include <string>
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
        Vector eta;
        Vector avg_contribution;
        EGTTools::RL::PopContainer population;

        CRDData(size_t length, EGTTools::RL::PopContainer &conainer);
        Vector & get_eta() { return eta;}
        Vector & get_avg_contribution() { return avg_contribution; }
        EGTTools::RL::PopContainer & get_population() { return population; }
    };
}

#endif //DYRWIN_RL_DATA_HPP
