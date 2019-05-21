//
// Created by Elias Fernandez on 2019-05-18.
//

#ifndef DYRWIN_RL_DATA_HPP
#define DYRWIN_RL_DATA_HPP


#include <string>
#include <sstream>
#include <vector>
#include <Dyrwin/RL/PopContainer.hpp>

namespace EGTTools::RL {
    struct CRDSimData {
        int generation;
        double avg_fitness;
        double avg_contributions;
        double eta;

        void update(int gen, double fitness, double contributions, double group_achievement) {
            this->generation = gen;
            this->avg_fitness = fitness;
            this->avg_contributions = contributions;
            this->eta = group_achievement;
        }

        std::string getCSVHeader() {
            return "generation,avg_fitness,avg_contributions,eta\n";
        }

        std::string getCSVData() {
            std::stringstream data;
            data << generation << "," << avg_fitness << "," << avg_contributions << "," << eta << "\n";
            return data.str();
        }
    };

    struct CRDData {
        std::vector<double> eta;
        std::vector<double> avg_contribution;
        std::vector<double> avg_payoff;
        EGTTools::RL::PopContainer population;
    };
}

#endif //DYRWIN_RL_DATA_HPP
