//
// Created by Elias Fernandez on 2019-08-08.
//

#include <Dyrwin/AbouCRD/DataStruct.hpp>

void CRDSimData::update(int generation, double avg_fitness, double avg_contributions, double avg_threshold) {
    this->generation = generation;
    this->avg_fitness = avg_fitness;
    this->avg_contributions = avg_contributions;
    this->avg_threshold = avg_threshold;
}

std::string CRDSimData::getCSVHeader() {
    return "generation,avg_fitness,avg_contributions,avg_threshold\n";
}

std::string CRDSimData::getCSVData() {
    std::stringstream data;
    data << generation << "," << avg_fitness << "," << avg_contributions << "," << avg_threshold << "\n";
    return data.str();
}