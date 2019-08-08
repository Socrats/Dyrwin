//
// Created by Elias Fernandez on 2019-08-08.
//

#ifndef DYRWIN_DATASTRUCT_HPP
#define DYRWIN_DATASTRUCT_HPP

#include <sstream>

struct CRDSimData {
    int generation;
    double avg_fitness;
    double avg_contributions;
    double avg_threshold;

    /**
     * @brief Updates the data structure
     * @param generation : generation at which the data is obtained
     * @param avg_fitness : average fitness at the generation
     * @param avg_contributions : average contribution at the generation
     * @param avg_threshold : average threshold
     */
    void update(int generation, double avg_fitness, double avg_contributions, double avg_threshold);

    /**
     * @brief generates a string with the names of the fields in the struct in scv format
     * @return the header for a csv file
     */
    std::string getCSVHeader();

    /**
     * @brief generates a string with the contents of the struct in csv format
     * @return the data in the struct in csv format
     */
    std::string getCSVData();
};

struct GameData {
    bool met_threshold;
    double public_account;
    double avg_donations;

    /**
     * @brief This structure stores data related to each individual game played
     */
    GameData() : met_threshold(false), public_account(0), avg_donations(0) {};

    /**
     * @brief This structure stores data related to each individual game played
     * @param met_threshold
     * @param public_account
     */
    GameData(bool met_threshold, double public_account) : met_threshold(met_threshold), public_account(public_account),
                                                          avg_donations(0) {};

    /**
     * @brief This structure stores data related to each individual game played
     * @param met_threshold
     * @param public_account
     * @param avg_donations
     */
    GameData(bool met_threshold, double public_account, double avg_donations) : met_threshold(met_threshold),
                                                                                public_account(public_account),
                                                                                avg_donations(avg_donations) {};
};

#endif //DYRWIN_DATASTRUCT_HPP
