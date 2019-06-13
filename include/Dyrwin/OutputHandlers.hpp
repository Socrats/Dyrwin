//
// Created by Elias Fernandez on 2019-06-13.
//

#ifndef DYRWIN_OUTPUTHANDLERS_HPP
#define DYRWIN_OUTPUTHANDLERS_HPP

#include <string>
#include <fstream>

template<typename T>
inline bool convert2CSV(std::string &filename, std::string &ofilename) {

    T data;

    std::ofstream outFile;
    std::ifstream inFile;
    inFile.open(filename, std::ios::in | std::ios::binary);
    outFile.open(ofilename, std::ios::out);

    outFile << data.getCSVHeader();

    while (!inFile.eof()) {
        inFile.read((char *) &data, sizeof(T));
        outFile << data.getCSVData();
    }

    inFile.close();
    outFile.close();

    return true;
}

#endif //DYRWIN_OUTPUTHANDLERS_HPP
