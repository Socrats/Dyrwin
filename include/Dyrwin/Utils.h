//
// Created by Elias Fernandez on 09/04/2018.
//

#ifndef DYRWIN_UTILS_H
#define DYRWIN_UTILS_H


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

#endif //DYRWIN_UTILS_H

