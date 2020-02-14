//
// Created by Elias Fernandez on 2019-06-13.
//

#ifndef DYRWIN_OUTPUTHANDLERS_HPP
#define DYRWIN_OUTPUTHANDLERS_HPP

#include <string>
#include <fstream>
#include <Eigen/Dense>
#include <Dyrwin/Types.h>

namespace EGTTools::OutputHandler {
void writeToCSVFile(const std::string &name, const std::string &header, const EGTTools::Matrix2D &matrix) {
  Eigen::IOFormat CSVFormat(Eigen::StreamPrecision, Eigen::DontAlignCols, ", ", "\n");
  std::ofstream file(name.c_str());
  file << header << std::endl;
  file << matrix.format(CSVFormat);
  file.close();
}

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
}

#endif //DYRWIN_OUTPUTHANDLERS_HPP
