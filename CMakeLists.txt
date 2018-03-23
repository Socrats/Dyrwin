cmake_minimum_required(VERSION 3.7)
project(Dyrwin)

set(CMAKE_CXX_STANDARD 11)

set(SOURCE_FILES src/main.cpp src/Player.cpp include/Player.h include/Constants.h src/WrightFisherModel.cpp include/WrightFisherModel.h src/PrisonersDilemma.cpp include/PrisonersDilemma.h src/CollectiveRiskDilemma.cpp include/CollectiveRiskDilemma.h src/CRDPlayer.cpp include/CRDPlayer.h src/ParameterHandler.cpp include/ParameterHandler.h src/MoranProcess.cpp include/MoranProcess.h src/BasePlayer.cpp include/BasePlayer.h src/CRDSimulator.cpp include/CRDSimulator.h src/CRDMutate.cpp include/CRDMutate.h OutputHandler.cpp include/OutputHandler.h)
add_executable(Dyrwin ${SOURCE_FILES})