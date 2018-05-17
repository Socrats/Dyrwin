cmake_minimum_required(VERSION 3.7)
project(Dyrwin)

set(CMAKE_CXX_STANDARD 11)

set(SOURCE_FILES src/main.cpp src/Player.cpp include/Dyrwin/Player.h include/Dyrwin/Constants.h src/WrightFisherModel.cpp include/Dyrwin/WrightFisherModel.h src/PrisonersDilemma.cpp include/Dyrwin/PrisonersDilemma.h src/CollectiveRiskDilemma.cpp include/Dyrwin/CollectiveRiskDilemma.h src/CRDPlayer.cpp include/Dyrwin/CRDPlayer.h src/ParameterHandler.cpp include/Dyrwin/ParameterHandler.h src/MoranProcess.cpp include/Dyrwin/MoranProcess.h src/BasePlayer.cpp include/Dyrwin/BasePlayer.h src/CRDSimulator.cpp include/Dyrwin/CRDSimulator.h src/CRDMutate.cpp include/Dyrwin/CRDMutate.h src/OutputHandler.cpp include/Dyrwin/OutputHandler.h include/Dyrwin/Utils.h src/SeedGenerator.cpp include/Dyrwin/SeedGenerator.h)
add_executable(Dyrwin ${SOURCE_FILES})