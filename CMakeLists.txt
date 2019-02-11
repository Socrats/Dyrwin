cmake_minimum_required(VERSION 3.7)
project(Dyrwin)

set(CMAKE_CXX_STANDARD 11)

FIND_PACKAGE(Boost COMPONENTS program_options REQUIRED)
INCLUDE_DIRECTORIES(${Boost_INCLUDE_DIRS})

set (CMAKE_EXE_LINKER_FLAGS -lboost_program_options )

set(SOURCE_FILES src/main.cpp src/CollectiveRiskDilemma.cpp include/Dyrwin/crd/CollectiveRiskDilemma.h src/CRDPlayer.cpp src/CRDSimulator.cpp include/Dyrwin/crd/CRDSimulator.h include/Dyrwin/Utils.h src/SeedGenerator.cpp include/Dyrwin/SeedGenerator.h include/Dyrwin/CommandLineParsing.h include/Dyrwin/ipd/IteratedPrisonersDilemma.cpp)
add_executable(Dyrwin ${SOURCE_FILES})
TARGET_LINK_LIBRARIES(Dyrwin ${Boost_LIBRARIES})

set(PYTHON_MORAN_SOURCE src/PyMoran/pmain.cpp include/Dyrwin/SeedGenerator.h src/SeedGenerator.cpp)
add_executable(pmoran ${PYTHON_MORAN_SOURCE})
TARGET_LINK_LIBRARIES(pmoran ${Boost_LIBRARIES})
