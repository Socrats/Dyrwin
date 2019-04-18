cmake_minimum_required(VERSION 3.9)
project(Dyrwin LANGUAGES CXX)

# Help variables to select what to build
#
# MAKE_ALL:      Builds all project executable and python bindings
# MAKE_DYRWIN:   Builds the Dyrwin library executables
# MAKE_PMORAN:   Build the moran process executable
# MAKE_PYTHON:   Builds Python bindings

##############################
##  CMake helper functions  ##
##############################

function(append value)
    foreach (variable ${ARGN})
        set(${variable} "${${variable}} ${value}" PARENT_SCOPE)
    endforeach (variable)
endfunction()

##############################
## Compiler/Linker Settings ##
##############################

# Set default cmake build type to release
if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING
            "Choose the type of build, options are:
          Debug Release RelWithDebInfo MinSizeRel."
            FORCE)
endif ()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if (NOT WIN32)
    add_definitions(
            -Wall
            -Wextra
    )
endif ()

# Check for Link Time Optimizations with this compiler
include(CheckIPOSupported)
check_ipo_supported(RESULT LTO_SUPPORTED OUTPUT LTO_ERROR)

if (LTO_SUPPORTED)
    message(STATUS "IPO / LTO enabled")
else ()
    message(STATUS "IPO / LTO not supported: <${LTO_ERROR}>")
endif ()

##############################
##     Project Settings     ##
##############################

# Default is to build everything
if (NOT MAKE_ALL AND NOT MAKE_DYRWIN AND NOT MAKE_PMORAN AND NOT MAKE_PYTHON AND NOT MAKE_CRD AND NOT MAKE_CRD_UNCERTAINTY)
    set(MAKE_ALL 1)
endif ()

if (MAKE_ALL)
    set(MAKE_DYRWIN 1)
    set(MAKE_PMORAN 1)
    set(MAKE_CRD 1)
    set(MAKE_CRD_UNCERTAINTY 1)
    set(MAKE_PYTHON 1)
endif ()

##############################
##       Dependencies       ##
##############################
if (MAKE_ALL OR MAKE_DYRWIN OR MAKE_PMORAN OR MAKE_CRD OR MAKE_CRD_UNCERTAINTY)
    set(BOOST_VERSION_REQUIRED 1.62)
endif ()
set(EIGEN_VERSION_REQUIRED 3.3)
set(PYTHON_VERSION 3)

# Optional to force Boost to use static libraries. Can be useful on Windows.
#
# set(Boost_USE_STATIC_LIBS ON)
# add_definitions(-DBOOST_PYTHON_STATIC_LIB)

set(PYBIND11_CPP_STANDARD -std=c++1z) # Experimental C++17 support

find_package(Boost ${BOOST_VERSION_REQUIRED} REQUIRED program_options system)
include_directories(SYSTEM ${Boost_INCLUDE_DIRS})

find_package(Eigen3 ${EIGEN_VERSION_REQUIRED} REQUIRED)
include_directories(SYSTEM ${EIGEN3_INCLUDE_DIR})

include_directories(SYSTEM ${PYTHON_INCLUDE_DIRS})

# Support for OpenMP parallelization
find_package(OpenMP)
if (OPENMP_FOUND)
    include_directories(SYSTEM ${OPENMP_INCLUDE_DIR})
    message(STATUS "OpenMP enabled")
endif ()
#if (OPENMP_FOUND)
#    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
#    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
#    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
#endif()

# Include the project directory
include_directories($(PROJECT_SOURCE_DIR)/include)
# add_subdirectory(${PROJECT_SOURCE_DIR}/src)

##############################
## Building executables     ##
##############################

set(SOURCE_FILES src/main.cpp
        src/CollectiveRiskDilemma.cpp
        include/Dyrwin/crd/CollectiveRiskDilemma.h
        src/CRDPlayer.cpp src/CRDSimulator.cpp
        include/Dyrwin/crd/CRDSimulator.h
        include/Dyrwin/Utils.h src/SeedGenerator.cpp
        include/Dyrwin/SeedGenerator.h
        include/Dyrwin/CommandLineParsing.h
        include/Dyrwin/ipd/IteratedPrisonersDilemma.cpp)

#if (MAKE_DYRWIN)
#    add_executable(Dyrwin ${SOURCE_FILES})
#    target_link_libraries(Dyrwin ${Boost_LIBRARIES} boost_program_options)
#    set_target_properties(Dyrwin PROPERTIES CMAKE_CXX_FLAGS lboost_program_options)
#endif ()

set(PYTHON_MORAN_SOURCE
        include/Dyrwin/Types.h
        include/Dyrwin/SeedGenerator.h src/SeedGenerator.cpp
        include/Dyrwin/PyMoran/PDImitation.h
        src/PyMoran/PDImitation.cpp
        include/Dyrwin/PyMoran/StochDynamics.h
        src/PyMoran/StochDynamics.cpp
        include/Dyrwin/PyMoran/TraulsenMoran.h
        src/PyMoran/TraulsenMoran.cpp
        include/Dyrwin/PyMoran/MoranProcess.hpp
        src/PyMoran/MoranProcess.cpp)

if (MAKE_PMORAN)
    add_executable(pmoran src/PyMoran/pmain.cpp
            include/Dyrwin/CommandLineParsing.h ${PYTHON_MORAN_SOURCE})
    TARGET_LINK_LIBRARIES(pmoran ${Boost_LIBRARIES} boost_program_options)
    set_target_properties(pmoran PROPERTIES CMAKE_CXX_FLAGS lboost_program_options)
    target_link_libraries(pmoran Eigen3::Eigen)
    if (OPENMP_FOUND)
        target_link_libraries(pmoran OpenMP::OpenMP_CXX)
    endif ()
endif ()

set(CRD_RL_SOURCE
        include/Dyrwin/SeedGenerator.h src/SeedGenerator.cpp
        include/Dyrwin/rl/CRDGame.h
        include/Dyrwin/rl/Agent.h
        src/rl/Agent.cpp
        include/Dyrwin/rl/RLUtils.h
        include/Dyrwin/Distributions.h
        src/rl/BatchQLearningAgent.cpp
        include/Dyrwin/rl/BatchQLearningAgent.h
        include/Dyrwin/rl/RothErevAgent.h
        include/Dyrwin/rl/QLearningAgent.h
        src/rl/QLearningAgent.cpp
        include/Dyrwin/Types.h include/Dyrwin/rl/histericQLearningAgent.hpp
        include/Dyrwin/rl/HistericQLearningAgent.hpp
        include/Dyrwin/rl/TimingUncertainty.hpp)

if (MAKE_CRD)
    add_executable(crd src/rl/main.cpp
            include/Dyrwin/CommandLineParsing.h ${CRD_RL_SOURCE})
    TARGET_LINK_LIBRARIES(crd ${Boost_LIBRARIES} boost_program_options)
    set_target_properties(crd PROPERTIES CMAKE_CXX_FLAGS lboost_program_options)
    target_link_libraries(crd Eigen3::Eigen)
endif ()

if (MAKE_CRD_UNCERTAINTY)
    add_executable(crdTU src/rl/main_uncertainty.cpp
                            include/Dyrwin/CommandLineParsing.h ${CRD_RL_SOURCE})
    TARGET_LINK_LIBRARIES(crdTU ${Boost_LIBRARIES} boost_program_options)
    set_target_properties(crdTU PROPERTIES CMAKE_CXX_FLAGS lboost_program_options)
    target_link_libraries(crdTU Eigen3::Eigen)
endif ()

# Python bindings
if (MAKE_PYTHON)
    add_subdirectory(pybind11)
    pybind11_add_module(EGTtools SHARED ${PYTHON_MORAN_SOURCE} src/PyMoran/EGTtoolsWrapper.cpp)
    if (OPENMP_FOUND)
        target_link_libraries(EGTtools PRIVATE OpenMP::OpenMP_CXX)
    endif ()
endif ()
# target_link_libraries(EGTtools ${${BOOST_PYTHON_LIBRARY_NAME}} ${PYTHON_LIBRARY})
# set_target_properties(EGTtools PROPERTIES PREFIX "")
# set_target_properties(EGTtools PROPERTIES SUFFIX ".so")
