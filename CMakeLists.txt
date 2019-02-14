cmake_minimum_required(VERSION 3.9)
project(Dyrwin LANGUAGES CXX)

set(MAKE_PYTHON 1)

##############################
## Compiler/Linker Settings ##
##############################

# Set default cmake build type to release
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING
            "Choose the type of build, options are:
          Debug Release RelWithDebInfo MinSizeRel."
            FORCE)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(BOOST_VERSION_REQUIRED 1.67)
set(PYTHON_VERSION 3)

set(PYBIND11_CPP_STANDARD -std=c++1z) # Experimental C++17 support

if (NOT WIN32)
    add_definitions(
            -Wall
            -Wextra
    )
endif()

# Check for Link Time Optimizations with this compiler
include(CheckIPOSupported)
check_ipo_supported(RESULT LTO_SUPPORTED OUTPUT LTO_ERROR)

if( LTO_SUPPORTED )
    message(STATUS "IPO / LTO enabled")
else()
    message(STATUS "IPO / LTO not supported: <${LTO_ERROR}>")
endif()

find_package(Boost ${BOOST_VERSION_REQUIRED} REQUIRED program_options system)
include_directories(SYSTEM ${Boost_INCLUDE_DIRS})

include_directories(SYSTEM ${PYTHON_INCLUDE_DIRS})

set(SOURCE_FILES src/main.cpp
        src/CollectiveRiskDilemma.cpp
        include/Dyrwin/crd/CollectiveRiskDilemma.h
        src/CRDPlayer.cpp src/CRDSimulator.cpp
        include/Dyrwin/crd/CRDSimulator.h
        include/Dyrwin/Utils.h src/SeedGenerator.cpp
        include/Dyrwin/SeedGenerator.h
        include/Dyrwin/CommandLineParsing.h
        include/Dyrwin/ipd/IteratedPrisonersDilemma.cpp)
add_executable(Dyrwin ${SOURCE_FILES})
target_link_libraries(Dyrwin ${Boost_LIBRARIES} boost_program_options)
set_target_properties(Dyrwin PROPERTIES CMAKE_CXX_FLAGS lboost_program_options)

set(PYTHON_MORAN_SOURCE
        include/Dyrwin/SeedGenerator.h src/SeedGenerator.cpp
        include/Dyrwin/PyMoran/PDImitation.h
        src/PyMoran/PDImitation.cpp)
add_executable(pmoran src/PyMoran/pmain.cpp
        include/Dyrwin/CommandLineParsing.h ${PYTHON_MORAN_SOURCE})
TARGET_LINK_LIBRARIES(pmoran ${Boost_LIBRARIES} boost_program_options)
set_target_properties(pmoran PROPERTIES CMAKE_CXX_FLAGS lboost_program_options)

# Python bindings
add_subdirectory(pybind11)
pybind11_add_module(EGTtools SHARED ${PYTHON_MORAN_SOURCE} src/PyMoran/PDImitationWrapper.cpp)
# target_link_libraries(EGTtools ${${BOOST_PYTHON_LIBRARY_NAME}} ${PYTHON_LIBRARY})
# set_target_properties(EGTtools PROPERTIES PREFIX "")
# set_target_properties(EGTtools PROPERTIES SUFFIX ".so")
