cmake_minimum_required(VERSION 3.9)
project(Dyrwin LANGUAGES CXX)

set(MAKE_PYTHON 1)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(BOOST_VERSION_REQUIRED 1.68)
set(PYTHON_VERSION 3.7)

#if (NOT WIN32)
#    add_definitions(
#            -Wall
#            -Wextra
#    )
#endif()

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

if (MAKE_PYTHON)
    # Try to find out which version of Python we should be targeting depending
    # on which interpreter is found. If the version has been selected
    # explicitly we search for it directly.
    find_package(PythonInterp ${PYTHON_VERSION} REQUIRED)
#    set(PYTHON_VERSION ${PYTHON_VERSION_MAJOR})
    message("-- Python version: " ${PYTHON_VERSION})

    # WARNING: The COMPONENTS parts of Boost here may need to be changed
    # depending on your system. For example, you may need to rename `python3` to
    # `python37`, depending on how your libraries are made.
    # This setup works for Ubuntu, but feel free to change it in Windows.
    if (${PYTHON_VERSION_MAJOR} EQUAL 3)
        find_package(PythonLibs 3 REQUIRED)
        find_package(Boost ${BOOST_VERSION_REQUIRED} COMPONENTS python3 REQUIRED)
        set(BOOST_PYTHON_LIBRARY_NAME "Boost_PYTHON3_LIBRARY")
    else ()
        find_package(PythonLibs 2.7 EXACT REQUIRED)
        find_package(Boost ${BOOST_VERSION_REQUIRED} COMPONENTS python REQUIRED)
        set(BOOST_PYTHON_LIBRARY_NAME "Boost_PYTHON_LIBRARY")
    endif ()

    include_directories(SYSTEM ${PYTHON_INCLUDE_DIRS})
endif ()

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

add_library(EGTtools SHARED ${PYTHON_MORAN_SOURCE} include/Dyrwin/PyMoran/PDImitationWrapper.cpp)
target_link_libraries(EGTtools ${${BOOST_PYTHON_LIBRARY_NAME}} ${PYTHON_LIBRARY})
set_target_properties(EGTtools PROPERTIES PREFIX "")
