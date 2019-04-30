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

# For additional Find library scripts
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${PROJECT_SOURCE_DIR}/cmake/Modules/")

##############################
##     Project Settings     ##
##############################

# Default is to build everything
if (NOT MAKE_ALL AND NOT MAKE_DYRWIN AND NOT MAKE_PMORAN AND NOT MAKE_PYTHON AND NOT MAKE_CRD AND NOT MAKE_CRD_UNCERTAINTY AND NOT MAKE_CRD_POP AND NOT MAKE_APPS)
    set(MAKE_ALL 1)
endif ()

if (MAKE_ALL)
    set(MAKE_DYRWIN 1)
    set(MAKE_PMORAN 1)
    set(MAKE_CRD 1)
    set(MAKE_CRD_UNCERTAINTY 1)
    set(MAKE_CRD_POP 1)
    set(MAKE_APPS 1)
    set(MAKE_PYTHON 1)
endif ()

#message("")
#message("Build type:  ${CMAKE_BUILD_TYPE}")
#message("Logging is  ${LOGGING_STATUS}")
#foreach(v MAKE_DYRWIN;MAKE_PMORAN;MAKE_CRD;MAKE_CRD_UNCERTAINTY;MAKE_CRD_POP;MAKE_APPS;MAKE_PYTHON)
#    if (${${v}})
#        message("${MAP_${v}}")
#    endif()
#endforeach(v)
#message("")

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

# Include the project directory
include_directories(${PROJECT_SOURCE_DIR}/include)
add_subdirectory(${PROJECT_SOURCE_DIR}/src)

if (MAKE_APPS)
    add_subdirectory(${PROJECT_SOURCE_DIR}/apps)
endif ()

# Python bindings
if (MAKE_PYTHON)
    # Wrappers -- if unused macros will skip them
    set(PYTHON_FILES
            src/SED/EGTtoolsWrapper.cpp
            )
    add_subdirectory(pybind11)
    pybind11_add_module(EGTtools SHARED ${PYTHON_FILES})
    target_link_libraries(EGTtools PRIVATE EGTTools)
endif ()


