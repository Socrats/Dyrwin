# Dyrwin
## Simulator for Evolutionary Game Theory problems

The intention of this repository is twofold:
- Reproduce results from top-tier papers that use EGT to solve different kinds of problems;
- and to provide a c++ framework to perform evolutionary analysis of game theoretical scenarios.

## Requirements
- Requires C++ compiler with c++17 support
- Eigen 3.3
- pybind 11 for the python bindings
- Boost >=1.67 and boost program_options

## Build
To install run:
````
mkdir build
cd build
cmake ..
make -j 4
````

## Execution
To get help on how to run the simulation execute:
````bash
./dyrwin --help
````

If you want your results to to be stored in an output file, you need to provide a file path

You can also run the simulation through a python configuration file like the ones provided in ./examples:

Some of the code to make build the CMakeLists.txt was taken from: https://github.com/Svalorzen/AI-Toolbox/blob/master/CMakeLists.txt
