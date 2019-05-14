cmake_minimum_required(VERSION 3.9)

add_library(EGTTools
        SeedGenerator.cpp
        SED/MoranProcess.cpp
        SED/PDImitation.cpp
        SED/StochDynamics.cpp
        SED/TraulsenMoran.cpp
        SED/structure/Group.cpp
        RL/Agent.cpp
        RL/BatchQLearningAgent.cpp
        RL/QLearningAgent.cpp
        RL/HistericQLearningAgent.cpp
        RL/RothErevAgent.cpp
        RL/CrdSim.cpp
        RL/Utils.cpp
        RL/PopContainer.cpp)
set_target_properties(EGTTools PROPERTIES INTERPROCEDURAL_OPTIMIZATION ${LTO_SUPPORTED})
target_link_libraries(EGTTools Eigen3::Eigen)
if (OPENMP_FOUND)
    target_link_libraries(EGTTools OpenMP::OpenMP_CXX)
endif ()