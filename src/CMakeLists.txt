cmake_minimum_required(VERSION 3.9)

add_library(EGTTools
        SeedGenerator.cpp
        Distributions.cpp
        SED/Utils.cpp
        SED/MoranProcess.cpp
        SED/PDImitation.cpp
        SED/StochDynamics.cpp
        SED/TraulsenMoran.cpp
        SED/structure/Group.cpp
        SED/structure/GarciaGroup.cpp
        SED/behaviors/CrdBehaviors.cpp
        SED/games/CrdGame.cpp
        SED/games/CrdGameTU.cpp
        SED/MLS.cpp
        RL/Agent.cpp
        RL/BatchQLearningAgent.cpp
        RL/QLearningAgent.cpp
        RL/HistericQLearningAgent.cpp
        RL/RothErevAgent.cpp
        RL/CrdSim.cpp
        RL/Utils.cpp
        RL/PopContainer.cpp
        AbouCRD/CollectiveRiskDilemma.cpp
        AbouCRD/CRDPlayer.cpp
        AbouCRD/CRDSimulator.cpp
        AbouCRD/DataStruct.cpp RL/Data.cpp)
#set_target_properties(EGTTools PROPERTIES INTERPROCEDURAL_OPTIMIZATION ${LTO_SUPPORTED})
target_link_libraries(EGTTools Eigen3::Eigen)
if (OPENMP_FOUND)
    target_link_libraries(EGTTools OpenMP::OpenMP_CXX)
endif ()