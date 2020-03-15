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
        ED/ReplicatorDynamics.cpp
        RL/Agent.cpp
        RL/BatchQLearningAgent.cpp
        RL/BatchQLearningForgetAgent.cpp
        RL/QLearningAgent.cpp
        RL/ESARSA.cpp
        RL/HistericQLearningAgent.cpp
        RL/RothErevAgent.cpp
        RL/DiscountedQLearning.cpp
        RL/CrdSim.cpp
        RL/Utils.cpp
        RL/PopContainer.cpp
        RL/algorithms/QLearning.cpp
        RL/simulators/CrdIslands.cpp
        AbouCRD/CollectiveRiskDilemma.cpp
        AbouCRD/CRDPlayer.cpp
        AbouCRD/CRDSimulator.cpp
        AbouCRD/DataStruct.cpp RL/Data.cpp)

# TODO: Separate libraries into EGT & RL
set_target_properties(EGTTools PROPERTIES INTERPROCEDURAL_OPTIMIZATION ${LTO_SUPPORTED})
set_target_properties(EGTTools PROPERTIES POSITION_INDEPENDENT_CODE TRUE)
# Fixing "ld: warning: direct access in function '...' from file '...' to global weak symbol '...' from file '...' means the weak symbol cannot
# be overridden at runtime. This was likely caused by different translation units being compiled with different visibility settings."
set_target_properties(EGTTools PROPERTIES C_VISIBILITY_PRESET hidden CXX_VISIBILITY_PRESET hidden)
# The new Eigen version does not required to be declared in target_link_libraries
# previously target_link_libraries(EGTTools Eigen3::Eigen)
target_link_libraries(EGTTools)
if (OPENMP_FOUND)
    target_link_libraries(EGTTools OpenMP::OpenMP_CXX)
endif ()