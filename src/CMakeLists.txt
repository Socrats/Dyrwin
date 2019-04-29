cmake_minimum_required(VERSION 3.9)

add_library(EGTTools
        ../include/Dyrwin/SeedGenerator.h
        ../include/Dyrwin/Types.h
        ../include/Dyrwin/Utils.h
        ../include/Dyrwin/SED/MLS.hpp
        ../include/Dyrwin/SED/MoranProcess.hpp
        ../include/Dyrwin/SED/StochDynamics.h
        ../include/Dyrwin/SED/TraulsenMoran.h
        ../include/Dyrwin/SED/Utils.hpp
        ../include/Dyrwin/RL/Agent.h
        ../include/Dyrwin/RL/BatchQLearningAgent.h
        ../include/Dyrwin/RL/HistericQlearningAgent.hpp
        ../include/Dyrwin/RL/QlearningAgent.h
        ../include/Dyrwin/RL/RLUtils.h
        ../include/Dyrwin/RL/RothErevAgent.h
        ../include/Dyrwin/RL/TimingUncertainty.hpp
        SeedGenerator.cpp
        SED/MoranProcess.cpp
        SED/PDImitation.cpp
        SED/StochDynamics.cpp
        SED/TraulsenMoran.cpp
        SED/structure/Group.cpp
        RL/Agent.cpp
        RL/BatchQLearningAgent.cpp
        RL/QLearningAgent.cpp
        )
set_target_properties(EGTTools PROPERTIES INTERPROCEDURAL_OPTIMIZATION ${LTO_SUPPORTED})
target_link_libraries(EGTTools Eigen3::Eigen)
if (OPENMP_FOUND)
    target_link_libraries(EGTTools OpenMP::OpenMP_CXX)
endif ()