cmake_minimum_required(VERSION 3.9)

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

if (MAKE_PMORAN)
    add_executable(pmoran SED/pmain.cpp ${PYTHON_MORAN_SOURCE})
    target_link_libraries(pmoran ${Boost_LIBRARIES} boost_program_options)
    set_target_properties(pmoran PROPERTIES CMAKE_CXX_FLAGS lboost_program_options)
    set_target_properties(pmoran PROPERTIES INTERPROCEDURAL_OPTIMIZATION ${LTO_SUPPORTED})
    target_link_libraries(pmoran EGTTools)
endif ()

if (MAKE_CRD)
    add_executable(crd RL/main.cpp ${CRD_RL_SOURCE} ../include/Dyrwin/RL/CrdSim.hpp)
    target_link_libraries(crd ${Boost_LIBRARIES} boost_program_options)
    set_target_properties(crd PROPERTIES CMAKE_CXX_FLAGS lboost_program_options)
    set_target_properties(crd PROPERTIES INTERPROCEDURAL_OPTIMIZATION ${LTO_SUPPORTED})
    target_link_libraries(crd EGTTools)
endif ()

#if (MAKE_CRD_UNCERTAINTY)
#    add_executable(crdTU RL/main_uncertainty.cpp ${CRD_RL_SOURCE})
#    target_link_libraries(crdTU ${Boost_LIBRARIES} boost_program_options)
#    set_target_properties(crdTU PROPERTIES CMAKE_CXX_FLAGS lboost_program_options)
##    set_target_properties(crdTU PROPERTIES INTERPROCEDURAL_OPTIMIZATION ${LTO_SUPPORTED})
#    target_link_libraries(crdTU EGTTools)
#endif ()

#if (MAKE_CRD_POP)
#    add_executable(crdP RL/main_pop.cpp ${CRD_RL_SOURCE})
#    target_link_libraries(crdP ${Boost_LIBRARIES} boost_program_options)
#    set_target_properties(crdP PROPERTIES CMAKE_CXX_FLAGS lboost_program_options)
#    set_target_properties(crdP PROPERTIES INTERPROCEDURAL_OPTIMIZATION ${LTO_SUPPORTED})
#    target_link_libraries(crdP EGTTools)
#endif ()

#add_executable(crdC RL/main_conditional.cpp ${CRD_RL_SOURCE})
#target_link_libraries(crdC ${Boost_LIBRARIES} boost_program_options)
#set_target_properties(crdC PROPERTIES CMAKE_CXX_FLAGS lboost_program_options)
#set_target_properties(crdC PROPERTIES INTERPROCEDURAL_OPTIMIZATION ${LTO_SUPPORTED})
#target_link_libraries(crdC EGTTools)
#
#add_executable(crdD RL/main_democracy.cpp ${CRD_RL_SOURCE})
#target_link_libraries(crdD ${Boost_LIBRARIES} boost_program_options)
#set_target_properties(crdD PROPERTIES CMAKE_CXX_FLAGS lboost_program_options)
#set_target_properties(crdD PROPERTIES INTERPROCEDURAL_OPTIMIZATION ${LTO_SUPPORTED})
#target_link_libraries(crdD EGTTools)
