cmake_minimum_required(VERSION 3.9)

#add_executable(AbouCRD AbouCRD/main.cpp)
#target_link_libraries(AbouCRD ${Boost_LIBRARIES} boost_program_options)
#set_target_properties(AbouCRD PROPERTIES CMAKE_CXX_FLAGS lboost_program_options)
#set_target_properties(AbouCRD PROPERTIES INTERPROCEDURAL_OPTIMIZATION ${LTO_SUPPORTED})
#target_link_libraries(AbouCRD EGTTools)


add_executable(pmoran SED/pmain.cpp)
target_link_libraries(pmoran ${Boost_LIBRARIES} boost_program_options)
set_target_properties(pmoran PROPERTIES CMAKE_CXX_FLAGS lboost_program_options)
set_target_properties(pmoran PROPERTIES INTERPROCEDURAL_OPTIMIZATION ${LTO_SUPPORTED})
target_link_libraries(pmoran EGTTools)

add_executable(behaviors SED/crd_behaviors.cpp)
target_link_libraries(behaviors ${Boost_LIBRARIES} boost_program_options)
set_target_properties(behaviors PROPERTIES CMAKE_CXX_FLAGS lboost_program_options)
#set_target_properties(behaviors PROPERTIES INTERPROCEDURAL_OPTIMIZATION ${LTO_SUPPORTED})
target_link_libraries(behaviors EGTTools)

add_executable(crd RL/main.cpp)
target_link_libraries(crd ${Boost_LIBRARIES} boost_program_options)
set_target_properties(crd PROPERTIES CMAKE_CXX_FLAGS lboost_program_options)
set_target_properties(crd PROPERTIES INTERPROCEDURAL_OPTIMIZATION ${LTO_SUPPORTED})
target_link_libraries(crd EGTTools)


add_executable(crdTU RL/main_uncertainty.cpp)
target_link_libraries(crdTU ${Boost_LIBRARIES} boost_program_options)
set_target_properties(crdTU PROPERTIES CMAKE_CXX_FLAGS lboost_program_options)
set_target_properties(crdTU PROPERTIES INTERPROCEDURAL_OPTIMIZATION ${LTO_SUPPORTED})
target_link_libraries(crdTU EGTTools)

add_executable(crdP RL/main_pop.cpp)
target_link_libraries(crdP ${Boost_LIBRARIES} boost_program_options)
set_target_properties(crdP PROPERTIES CMAKE_CXX_FLAGS lboost_program_options)
set_target_properties(crdP PROPERTIES INTERPROCEDURAL_OPTIMIZATION ${LTO_SUPPORTED})
target_link_libraries(crdP EGTTools)

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
