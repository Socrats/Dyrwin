cmake_minimum_required(VERSION 3.9)


add_executable(testTU TestTimingUncertainty.cpp)
target_link_libraries(testTU ${Boost_LIBRARIES} boost_program_options)
set_target_properties(testTU PROPERTIES CMAKE_CXX_FLAGS lboost_program_options)
set_target_properties(testTU PROPERTIES INTERPROCEDURAL_OPTIMIZATION ${LTO_SUPPORTED})
target_link_libraries(testTU EGTTools)
