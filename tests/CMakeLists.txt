cmake_minimum_required(VERSION 3.9)


add_executable(testTU TestTimingUncertainty.cpp)
set_target_properties(testTU PROPERTIES INTERPROCEDURAL_OPTIMIZATION ${LTO_SUPPORTED})
target_link_libraries(testTU EGTTools)

add_executable(testCrdGame TestCrdGame.cpp)
set_target_properties(testCrdGame PROPERTIES INTERPROCEDURAL_OPTIMIZATION ${LTO_SUPPORTED})
target_link_libraries(testCrdGame EGTTools)

add_executable(testGeometric TestGeometric.cpp)
set_target_properties(testGeometric PROPERTIES INTERPROCEDURAL_OPTIMIZATION ${LTO_SUPPORTED})
target_link_libraries(testGeometric EGTTools)

add_executable(testCounter TestCounter.cpp)
set_target_properties(testCounter PROPERTIES INTERPROCEDURAL_OPTIMIZATION ${LTO_SUPPORTED})

add_executable(testPM TestPairwiseMoran.cpp)
#set_target_properties(testPM PROPERTIES INTERPROCEDURAL_OPTIMIZATION ${LTO_SUPPORTED})
target_link_libraries(testPM EGTTools)

add_executable(testSP TestIntSampling.cpp)
set_target_properties(testPM PROPERTIES INTERPROCEDURAL_OPTIMIZATION ${LTO_SUPPORTED})
