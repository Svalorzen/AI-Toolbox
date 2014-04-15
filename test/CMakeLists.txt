cmake_minimum_required (VERSION 2.6)

find_package(Boost 1.53 COMPONENTS unit_test_framework REQUIRED)
include_directories(${Boost_INCLUDE_DIRS})

function (AddTestMDP name)
    add_executable(${name}Tests MDP/${name}Tests.cpp)
    target_link_libraries(${name}Tests AIToolboxMDP ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY} ${ARGN})
    add_test(NAME ${name} WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY} COMMAND $<TARGET_FILE:${name}Tests>)
endfunction (AddTestMDP)

AddTestMDP(Experience)
AddTestMDP(RLModel)
AddTestMDP(QLearning)
AddTestMDP(ValueIteration)
AddTestMDP(WoLFPolicy)
