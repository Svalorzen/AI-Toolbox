cmake_minimum_required (VERSION 2.6)

find_package(Boost 1.53 COMPONENTS system filesystem unit_test_framework REQUIRED)
include_directories(${Boost_INCLUDE_DIRS})

add_executable(solve_mdp main.cpp)
target_link_libraries(solve_mdp AIToolbox ${Boost_FILESYSTEM_LIBRARY} ${Boost_SYSTEM_LIBRARY})

function (AddTest name)
    add_executable(${name} ${name}.cpp)
    target_link_libraries(${name} AIToolbox ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY} ${ARGN})
    add_test(NAME ${name} WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY} COMMAND $<TARGET_FILE:${name}>)
endfunction (AddTest)

AddTest(ExperienceTests)
AddTest(MDP_RLModelTests)
AddTest(QLearningTests)
AddTest(WoLFPolicyTests)
