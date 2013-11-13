cmake_minimum_required (VERSION 2.6)

add_executable(solve_mdp main.cpp Table.cpp)

target_link_libraries(solve_mdp MDPToolbox)



