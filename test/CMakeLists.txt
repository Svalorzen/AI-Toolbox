cmake_minimum_required (VERSION 2.6)

add_executable(solve_mdp main.cpp)

find_package(Boost COMPONENTS system filesystem REQUIRED)
target_link_libraries(solve_mdp MDPToolbox ${Boost_FILESYSTEM_LIBRARY} ${Boost_SYSTEM_LIBRARY})



