cmake_minimum_required (VERSION 2.6)

add_executable(solve_mdp main.cpp)

find_package(Boost 1.53 COMPONENTS system filesystem REQUIRED)
include_directories(${Boost_INCLUDE_DIRS})

target_link_libraries(solve_mdp AIToolbox ${Boost_FILESYSTEM_LIBRARY} ${Boost_SYSTEM_LIBRARY})



