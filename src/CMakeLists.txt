cmake_minimum_required (VERSION 2.6)

find_package(Boost 1.53 REQUIRED)
include_directories(${Boost_INCLUDE_DIRS})

add_library(AIToolboxMDP
    Impl/Seeder.cpp
    MDP/Experience.cpp
    MDP/Utils.cpp
    MDP/Model.cpp
    MDP/RLModel.cpp
    MDP/IO.cpp
    MDP/Algorithms/QLearning.cpp
    MDP/Algorithms/ValueIteration.cpp
    MDP/Policies/Policy.cpp
    MDP/Policies/QPolicyInterface.cpp
    MDP/Policies/QGreedyPolicy.cpp
    MDP/Policies/WoLFPolicy.cpp)
