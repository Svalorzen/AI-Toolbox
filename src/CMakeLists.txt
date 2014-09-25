cmake_minimum_required (VERSION 2.6)

find_package(Boost 1.53 REQUIRED)
include_directories(${Boost_INCLUDE_DIRS})

if (MAKE_MDP)
    add_library(AIToolboxMDP
        Impl/Seeder.cpp
        MDP/Experience.cpp
        MDP/Utils.cpp
        MDP/Model.cpp
        MDP/RLModel.cpp
        MDP/IO.cpp
        MDP/Algorithms/ValueIteration.cpp
        MDP/Policies/Policy.cpp
        MDP/Policies/QPolicyInterface.cpp
        MDP/Policies/QGreedyPolicy.cpp
        MDP/Policies/WoLFPolicy.cpp)
endif()

if (MAKE_POMDP)
    find_package(LpSolve REQUIRED)

    include_directories(${LPSOLVE_INCLUDE_DIR})

    add_library(AIToolboxPOMDP
        POMDP/Utils.cpp
        POMDP/Algorithms/IncrementalPruning.cpp
        POMDP/Algorithms/PBVI.cpp
        POMDP/Algorithms/QMDP.cpp
        POMDP/Algorithms/Utils/WitnessLP_lpsolve.cpp
        POMDP/Policies/Policy.cpp)

    target_link_libraries(AIToolboxPOMDP AIToolboxMDP ${LPSOLVE_LIBRARIES})
endif()
