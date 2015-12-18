cmake_minimum_required (VERSION 2.6)

find_package(Boost 1.53 REQUIRED)
include_directories(${Boost_INCLUDE_DIRS})

find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

if (MAKE_MDP)
    add_library(AIToolboxMDP
        Impl/Seeder.cpp
        MDP/Experience.cpp
        MDP/Utils.cpp
        MDP/Model.cpp
        MDP/SparseExperience.cpp
        MDP/SparseModel.cpp
        MDP/IO.cpp
        MDP/Policies/Policy.cpp
        MDP/Policies/QPolicyInterface.cpp
        MDP/Policies/QGreedyPolicy.cpp
        MDP/Policies/WoLFPolicy.cpp)
endif()

if (MAKE_POMDP)
    find_package(LpSolve REQUIRED)
#    find_package(COIN REQUIRED)

    include_directories(${LPSOLVE_INCLUDE_DIR})
#    include_directories(${COIN_INCLUDE_DIR})

    add_library(AIToolboxPOMDP
        POMDP/Utils.cpp
        POMDP/Algorithms/IncrementalPruning.cpp
        POMDP/Algorithms/Witness.cpp
        POMDP/Algorithms/PBVI.cpp
        POMDP/Algorithms/PERSEUS.cpp
        POMDP/Algorithms/AMDP.cpp
        POMDP/Algorithms/Utils/WitnessLP_lpsolve.cpp
#        POMDP/Algorithms/Utils/WitnessLP_clp.cpp
        POMDP/Policies/Policy.cpp)

    target_link_libraries(AIToolboxPOMDP AIToolboxMDP
#           ${COIN_CLP_LIBRARY} ${COIN_COIN_UTILS_LIBRARY}
            ${LPSOLVE_LIBRARIES})
endif()
