cmake_minimum_required (VERSION 3.0)

find_package(Boost 1.53 REQUIRED)
include_directories(${Boost_INCLUDE_DIRS})

find_package(Eigen3 3.2 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

if (MAKE_PYTHON)
    find_package(PythonLibs 2.7 REQUIRED)
    find_package(Boost 1.53 COMPONENTS python REQUIRED)
    # Save version of python libs loaded to run tests later.
    set(PYTHONLIBS_VERSION_STRING ${PYTHONLIBS_VERSION_STRING} PARENT_SCOPE)
    include_directories(${PYTHON_INCLUDE_DIRS})
endif()

if (MAKE_MDP)
    add_library(AIToolboxMDP
        Impl/Seeder.cpp
        MDP/Experience.cpp
        MDP/Utils.cpp
        MDP/Model.cpp
        MDP/SparseExperience.cpp
        MDP/SparseModel.cpp
        MDP/IO.cpp
        MDP/Algorithms/QLearning.cpp
        MDP/Algorithms/SARSA.cpp
        MDP/Policies/Policy.cpp
        MDP/Policies/QPolicyInterface.cpp
        MDP/Policies/QGreedyPolicy.cpp
        MDP/Policies/QSoftmaxPolicy.cpp
        MDP/Policies/WoLFPolicy.cpp
        FactoredMDP/FactoredContainer.cpp)

    if (MAKE_PYTHON)
        add_library(MDP SHARED
            Impl/Seeder.cpp
            MDP/Utils.cpp
            MDP/Experience.cpp
            MDP/SparseExperience.cpp
            MDP/Model.cpp
            MDP/SparseModel.cpp
            MDP/Algorithms/QLearning.cpp
            MDP/Algorithms/SARSA.cpp
            MDP/Policies/Policy.cpp
            MDP/Policies/QPolicyInterface.cpp
            MDP/Policies/QGreedyPolicy.cpp
            MDP/Policies/QSoftmaxPolicy.cpp
            MDP/Policies/WoLFPolicy.cpp
            Python/Utils.cpp
            Python/MDP/Experience.cpp
            Python/MDP/RLModel.cpp
            Python/MDP/SparseExperience.cpp
            Python/MDP/SparseRLModel.cpp
            Python/MDP/Model.cpp
            Python/MDP/SparseModel.cpp
            Python/MDP/Algorithms/QLearning.cpp
            Python/MDP/Algorithms/SARSA.cpp
            Python/MDP/Algorithms/ValueIteration.cpp
            Python/MDP/Algorithms/PrioritizedSweeping.cpp
            Python/MDP/Algorithms/MCTS.cpp
            Python/MDP/PolicyInterface.cpp
            Python/MDP/Policies/QPolicyInterface.cpp
            Python/MDP/Policies/Policy.cpp
            Python/MDP/Policies/QGreedyPolicy.cpp
            Python/MDP/Policies/QSoftmaxPolicy.cpp
            Python/MDP/Policies/EpsilonPolicy.cpp
            Python/MDP/Policies/WoLFPolicy.cpp
            Python/MDPWrappers.cpp)
        target_link_libraries(MDP ${Boost_PYTHON_LIBRARY})
        set_target_properties(MDP PROPERTIES PREFIX "")
    endif()
endif()

if (MAKE_POMDP)
    find_package(LpSolve REQUIRED)
#   find_package(COIN REQUIRED)

    include_directories(${LPSOLVE_INCLUDE_DIR})
#   include_directories(${COIN_INCLUDE_DIR})

    add_library(AIToolboxPOMDP
        POMDP/Utils.cpp
        POMDP/Algorithms/IncrementalPruning.cpp
        POMDP/Algorithms/Witness.cpp
        POMDP/Algorithms/PBVI.cpp
        POMDP/Algorithms/PERSEUS.cpp
        POMDP/Algorithms/AMDP.cpp
        POMDP/Algorithms/Utils/WitnessLP_lpsolve.cpp
#       POMDP/Algorithms/Utils/WitnessLP_clp.cpp
        POMDP/Policies/Policy.cpp)

    target_link_libraries(AIToolboxPOMDP AIToolboxMDP
#           ${COIN_CLP_LIBRARY} ${COIN_COIN_UTILS_LIBRARY}
            ${LPSOLVE_LIBRARIES})

    if (MAKE_PYTHON)
        add_library(POMDP SHARED
            Python/POMDP/Model.cpp
            Python/POMDP/SparseModel.cpp
            Python/POMDP/Algorithms/POMCP.cpp
            Python/POMDP/Algorithms/Witness.cpp
            Python/POMDP/Algorithms/IncrementalPruning.cpp
            Python/POMDP/Algorithms/QMDP.cpp
            Python/POMDP/Algorithms/RTBSS.cpp
            Python/POMDP/Algorithms/AMDP.cpp
            Python/POMDP/Algorithms/PERSEUS.cpp
            Python/POMDPWrappers.cpp)
        target_link_libraries(POMDP ${Boost_PYTHON_LIBRARY} MDP)
        set_target_properties(POMDP PROPERTIES PREFIX "")
    endif()
endif()
