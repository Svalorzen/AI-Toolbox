cmake_minimum_required (VERSION 3.0)

find_package(Boost 1.53 REQUIRED)
include_directories(${Boost_INCLUDE_DIRS})

find_package(Eigen3 3.2 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

if (MAKE_PYTHON)
    find_package(PythonLibs 2.7 REQUIRED)
    find_package(Boost 1.53 COMPONENTS python REQUIRED)
    # Remove junk at the end of the version that breaks requiring the interpreter for tests later.
    STRING(REGEX REPLACE "[^0-9]+$" "" PYTHONLIBS_VERSION_STRING_STRIPPED ${PYTHONLIBS_VERSION_STRING})
    # Save version of python libs loaded to run tests later.
    set(PYTHONLIBS_VERSION_STRING_STRIPPED ${PYTHONLIBS_VERSION_STRING_STRIPPED} PARENT_SCOPE)
    include_directories(${PYTHON_INCLUDE_DIRS})
endif()

if (MAKE_MDP)
    add_library(AIToolboxMDP
        Impl/Seeder.cpp
        Game/Policies/LRPPolicy.cpp
        MDP/Experience.cpp
        MDP/Utils.cpp
        MDP/Model.cpp
        MDP/SparseExperience.cpp
        MDP/SparseModel.cpp
        MDP/IO.cpp
        MDP/Algorithms/QLearning.cpp
        MDP/Algorithms/HystereticQLearning.cpp
        MDP/Algorithms/SARSA.cpp
        MDP/Algorithms/ExpectedSARSA.cpp
        MDP/Algorithms/SARSAL.cpp
        MDP/Algorithms/ValueIteration.cpp
        MDP/Algorithms/PolicyIteration.cpp
        MDP/Policies/Policy.cpp
        MDP/Policies/EpsilonPolicy.cpp
        MDP/Policies/QPolicyInterface.cpp
        MDP/Policies/QGreedyPolicy.cpp
        MDP/Policies/QSoftmaxPolicy.cpp
        MDP/Policies/WoLFPolicy.cpp
    )

    if (MAKE_PYTHON)
        add_library(MDP SHARED
            Impl/Seeder.cpp
            MDP/Utils.cpp
            MDP/Experience.cpp
            MDP/SparseExperience.cpp
            MDP/Model.cpp
            MDP/SparseModel.cpp
            MDP/Algorithms/QLearning.cpp
            MDP/Algorithms/HystereticQLearning.cpp
            MDP/Algorithms/SARSA.cpp
            MDP/Algorithms/ExpectedSARSA.cpp
            MDP/Algorithms/SARSAL.cpp
            MDP/Algorithms/ValueIteration.cpp
            MDP/Algorithms/PolicyIteration.cpp
            MDP/Policies/Policy.cpp
            MDP/Policies/EpsilonPolicy.cpp
            MDP/Policies/QPolicyInterface.cpp
            MDP/Policies/QGreedyPolicy.cpp
            MDP/Policies/QSoftmaxPolicy.cpp
            MDP/Policies/WoLFPolicy.cpp
            Python/Utils.cpp
            Python/Types.cpp
            Python/MDP/Utils.cpp
            Python/MDP/Types.cpp
            Python/MDP/Experience.cpp
            Python/MDP/RLModel.cpp
            Python/MDP/SparseExperience.cpp
            Python/MDP/SparseRLModel.cpp
            Python/MDP/Model.cpp
            Python/MDP/SparseModel.cpp
            Python/MDP/Algorithms/QLearning.cpp
            Python/MDP/Algorithms/HystereticQLearning.cpp
            Python/MDP/Algorithms/SARSA.cpp
            Python/MDP/Algorithms/ExpectedSARSA.cpp
            Python/MDP/Algorithms/SARSAL.cpp
            Python/MDP/Algorithms/ValueIteration.cpp
            Python/MDP/Algorithms/PolicyIteration.cpp
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

if (MAKE_FMDP OR MAKE_POMDP)
    find_package(LpSolve REQUIRED)
    include_directories(${LPSOLVE_INCLUDE_DIR})
endif()

if (MAKE_FMDP)
    add_library(AIToolboxFMDP
        LP/LpSolveWrapper.cpp
        FactoredMDP/FactoredContainer.cpp
        FactoredMDP/Utils.cpp
        FactoredMDP/Policies/EpsilonPolicy.cpp
        FactoredMDP/Policies/SingleActionPolicy.cpp
        FactoredMDP/Policies/QGreedyPolicy.cpp
        FactoredMDP/Algorithms/Utils/VariableElimination.cpp
        FactoredMDP/Algorithms/Utils/MultiObjectiveVariableElimination.cpp
        FactoredMDP/Algorithms/Utils/FactoredLP.cpp
        FactoredMDP/Algorithms/SparseCooperativeQLearning.cpp
        FactoredMDP/Algorithms/LLR.cpp)

    target_link_libraries(AIToolboxFMDP AIToolboxMDP ${LPSOLVE_LIBRARIES})
endif()

if (MAKE_POMDP)
    add_library(AIToolboxPOMDP
        LP/LpSolveWrapper.cpp
        POMDP/Utils.cpp
        POMDP/Algorithms/IncrementalPruning.cpp
        POMDP/Algorithms/Witness.cpp
        POMDP/Algorithms/PBVI.cpp
        POMDP/Algorithms/PERSEUS.cpp
        POMDP/Algorithms/AMDP.cpp
        POMDP/Algorithms/QMDP.cpp
        POMDP/Algorithms/Utils/WitnessLP.cpp
        POMDP/IO.cpp
        POMDP/Policies/Policy.cpp)

    target_link_libraries(AIToolboxPOMDP AIToolboxMDP ${LPSOLVE_LIBRARIES})

    if (MAKE_PYTHON)
        if ("${LPSOLVE_LIBRARIES}" MATCHES "\\.a")
            message(WARNING "LpSolve libraries found could be static, linking against Python dynamic libraries may not work!")
        endif()
        add_library(POMDP SHARED
            LP/LpSolveWrapper.cpp
            POMDP/Utils.cpp
            POMDP/Algorithms/IncrementalPruning.cpp
            POMDP/Algorithms/Witness.cpp
            POMDP/Algorithms/PBVI.cpp
            POMDP/Algorithms/PERSEUS.cpp
            POMDP/Algorithms/AMDP.cpp
            POMDP/Algorithms/QMDP.cpp
            POMDP/Policies/Policy.cpp
            POMDP/Algorithms/Utils/WitnessLP.cpp
            Python/POMDP/Types.cpp
            Python/POMDP/Utils.cpp
            Python/POMDP/Model.cpp
            Python/POMDP/SparseModel.cpp
            Python/POMDP/Algorithms/POMCP.cpp
            Python/POMDP/Algorithms/Witness.cpp
            Python/POMDP/Algorithms/IncrementalPruning.cpp
            Python/POMDP/Algorithms/QMDP.cpp
            Python/POMDP/Algorithms/RTBSS.cpp
            Python/POMDP/Algorithms/AMDP.cpp
            Python/POMDP/Algorithms/PERSEUS.cpp
            Python/POMDP/Algorithms/PBVI.cpp
            Python/POMDP/PolicyInterface.cpp
            Python/POMDP/Policies/Policy.cpp
            Python/POMDPWrappers.cpp)
        target_link_libraries(POMDP ${Boost_PYTHON_LIBRARY} MDP
            ${LPSOLVE_LIBRARIES})
        set_target_properties(POMDP PROPERTIES PREFIX "")
    endif()
endif()
