cmake_minimum_required(VERSION 3.9) # CMP0069 NEW

if (MAKE_MDP)
    add_library(AIToolboxMDP
        Impl/Seeder.cpp
        Impl/CassandraParser.cpp
        Utils/Combinatorics.cpp
        Utils/Probability.cpp
        Utils/Polytope.cpp
        Utils/LP/LpSolveWrapper.cpp
        Tools/Statistics.cpp
        Bandit/Algorithms/RollingAverage.cpp
        Bandit/Policies/EpsilonPolicy.cpp
        Bandit/Policies/RandomPolicy.cpp
        Bandit/Policies/QGreedyPolicy.cpp
        Bandit/Policies/QSoftmaxPolicy.cpp
        Bandit/Policies/ThompsonSamplingPolicy.cpp
        Bandit/Policies/LRPPolicy.cpp
        Bandit/Policies/ESRLPolicy.cpp
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
        MDP/Algorithms/Utils/OffPolicyTemplate.cpp
        MDP/Policies/PolicyWrapper.cpp
        MDP/Policies/Policy.cpp
        MDP/Policies/EpsilonPolicy.cpp
        MDP/Policies/QPolicyInterface.cpp
        MDP/Policies/QGreedyPolicy.cpp
        MDP/Policies/QSoftmaxPolicy.cpp
        MDP/Policies/WoLFPolicy.cpp
        MDP/Policies/PGAAPPPolicy.cpp
        MDP/Environments/Utils/GridWorld.cpp
    )
    set_target_properties(AIToolboxMDP PROPERTIES INTERPROCEDURAL_OPTIMIZATION ${LTO_SUPPORTED})
    target_link_libraries(AIToolboxMDP ${LPSOLVE_LIBRARIES})
endif()

if (MAKE_POMDP)
    add_library(AIToolboxPOMDP
        POMDP/Utils.cpp
        POMDP/IO.cpp
        POMDP/Algorithms/AMDP.cpp
        POMDP/Algorithms/GapMin.cpp
        POMDP/Algorithms/IncrementalPruning.cpp
        POMDP/Algorithms/LinearSupport.cpp
        POMDP/Algorithms/PBVI.cpp
        POMDP/Algorithms/PERSEUS.cpp
        POMDP/Algorithms/BlindStrategies.cpp
        POMDP/Algorithms/FastInformedBound.cpp
        POMDP/Algorithms/QMDP.cpp
        POMDP/Algorithms/Witness.cpp
        POMDP/Policies/Policy.cpp
    )
    set_target_properties(AIToolboxPOMDP PROPERTIES INTERPROCEDURAL_OPTIMIZATION ${LTO_SUPPORTED})
    target_link_libraries(AIToolboxPOMDP AIToolboxMDP ${LPSOLVE_LIBRARIES})
endif()

if (MAKE_FMDP)
    add_library(AIToolboxFMDP
        Factored/Utils/Trie.cpp
        Factored/Utils/FasterTrie.cpp
        Factored/Utils/Core.cpp
        Factored/Utils/FactoredMatrix.cpp
        Factored/Utils/FactoredVectorOps.cpp
        Factored/Utils/FactoredMatrix2DOps.cpp
        Factored/Utils/BayesianNetwork.cpp
        Factored/Bandit/Algorithms/Utils/VariableElimination.cpp
        Factored/Bandit/Algorithms/Utils/MultiObjectiveVariableElimination.cpp
        Factored/Bandit/Algorithms/Utils/UCVE.cpp
        Factored/Bandit/Algorithms/LLR.cpp
        Factored/Bandit/Algorithms/MAUCE.cpp
        Factored/Bandit/Algorithms/RollingAverage.cpp
        Factored/Bandit/Policies/SingleActionPolicy.cpp
        Factored/Bandit/Policies/RandomPolicy.cpp
        Factored/Bandit/Policies/EpsilonPolicy.cpp
        Factored/Bandit/Policies/QGreedyPolicy.cpp
        Factored/Bandit/Policies/ThompsonSamplingPolicy.cpp
        Factored/MDP/Utils.cpp
        Factored/MDP/Model.cpp
        Factored/MDP/CooperativeExperience.cpp
        Factored/MDP/CooperativeRLModel.cpp
        Factored/MDP/CooperativeModel.cpp
        Factored/MDP/Policies/EpsilonPolicy.cpp
        Factored/MDP/Policies/QGreedyPolicy.cpp
        Factored/MDP/Algorithms/Utils/FactoredLP.cpp
        Factored/MDP/Algorithms/Utils/CPSQueue.cpp
        Factored/MDP/Algorithms/SparseCooperativeQLearning.cpp
        Factored/MDP/Algorithms/JointActionLearner.cpp
        Factored/MDP/Algorithms/LinearProgramming.cpp
        Factored/MDP/Environments/SysAdminRing.cpp
        Factored/MDP/Environments/SysAdminGrid.cpp
        Factored/MDP/Environments/TigerAntelope.cpp
    )
    set_target_properties(AIToolboxFMDP PROPERTIES INTERPROCEDURAL_OPTIMIZATION ${LTO_SUPPORTED})
    target_link_libraries(AIToolboxFMDP AIToolboxMDP ${LPSOLVE_LIBRARIES})
endif()

if (MAKE_PYTHON)
    # Wrappers -- if unused macros will skip them
    set(PYTHON_FILES
        Python/AIToolboxWrappers.cpp
        Python/BanditWrappers.cpp
        Python/MDPWrappers.cpp
        Python/POMDPWrappers.cpp
        Python/FactoredWrappers.cpp
        # Python Utilities
        Python/Utils.cpp
        Python/Types.cpp
    )

    if (MAKE_MDP)
        set(PYTHON_FILES ${PYTHON_FILES}
            # Bandits
            Python/Bandit/Algorithms/RollingAverage.cpp
            Python/Bandit/Policies/PolicyInterface.cpp
            Python/Bandit/Policies/EpsilonPolicy.cpp
            Python/Bandit/Policies/RandomPolicy.cpp
            Python/Bandit/Policies/QGreedyPolicy.cpp
            Python/Bandit/Policies/QSoftmaxPolicy.cpp
            Python/Bandit/Policies/ThompsonSamplingPolicy.cpp
            Python/Bandit/Policies/LRPPolicy.cpp
            Python/Bandit/Policies/ESRLPolicy.cpp
            Python/BanditWrappers.cpp
            # MDP
            Python/MDP/Utils.cpp
            Python/MDP/Types.cpp
            Python/MDP/Experience.cpp
            Python/MDP/RLModel.cpp
            Python/MDP/SparseExperience.cpp
            Python/MDP/SparseRLModel.cpp
            Python/MDP/Model.cpp
            Python/MDP/SparseModel.cpp
            Python/MDP/GenerativeModelPython.cpp
            Python/MDP/Algorithms/QLearning.cpp
            Python/MDP/Algorithms/HystereticQLearning.cpp
            Python/MDP/Algorithms/SARSA.cpp
            Python/MDP/Algorithms/ExpectedSARSA.cpp
            Python/MDP/Algorithms/SARSAL.cpp
            Python/MDP/Algorithms/QL.cpp
            Python/MDP/Algorithms/ValueIteration.cpp
            Python/MDP/Algorithms/PolicyIteration.cpp
            Python/MDP/Algorithms/PrioritizedSweeping.cpp
            Python/MDP/Algorithms/MCTS.cpp
            Python/MDP/Policies/PolicyInterface.cpp
            Python/MDP/Policies/QPolicyInterface.cpp
            Python/MDP/Policies/Policy.cpp
            Python/MDP/Policies/QGreedyPolicy.cpp
            Python/MDP/Policies/QSoftmaxPolicy.cpp
            Python/MDP/Policies/EpsilonPolicy.cpp
            Python/MDP/Policies/WoLFPolicy.cpp
            Python/MDP/Policies/PGAAPPPolicy.cpp
        )
        set(PYTHON_LINKS AIToolboxMDP)
        set(PYTHON_DEFINES -DAITOOLBOX_EXPORT_BANDIT -DAITOOLBOX_EXPORT_MDP)
    endif()

    if (MAKE_POMDP)
        # This may need to be added to FMDP too if we port FactoredLP to Python.
        if ("${LPSOLVE_LIBRARIES}" MATCHES "\\.a")
            message(WARNING "LpSolve libraries found could be static, linking against Python dynamic libraries may not work!")
        endif()
        set(PYTHON_FILES ${PYTHON_FILES}
            Python/POMDP/Types.cpp
            Python/POMDP/Utils.cpp
            Python/POMDP/Model.cpp
            Python/POMDP/SparseModel.cpp
            Python/POMDP/Algorithms/POMCP.cpp
            Python/POMDP/Algorithms/GapMin.cpp
            Python/POMDP/Algorithms/Witness.cpp
            Python/POMDP/Algorithms/IncrementalPruning.cpp
            Python/POMDP/Algorithms/LinearSupport.cpp
            Python/POMDP/Algorithms/QMDP.cpp
            Python/POMDP/Algorithms/RTBSS.cpp
            Python/POMDP/Algorithms/AMDP.cpp
            Python/POMDP/Algorithms/PERSEUS.cpp
            Python/POMDP/Algorithms/PBVI.cpp
            Python/POMDP/Policies/PolicyInterface.cpp
            Python/POMDP/Policies/Policy.cpp
        )
        set(PYTHON_LINKS AIToolboxPOMDP ${PYTHON_LINKS} ${LPSOLVE_LIBRARIES})
        set(PYTHON_DEFINES ${PYTHON_DEFINES} -DAITOOLBOX_EXPORT_POMDP)
    endif()

    if (MAKE_FMDP)
        set(PYTHON_FILES ${PYTHON_FILES}
            Python/Factored/MDP/Algorithms/JointActionLearner.cpp
        )
        set(PYTHON_LINKS AIToolboxFMDP ${PYTHON_LINKS})
        set(PYTHON_DEFINES ${PYTHON_DEFINES} -DAITOOLBOX_EXPORT_FACTORED)
    endif()

    add_library(AIToolbox SHARED ${PYTHON_FILES})
    target_compile_definitions(AIToolbox PUBLIC ${PYTHON_DEFINES})
    target_link_libraries(AIToolbox ${${BOOST_PYTHON_LIBRARY_NAME}} ${PYTHON_LIBRARY} ${PYTHON_LINKS})
    set_target_properties(AIToolbox PROPERTIES PREFIX "")
    set_target_properties(AIToolbox PROPERTIES INTERPROCEDURAL_OPTIMIZATION ${LTO_SUPPORTED})
endif()
