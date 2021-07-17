cmake_minimum_required (VERSION 3.12) # CMP0069 NEW

set(GlobalFileDependencies
    ${PROJECT_SOURCE_DIR}/src/Impl/Seeder.cpp
    ${PROJECT_SOURCE_DIR}/src/Tools/Statistics.cpp
    ${PROJECT_SOURCE_DIR}/src/Utils/Adam.cpp
    ${PROJECT_SOURCE_DIR}/src/Utils/Combinatorics.cpp
    ${PROJECT_SOURCE_DIR}/src/Utils/Probability.cpp
    ${PROJECT_SOURCE_DIR}/src/Utils/LP/LpSolveWrapper.cpp
)
set(GlobalDependencies      ${LPSOLVE_LIBRARIES})
set(BanditDependencies      AIToolboxMDP)
set(MDPDependencies         AIToolboxMDP)
set(POMDPDependencies       AIToolboxMDP AIToolboxPOMDP)
set(FactoredDependencies    AIToolboxMDP AIToolboxFMDP)

function (AddTestGlobal name)
    set(exename Global_${name})
    add_executable(${exename}Tests ${name}Tests.cpp ${GlobalFileDependencies})
    target_link_libraries(${exename}Tests ${GlobalDependencies} ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY} ${ARGN})
    add_test(NAME ${exename} WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY} COMMAND $<TARGET_FILE:${exename}Tests>)
    set_target_properties(${exename}Tests PROPERTIES INTERPROCEDURAL_OPTIMIZATION ${LTO_SUPPORTED})
endfunction (AddTestGlobal)

function (AddTest dir name)
    string(REPLACE "/" "_" prefix ${dir})
    string(REPLACE "/" ";" lst ${dir})
    list(GET lst 0 type)

    set(exename ${prefix}_${name})
    add_executable(${exename}Tests ${dir}/${name}Tests.cpp)
    target_link_libraries(${exename}Tests ${${type}Dependencies} ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY} ${ARGN})
    add_test(NAME ${exename} WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY} COMMAND $<TARGET_FILE:${exename}Tests>)
    set_target_properties(${exename}Tests PROPERTIES INTERPROCEDURAL_OPTIMIZATION ${LTO_SUPPORTED})
endfunction (AddTest)

function (AddTestPython type name)
    set(exename ${type}_Python_${name})
    add_test(NAME ${exename} WORKING_DIRECTORY ${CMAKE_LIBRARY_OUTPUT_DIRECTORY} COMMAND ${Python_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/Python/${type}/${name}Tests.py)
endfunction (AddTestPython)

if (MAKE_MDP)
    AddTestGlobal(UtilsAdam)
    AddTestGlobal(UtilsCore)
    AddTestGlobal(UtilsProbability)
    AddTestGlobal(UtilsPrune)
    AddTestGlobal(Tools)

    AddTest(Bandit Model)
    AddTest(Bandit QGreedyPolicy)
    AddTest(Bandit QSoftmaxPolicy)
    AddTest(Bandit ThompsonSamplingPolicy)
    AddTest(Bandit LRPPolicy)
    AddTest(Bandit ESRLPolicy)

    AddTest(MDP Environments)

    AddTest(MDP Types)
    AddTest(MDP UtilsPolytope)

    AddTest(MDP Experience)
    AddTest(MDP Model)
    AddTest(MDP MaximumLikelihoodModel)
    AddTest(MDP ThompsonModel)
    AddTest(MDP SparseExperience)
    AddTest(MDP SparseModel)
    AddTest(MDP SparseMaximumLikelihoodModel)

    AddTest(MDP PGAAPPPolicy)
    AddTest(MDP QGreedyPolicy)
    AddTest(MDP WoLFPolicy)

    AddTest(MDP Dyna2)
    AddTest(MDP DynaQ)
    AddTest(MDP ExpectedSARSA)
    AddTest(MDP HystereticQLearning)
    AddTest(MDP MCTS)
    AddTest(MDP PolicyEvaluation)
    AddTest(MDP PolicyIteration)
    AddTest(MDP PrioritizedSweeping)
    AddTest(MDP QL)
    AddTest(MDP QLearning)
    AddTest(MDP DoubleQLearning)
    AddTest(MDP RetraceL)
    AddTest(MDP SARSA)
    AddTest(MDP SARSAL)
    AddTest(MDP TreeBackupL)
    AddTest(MDP ValueIteration)
    AddTest(MDP LinearProgramming)

    if (MAKE_PYTHON)
        AddTestPython(MDP Experience)
        AddTestPython(MDP Model)
        AddTestPython(MDP SparseModel)
        AddTestPython(MDP PolicyIteration)
        AddTestPython(MDP QLearning)
        AddTestPython(MDP ValueIteration)
        AddTestPython(MDP Policy)
        AddTestPython(MDP MCTS)
    endif()

    add_custom_command(
        TARGET Global_UtilsCoreTests
        POST_BUILD
        COMMAND rsync
        ARGS -r
                "${PROJECT_SOURCE_DIR}/test/data"
                "${CMAKE_CURRENT_BINARY_DIR}"
        COMMENT "Copying data to test folder...")
endif()

if (MAKE_FMDP)
    AddTest(Factored Utils)
    AddTest(Factored BayesianNetwork)
    AddTest(Factored FilterMap)
    AddTest(Factored FactorGraph)

    AddTest(Factored/Bandit Model)
    AddTest(Factored/Bandit FlattenedModel)
    AddTest(Factored/Bandit VariableElimination)
    AddTest(Factored/Bandit MultiObjectiveVariableElimination)
    AddTest(Factored/Bandit UCVE)
    AddTest(Factored/Bandit MaxPlus)
    AddTest(Factored/Bandit LLRPolicy)
    AddTest(Factored/Bandit MAUCEPolicy)

    AddTest(Factored/MDP CooperativeExperience)
    AddTest(Factored/MDP CooperativeMaximumLikelihoodModel)
    AddTest(Factored/MDP CooperativeModel)
    AddTest(Factored/MDP FactoredLP)
    AddTest(Factored/MDP SparseCooperativeQLearning)
    AddTest(Factored/MDP CooperativeQLearning)
    AddTest(Factored/MDP LinearProgramming)
    AddTest(Factored/MDP JointActionLearner)
    AddTest(Factored/MDP CooperativePrioritizedSweeping)

    if (MAKE_PYTHON)
        AddTestPython(Factored JointActionLearner)
    endif()
endif()

if (MAKE_POMDP)
    AddTest(POMDP Types)
    AddTest(POMDP Utils)

    AddTest(POMDP Model)
    AddTest(POMDP SparseModel)

    AddTest(POMDP Policy)

    AddTest(POMDP AMDP)
    AddTest(POMDP BlindStrategies)
    AddTest(POMDP FastInformedBound)
    AddTest(POMDP GapMin)
    AddTest(POMDP SARSOP)
    AddTest(POMDP IncrementalPruning)
    AddTest(POMDP LinearSupport)
    AddTest(POMDP PBVI)
    AddTest(POMDP POMCP)
    AddTest(POMDP RTBSS)
    AddTest(POMDP Witness)
    AddTest(POMDP rPOMCP)

    if (MAKE_PYTHON)
        # Here we have surely passed in the MDP Python test branch, so we know
        # we have everything we need.
        AddTestPython(POMDP IncrementalPruning)
        AddTestPython(POMDP Policy)
        AddTestPython(POMDP GapMin)
    endif()
endif()
