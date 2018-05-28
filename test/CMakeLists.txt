cmake_minimum_required(VERSION 3.9) # CMP0069 NEW

set(BanditDependencies      AIToolboxMDP)
set(MDPDependencies         AIToolboxMDP)
set(POMDPDependencies       AIToolboxMDP AIToolboxPOMDP)
set(FactoredDependencies    AIToolboxMDP AIToolboxFMDP)

function (AddTestGlobal name)
    set(exename Global_${name})
    add_executable(${exename}Tests ${name}Tests.cpp ${PROJECT_SOURCE_DIR}/src/Impl/Seeder.cpp)
    target_link_libraries(${exename}Tests ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY} ${ARGN})
    add_test(NAME ${exename} WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY} COMMAND $<TARGET_FILE:${exename}Tests>)
    set_target_properties(${exename}Tests PROPERTIES INTERPROCEDURAL_OPTIMIZATION ${LTO_SUPPORTED})
endfunction (AddTestGlobal)

function (AddTest type name)
    set(exename ${type}_${name})
    add_executable(${exename}Tests ${type}/${name}Tests.cpp)
    target_link_libraries(${exename}Tests ${${type}Dependencies} ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY} ${ARGN})
    add_test(NAME ${exename} WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY} COMMAND $<TARGET_FILE:${exename}Tests>)
    set_target_properties(${exename}Tests PROPERTIES INTERPROCEDURAL_OPTIMIZATION ${LTO_SUPPORTED})
endfunction (AddTest)

function (AddTestPython type name)
    set(exename ${type}_Python_${name})
    add_test(NAME ${exename} WORKING_DIRECTORY ${CMAKE_LIBRARY_OUTPUT_DIRECTORY} COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/Python/${type}/${name}Tests.py)
endfunction (AddTestPython)

if (MAKE_MDP)
    find_package(Boost 1.53 COMPONENTS unit_test_framework REQUIRED)
    include_directories(SYSTEM ${Boost_INCLUDE_DIRS})

    find_package(Eigen3 3.2.92 REQUIRED)
    include_directories(SYSTEM ${EIGEN3_INCLUDE_DIR})

    AddTestGlobal(Utils)

    AddTest(Bandit GreedyPolicy)
    AddTest(Bandit ThompsonSamplingPolicy)
    AddTest(Bandit LRPPolicy)
    AddTest(Bandit ESRLPolicy)

    AddTest(MDP Types)

    AddTest(MDP Experience)
    AddTest(MDP SparseExperience)
    AddTest(MDP Model)
    AddTest(MDP SparseModel)
    AddTest(MDP RLModel)
    AddTest(MDP SparseRLModel)

    AddTest(MDP QGreedyPolicy)
    AddTest(MDP WoLFPolicy)

    AddTest(MDP PrioritizedSweeping)
    AddTest(MDP QLearning)
    AddTest(MDP HystereticQLearning)
    AddTest(MDP SARSA)
    AddTest(MDP ExpectedSARSA)
    AddTest(MDP SARSAL)
    AddTest(MDP DynaQ)
    AddTest(MDP Dyna2)
    AddTest(MDP PolicyEvaluation)
    AddTest(MDP PolicyIteration)
    AddTest(MDP ValueIteration)
    AddTest(MDP MCTS)

    if (MAKE_PYTHON)
        # Normally one loads PythonInterp first, but since
        # here the interpreter is only optional then libs
        # come first. The version must be the same though.
        find_package(PythonInterp ${PYTHONLIBS_VERSION_STRING_STRIPPED} EXACT)
        if (PYTHONINTERP_FOUND)
            AddTestPython(MDP Experience)
            AddTestPython(MDP QLearning)
            AddTestPython(MDP ValueIteration)
            AddTestPython(MDP PolicyIteration)
        endif()
    endif()

    add_custom_command(
        TARGET Global_UtilsTests
        POST_BUILD
        COMMAND rsync
        ARGS -r
                "${PROJECT_SOURCE_DIR}/test/data"
                "${CMAKE_CURRENT_BINARY_DIR}"
        COMMENT "Copying data to test folder...")
endif()

if (MAKE_FMDP)
    AddTest(Factored Utils)
    AddTest(Factored FactoredContainer)
    AddTest(Factored FactorGraph)

    AddTest(Factored FactoredLP)

    AddTest(Factored LLR)
    AddTest(Factored MultiObjectiveVariableElimination)
    AddTest(Factored SparseCooperativeQLearning)
    AddTest(Factored JointActionLearner)
    AddTest(Factored VariableElimination)

    if (MAKE_PYTHON)
        AddTestPython(Factored JointActionLearner)
    endif()
endif()

if (MAKE_POMDP)
    AddTest(POMDP Types)
    AddTest(POMDP Utils)

    AddTest(POMDP Model)
    AddTest(POMDP SparseModel)

    AddTest(POMDP AMDP)
    AddTest(POMDP IncrementalPruning)
    AddTest(POMDP PBVI)
    AddTest(POMDP POMCP)
    AddTest(POMDP BlindStrategies)
    AddTest(POMDP FastInformedBound)
    AddTest(POMDP rPOMCP)
    AddTest(POMDP RTBSS)
    AddTest(POMDP Witness)
    AddTest(POMDP GapMin)

    if (MAKE_PYTHON)
        # Here we have surely passed in the MDP Python test branch, so we know
        # we have everything we need.
        AddTestPython(POMDP IncrementalPruning)
        AddTestPython(POMDP GapMin)
    endif()
endif()
