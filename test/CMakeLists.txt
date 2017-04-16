cmake_minimum_required (VERSION 3.0)

function (AddTestGlobal name)
    set(exename Global_${name})
    add_executable(${exename}Tests ${name}Tests.cpp)
    target_link_libraries(${exename}Tests ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY} ${ARGN})
    add_test(NAME ${exename} WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY} COMMAND $<TARGET_FILE:${exename}Tests>)
endfunction (AddTestGlobal)

function (AddTestMDP name)
    set(exename MDP_${name})
    add_executable(${exename}Tests MDP/${name}Tests.cpp)
    target_link_libraries(${exename}Tests AIToolboxMDP ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY} ${ARGN})
    add_test(NAME ${exename} WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY} COMMAND $<TARGET_FILE:${exename}Tests>)
endfunction (AddTestMDP)

function (AddTestFactoredMDP name)
    set(exename FactoredMDP_${name})
    add_executable(${exename}Tests FactoredMDP/${name}Tests.cpp)
    target_link_libraries(${exename}Tests AIToolboxMDP ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY} ${ARGN})
    add_test(NAME ${exename} WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY} COMMAND $<TARGET_FILE:${exename}Tests>)
endfunction (AddTestFactoredMDP)

function (AddTestMDPPython name)
    set(exename MDP_Python_${name})
    add_test(NAME ${exename} WORKING_DIRECTORY ${CMAKE_LIBRARY_OUTPUT_DIRECTORY} COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/Python/MDP/${name}Tests.py)
endfunction (AddTestMDPPython)

function (AddTestPOMDPPython name)
    set(exename POMDP_Python_${name})
    add_test(NAME ${exename} WORKING_DIRECTORY ${CMAKE_LIBRARY_OUTPUT_DIRECTORY} COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/Python/POMDP/${name}Tests.py)
endfunction (AddTestPOMDPPython)

function (AddTestPOMDP name)
    set(exename POMDP_${name})
    add_executable(${exename}Tests POMDP/${name}Tests.cpp)
    target_link_libraries(${exename}Tests AIToolboxMDP AIToolboxPOMDP ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY} ${ARGN})
    add_test(NAME ${exename} WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY} COMMAND $<TARGET_FILE:${exename}Tests>)
endfunction (AddTestPOMDP)

if (MAKE_MDP)
    find_package(Boost 1.53 COMPONENTS unit_test_framework REQUIRED)
    include_directories(${Boost_INCLUDE_DIRS})

    find_package(Eigen3 REQUIRED)
    include_directories(${EIGEN3_INCLUDE_DIR})

    AddTestGlobal(Utils)

    AddTestMDP(Types)
    AddTestMDP(Experience)
    AddTestMDP(MCTS)
    AddTestMDP(Model)
    AddTestMDP(PrioritizedSweeping)
    AddTestMDP(QLearning)
    AddTestMDP(RLModel)
    AddTestMDP(SARSA)
    AddTestMDP(SparseExperience)
    AddTestMDP(SparseModel)
    AddTestMDP(SparseRLModel)
    AddTestMDP(PolicyEvaluation)
    AddTestMDP(ValueIteration)
    AddTestMDP(WoLFPolicy)
    AddTestMDP(QGreedyPolicy)

    AddTestFactoredMDP(Utils)
    AddTestFactoredMDP(FactoredContainer)
    AddTestFactoredMDP(FactorGraph)
    AddTestFactoredMDP(VariableElimination)
    AddTestFactoredMDP(MultiObjectiveVariableElimination)
    AddTestFactoredMDP(SparseCooperativeQLearning)

    if (MAKE_PYTHON)
        # Normally one loads PythonInterp first, but since
        # here the interpreter is only optional then libs
        # come first. The version must be the same though.
        find_package(PythonInterp ${PYTHONLIBS_VERSION_STRING_STRIPPED} EXACT)
        if (PYTHONINTERP_FOUND)
            AddTestMDPPython(Experience)
            AddTestMDPPython(QLearning)
            AddTestMDPPython(ValueIteration)
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

if (MAKE_POMDP)
    AddTestPOMDP(Types)
    AddTestPOMDP(AMDP)
    AddTestPOMDP(IncrementalPruning)
    AddTestPOMDP(Model)
    AddTestPOMDP(PBVI)
    AddTestPOMDP(POMCP)
    AddTestPOMDP(RTBSS)
    AddTestPOMDP(SparseModel)
    AddTestPOMDP(Witness)

    if (MAKE_PYTHON)
        # Here we have surely passed in the MDP Python test branch, so we know
        # we have everything we need.
        AddTestPOMDPPython(IncrementalPruning)
    endif()
endif()
