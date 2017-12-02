cmake_minimum_required (VERSION 3.0)

function (AddTestGlobal name)
    set(exename Global_${name})
    add_executable(${exename}Tests ${name}Tests.cpp ${PROJECT_SOURCE_DIR}/src/Impl/Seeder.cpp)
    target_link_libraries(${exename}Tests ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY} ${ARGN})
    add_test(NAME ${exename} WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY} COMMAND $<TARGET_FILE:${exename}Tests>)
endfunction (AddTestGlobal)

function (AddTestGame name)
    set(exename Game_${name})
    add_executable(${exename}Tests Game/${name}Tests.cpp)
    target_link_libraries(${exename}Tests AIToolboxMDP ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY} ${ARGN})
    add_test(NAME ${exename} WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY} COMMAND $<TARGET_FILE:${exename}Tests>)
endfunction (AddTestGame)

function (AddTestMDP name)
    set(exename MDP_${name})
    add_executable(${exename}Tests MDP/${name}Tests.cpp)
    target_link_libraries(${exename}Tests AIToolboxMDP ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY} ${ARGN})
    add_test(NAME ${exename} WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY} COMMAND $<TARGET_FILE:${exename}Tests>)
endfunction (AddTestMDP)

function (AddTestFactoredMDP name)
    set(exename FactoredMDP_${name})
    add_executable(${exename}Tests FactoredMDP/${name}Tests.cpp)
    target_link_libraries(${exename}Tests AIToolboxMDP AIToolboxFMDP ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY} ${ARGN})
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

    AddTestGame(LRPPolicy)
    AddTestGame(ESRLPolicy)

    AddTestMDP(Types)

    AddTestMDP(Experience)
    AddTestMDP(SparseExperience)
    AddTestMDP(Model)
    AddTestMDP(SparseModel)
    AddTestMDP(RLModel)
    AddTestMDP(SparseRLModel)

    AddTestMDP(QGreedyPolicy)
    AddTestMDP(WoLFPolicy)

    AddTestMDP(PrioritizedSweeping)
    AddTestMDP(QLearning)
    AddTestMDP(HystereticQLearning)
    AddTestMDP(SARSA)
    AddTestMDP(ExpectedSARSA)
    AddTestMDP(SARSAL)
    AddTestMDP(DynaQ)
    AddTestMDP(Dyna2)
    AddTestMDP(PolicyEvaluation)
    AddTestMDP(PolicyIteration)
    AddTestMDP(ValueIteration)
    AddTestMDP(MCTS)

    if (MAKE_PYTHON)
        # Normally one loads PythonInterp first, but since
        # here the interpreter is only optional then libs
        # come first. The version must be the same though.
        find_package(PythonInterp ${PYTHONLIBS_VERSION_STRING_STRIPPED} EXACT)
        if (PYTHONINTERP_FOUND)
            AddTestMDPPython(Experience)
            AddTestMDPPython(QLearning)
            AddTestMDPPython(ValueIteration)
            AddTestMDPPython(PolicyIteration)
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
    AddTestFactoredMDP(Utils)
    AddTestFactoredMDP(FactoredContainer)
    AddTestFactoredMDP(FactorGraph)

    AddTestFactoredMDP(FactoredLP)

    AddTestFactoredMDP(LLR)
    AddTestFactoredMDP(MultiObjectiveVariableElimination)
    AddTestFactoredMDP(SparseCooperativeQLearning)
    AddTestFactoredMDP(VariableElimination)
endif()

if (MAKE_POMDP)
    AddTestPOMDP(Types)
    AddTestPOMDP(Utils)

    AddTestPOMDP(Model)
    AddTestPOMDP(SparseModel)

    AddTestPOMDP(AMDP)
    AddTestPOMDP(IncrementalPruning)
    AddTestPOMDP(PBVI)
    AddTestPOMDP(POMCP)
    AddTestPOMDP(BlindStrategies)
    AddTestPOMDP(rPOMCP)
    AddTestPOMDP(RTBSS)
    AddTestPOMDP(Witness)

    if (MAKE_PYTHON)
        # Here we have surely passed in the MDP Python test branch, so we know
        # we have everything we need.
        AddTestPOMDPPython(IncrementalPruning)
    endif()
endif()
