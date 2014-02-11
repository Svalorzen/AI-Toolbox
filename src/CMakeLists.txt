cmake_minimum_required (VERSION 2.6)

find_package(Boost 1.53 REQUIRED)
include_directories(${Boost_INCLUDE_DIRS})

add_library(AIToolbox DynaQ.cpp DynaQInterface.cpp EpsilonPolicy.cpp Experience.cpp MDPUtils.cpp Model.cpp Policy.cpp PolicyInterface.cpp QLearning.cpp QPolicyInterface.cpp QGreedyPolicy.cpp RLModel.cpp ValueIteration.cpp Seeder.cpp) 
