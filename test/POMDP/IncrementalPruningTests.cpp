#define BOOST_TEST_MODULE POMDP_Incremental_Pruning
#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MAIN
#include <boost/test/unit_test.hpp>

#include <AIToolbox/POMDP/Algorithms/IncrementalPruning.hpp>
#include <AIToolbox/POMDP/Model.hpp>
#include <AIToolbox/MDP/Experience.hpp>
#include <AIToolbox/MDP/Model.hpp>
#include <AIToolbox/MDP/RLModel.hpp>

#include <AIToolbox/MDP/IO.hpp>

#include <array>
#include <algorithm>
#include <fstream>
#include <cstdio>

#include <iostream>
#include <fstream>

BOOST_AUTO_TEST_CASE( construction ) {
    using namespace AIToolbox;

    // Actions are: open-left, listen, open-right
    size_t S = 2, A = 3, O = 2;

    POMDP::Model<MDP::Model> model(O, S, A);

    AIToolbox::Table3D transitions(boost::extents[S][A][S]);
    AIToolbox::Table3D rewards(boost::extents[S][A][S]);
    AIToolbox::Table3D observations(boost::extents[S][A][O]);

    // Transitions
    for ( size_t s = 0; s < S; ++s )
        transitions[s][1][s] = 1.0;

    for ( size_t s = 0; s < S; ++s ) {
        for ( size_t s1 = 0; s1 < S; ++s1 ) {
            transitions[s][0][s1] = 1.0 / S;
            transitions[s][2][s1] = 1.0 / S;
        }
    }

    // Observations
    observations[0][1][0] = 0.85;
    observations[0][1][1] = 0.15;

    observations[1][1][1] = 0.85;
    observations[1][1][0] = 0.15;

    for ( size_t s = 0; s < S; ++s ) {
        for ( size_t o = 0; o < O; ++o ) {
            observations[s][0][o] = 1.0 / O;
            observations[s][2][o] = 1.0 / O;
        }
    }

    // Rewards
    for ( size_t s = 0; s < S; ++s )
        for ( size_t s1 = 0; s1 < S; ++s1 )
            rewards[s][1][s1] = -1.0;

    for ( size_t s1 = 0; s1 < S; ++s1 ) {
        rewards[1][0][s1] = 10.0;
        rewards[0][0][s1] = -100.0;

        rewards[0][2][s1] = 10.0;
        rewards[1][2][s1] = -100.0;
    }

    model.setTransitionFunction(transitions);
    model.setRewardFunction(rewards);
    model.setObservationFunction(observations);

    POMDP::IncrementalPruning solver;
    solver(model, 3);
}
