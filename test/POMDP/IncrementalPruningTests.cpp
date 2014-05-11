#define BOOST_TEST_MODULE POMDP_Incremental_Pruning
#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MAIN
#include <boost/test/unit_test.hpp>

#include <AIToolbox/POMDP/Algorithms/IncrementalPruning.hpp>
#include <AIToolbox/POMDP/Model.hpp>
#include <AIToolbox/MDP/Model.hpp>
#include <AIToolbox/MDP/Algorithms/ValueIteration.hpp>

#include <array>
#include <algorithm>
#include <fstream>
#include <cstdio>

#include <iostream>

BOOST_AUTO_TEST_CASE( construction ) {
    using namespace AIToolbox;
    POMDP::Model<MDP::Model> model(5,4,3);
    MDP::Model test(4,3);

    MDP::ValueIteration vi;
    vi(model);
    vi(test);
}
