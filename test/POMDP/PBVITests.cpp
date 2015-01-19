#define BOOST_TEST_MODULE POMDP_PBVI
#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MAIN
#include <boost/test/unit_test.hpp>

#include <AIToolbox/POMDP/Algorithms/PBVI.hpp>
#include <AIToolbox/POMDP/Algorithms/IncrementalPruning.hpp>
#include <AIToolbox/POMDP/Types.hpp>
#include <AIToolbox/Utils.hpp>
#include "TigerProblem.hpp"

BOOST_AUTO_TEST_CASE( discountedHorizon ) {
    using namespace AIToolbox;

    auto model = makeTigerProblem();
    model.setDiscount(0.95);

    // For higher horizons PBVI may not find all the possible solutions, but
    // generally gets close. The solution also depends on which beliefs were
    // randomly sampled.
    unsigned horizon = 5;
    POMDP::PBVI solver(1000, horizon, 0.01);
    auto solution = solver(model);

    // Yeah not really truth, but as long as the
    // IP tests all pass I guess it's truth enough.
    POMDP::IncrementalPruning ipsolver(horizon, 0.0);
    auto truth = ipsolver(model);

    auto vf = std::get<1>(solution);
    auto vt = std::get<1>(truth);

    for ( auto & vl : vt ) std::sort(std::begin(vl), std::end(vl));
    for ( auto & vl : vf ) std::sort(std::begin(vl), std::end(vl));

    bool sizeEqual1, sizeEqual2;
    sizeEqual1 = vf.size() == vt.size();

    BOOST_CHECK(sizeEqual1);
    if ( !sizeEqual1 ) return;
    for ( size_t i = 0; i < vf.size(); ++i ) {
        sizeEqual2 = vf[i].size() == vt[i].size();
        BOOST_CHECK(sizeEqual2);
        if ( !sizeEqual2 ) continue;
        for ( size_t j = 0; j < vf[i].size(); ++j ) {
            BOOST_CHECK(std::get<POMDP::VALUES>(vf[i][j]) == std::get<POMDP::VALUES>(vt[i][j]));
            BOOST_CHECK(std::get<POMDP::ACTION>(vf[i][j]) == std::get<POMDP::ACTION>(vt[i][j]));
            // Obs we can't check since we shuffle, they won't necessarily
            // be the same.
        }
    }
}

BOOST_AUTO_TEST_CASE( undiscountedHorizon ) {
    using namespace AIToolbox;
    // NOTE: This test has been added since I noticed that the action results
    // for the undiscounted tiger problem for an horizon of 2 gave me different
    // results from both Cassandra's code and what is published in the literature.
    // In particular, there is a single ValueFunction which suggests to act, while
    // in the literature usually in this step all ValueFunctions point to the
    // listening action. This alternative solution is actually correct, as in an
    // undiscounted scenario it doesn't matter, if the belief in a state is high
    // enough, whether we act now and listen later, or vice-versa.

    auto model = makeTigerProblem();
    model.setDiscount(1.0);

    unsigned horizon = 2;
    POMDP::PBVI solver(1000, horizon, 0.01);
    auto solution = solver(model);

    BOOST_CHECK_EQUAL(std::get<0>(solution), true);

    auto & vf = std::get<1>(solution);
    auto vlist = vf[horizon];

    // This is the correct solution
    POMDP::VList truth = {
        // Action 10 here (which does not exist) is used to mark the values for which both listening or acting is a correct
        // action. We will not test those.
        std::make_tuple(MDP::Values({-101.0000000000000000000000000, 9.0000000000000000000000000   }), 10u, POMDP::VObs(0)),
        std::make_tuple(MDP::Values({-16.8500000000000014210854715 , 7.3499999999999996447286321   }), 0u, POMDP::VObs(0)),
        std::make_tuple(MDP::Values({-2.0000000000000000000000000  , -2.0000000000000000000000000  }), 0u, POMDP::VObs(0)),
        std::make_tuple(MDP::Values({7.3499999999999996447286321   , -16.8500000000000014210854715 }), 0u, POMDP::VObs(0)),
        std::make_tuple(MDP::Values({9.0000000000000000000000000   , -101.0000000000000000000000000}), 10u, POMDP::VObs(0)),
    };

    // We check that all entries PBVI found exist in the ground truth.
    for ( size_t i = 0; i < vlist.size(); ++i ) {
        auto & values = std::get<POMDP::VALUES>(vlist[i]);
        auto it = std::find_if(std::begin(truth), std::end(truth), [&](const POMDP::VEntry & ve) {
                return std::get<POMDP::VALUES>(ve) == values;
                });
        BOOST_CHECK( it != std::end(truth) );

        if ( std::get<POMDP::ACTION>(*it) == 0u )
            BOOST_CHECK_EQUAL(std::get<POMDP::ACTION>(vlist[i]), std::get<POMDP::ACTION>(*it));
    }
}
