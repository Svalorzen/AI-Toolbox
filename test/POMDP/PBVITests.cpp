#define BOOST_TEST_MODULE POMDP_PBVI
#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MAIN
#include <boost/test/unit_test.hpp>

#include <AIToolbox/POMDP/Algorithms/PBVI.hpp>
#include <AIToolbox/POMDP/Algorithms/IncrementalPruning.hpp>
#include <AIToolbox/POMDP/Types.hpp>
#include <AIToolbox/Utils/Core.hpp>

#include <AIToolbox/POMDP/Environments/TigerProblem.hpp>

BOOST_AUTO_TEST_CASE( discountedHorizon ) {
    using namespace AIToolbox::POMDP;

    auto model = makeTigerProblem();
    model.setDiscount(0.95);

    // For higher horizons PBVI may not find all the possible solutions, but
    // generally gets close. The solution also depends on which beliefs were
    // randomly sampled.
    unsigned horizon = 5;
    PBVI solver(2000, horizon, 0.01);
    auto solution = solver(model);

    // Yeah not really truth, but as long as the
    // IP tests all pass I guess it's truth enough.
    IncrementalPruning ipsolver(horizon, 0.0);
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
            BOOST_CHECK(vf[i][j].values == vt[i][j].values);
            BOOST_CHECK(vf[i][j].action == vt[i][j].action);
            // Obs we can't check since we shuffle, they won't necessarily
            // be the same.
        }
    }
}

BOOST_AUTO_TEST_CASE( undiscountedHorizon ) {
    using namespace AIToolbox;
    using namespace AIToolbox::POMDP;
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
    PBVI solver(1000, horizon, 0.01);
    auto solution = solver(model);

    auto & vf = std::get<1>(solution);
    auto vlist = vf[horizon];

    // This is the correct solution
    VList truth = {
        // Action 10 here (which does not exist) is used to mark the values for which both listening or acting is a correct
        // action. We will not test those.
        {(MDP::Values(2) << -101.0000000000000000000000000, 9.0000000000000000000000000   ).finished(), 10u, VObs(0)},
        {(MDP::Values(2) << -16.8500000000000014210854715 , 7.3499999999999996447286321   ).finished(), 0u,  VObs(0)},
        {(MDP::Values(2) << -2.0000000000000000000000000  , -2.0000000000000000000000000  ).finished(), 0u,  VObs(0)},
        {(MDP::Values(2) << 7.3499999999999996447286321   , -16.8500000000000014210854715 ).finished(), 0u,  VObs(0)},
        {(MDP::Values(2) << 9.0000000000000000000000000   , -101.0000000000000000000000000).finished(), 10u, VObs(0)},
    };

    // We check that all entries PBVI found exist in the ground truth.
    for ( size_t i = 0; i < vlist.size(); ++i ) {
        auto & values = vlist[i].values;
        auto it = std::find_if(std::begin(truth), std::end(truth), [&](const VEntry & ve) {
            return ve.values == values;
        });
        BOOST_CHECK( it != std::end(truth) );

        if ( it->action == 0u )
            BOOST_CHECK_EQUAL(vlist[i].action, it->action);
    }
}
