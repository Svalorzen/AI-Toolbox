#define BOOST_TEST_MODULE POMDP_Incremental_Pruning
#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MAIN
#include <boost/test/unit_test.hpp>

#include <AIToolbox/POMDP/Algorithms/LinearSupport.hpp>
#include <AIToolbox/POMDP/Types.hpp>
#include <AIToolbox/POMDP/SparseModel.hpp>
#include <AIToolbox/MDP/SparseModel.hpp>
#include <AIToolbox/Utils/Core.hpp>

#include "Utils/TigerProblem.hpp"

BOOST_AUTO_TEST_CASE( optimistic_value_discovery ) {
    using namespace AIToolbox;

    std::vector<std::pair<Vector, double>> beliefs = {
        {(Vector(3) << 1.0, 0.0, 0.0).finished(), 10.0},
        {(Vector(3) << 0.0, 1.0, 0.0).finished(), 5.0},
        {(Vector(3) << 0.0, 0.0, 1.0).finished(), -10.0},
    };

    Vector b(3);
    b.fill(1.0/3.0);

    constexpr double solution = (10.0 + 5.0 - 10.0) / 3.0;

    const auto v = POMDP::computeOptimisticValue(b, std::begin(beliefs), std::end(beliefs));

    BOOST_CHECK(checkEqualGeneral(v, solution));
}

BOOST_AUTO_TEST_CASE( undiscountedHorizon ) {
    //using namespace AIToolbox;
    //// NOTE: This test has been added since I noticed that the action results
    //// for the undiscounted tiger problem for an horizon of 2 gave me different
    //// results from both Cassandra's code and what is published in the literature.
    //// In particular, there is a single ValueFunction which suggests to act, while
    //// in the literature usually in this step all ValueFunctions point to the
    //// listening action. This alternative solution is actually correct, as in an
    //// undiscounted scenario it doesn't matter, if the belief in a state is high
    //// enough, whether we act now and listen later, or vice-versa.

    //auto model = makeTigerProblem();
    //model.setDiscount(1.0);

    //constexpr unsigned horizon = 2;
    //POMDP::LinearSupport solver(horizon, 0.0);
    //const auto solution = solver(model);

    //const auto & vf = std::get<1>(solution);
    //auto vlist = vf[horizon];

    //// This is the correct solution
    //POMDP::VList truth = {
    //    // Action 10 here (which does not exist) is used to mark the values for which both listening or acting is a correct
    //    // action. We will not test those.
    //    {(MDP::Values(2) << -101.0000000000000000000000000, 9.0000000000000000000000000   ).finished(), 10u, POMDP::VObs(0)},
    //    {(MDP::Values(2) << -16.8500000000000014210854715 , 7.3499999999999996447286321   ).finished(), 0u,  POMDP::VObs(0)},
    //    {(MDP::Values(2) << -2.0000000000000000000000000  , -2.0000000000000000000000000  ).finished(), 0u,  POMDP::VObs(0)},
    //    {(MDP::Values(2) << 7.3499999999999996447286321   , -16.8500000000000014210854715 ).finished(), 0u,  POMDP::VObs(0)},
    //    {(MDP::Values(2) << 9.0000000000000000000000000   , -101.0000000000000000000000000).finished(), 10u, POMDP::VObs(0)},
    //};

    //const auto comparer = [](const POMDP::VEntry & lhs, const POMDP::VEntry & rhs) {
    //    return POMDP::operator<(lhs, rhs);
    //};

    //// Make sure we can actually compare them
    //std::sort(std::begin(vlist), std::end(vlist), comparer);
    //std::sort(std::begin(truth), std::end(truth), comparer);

    //BOOST_CHECK_EQUAL(vlist.size(), truth.size());
    //// We check each entry by itself to avoid checking observations
    //for ( size_t i = 0; i < vlist.size(); ++i ) {
    //    // Avoid checking actions with multiple possible answers.
    //    if ( truth[i].action != 10u )
    //        BOOST_CHECK_EQUAL(vlist[i].action, truth[i].action);

    //    const auto & values      = vlist[i].values;
    //    const auto & truthValues = truth[i].values;
    //    BOOST_CHECK_EQUAL(values, truthValues);
    //}
}
