#define BOOST_TEST_MODULE POMDP_GapMin
#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MAIN
#include <boost/test/unit_test.hpp>

#include <AIToolbox/POMDP/Algorithms/GapMin.hpp>
#include <AIToolbox/POMDP/Types.hpp>

#include <AIToolbox/POMDP/Environments/ChengD35.hpp>

BOOST_AUTO_TEST_CASE( discountedHorizon ) {
    using namespace AIToolbox::POMDP;

    GapMin gm(0.005, 3);

    auto model = makeChengD35();

    Belief initialBelief(model.getS());
    initialBelief.fill(1.0 / model.getS());

    const auto [lb, ub, vlist, qfun] = gm(model, initialBelief);

    BOOST_CHECK(9.0 < ub - lb && ub - lb < 11.0);
    (void)vlist;
    (void)qfun;
}
