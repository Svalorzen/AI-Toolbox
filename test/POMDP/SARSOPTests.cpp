#define BOOST_TEST_MODULE POMDP_SARSOP
#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MAIN
#include <boost/test/unit_test.hpp>

#include <AIToolbox/POMDP/Algorithms/SARSOP.hpp>
#include <AIToolbox/POMDP/Types.hpp>

#include <AIToolbox/POMDP/Environments/ChengD35.hpp>

BOOST_AUTO_TEST_CASE( discountedHorizon ) {
    using namespace AIToolbox::POMDP;

    SARSOP sarsop(34);
    const auto model = makeChengD35();

    Belief initialBelief(model.getS());
    initialBelief.fill(1.0 / model.getS());

    const auto [lb, ub, vlist, qfun] = sarsop(model, initialBelief);

    // This are the bounds from the gapmin paper.
    BOOST_CHECK(8705 < ub && ub < 8707);
    BOOST_CHECK(8672 < lb && lb < 8674);
    (void)vlist;
    (void)qfun;
}
