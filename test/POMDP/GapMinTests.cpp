#define BOOST_TEST_MODULE POMDP_GapMin
#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MAIN
#include <boost/test/unit_test.hpp>

#include <AIToolbox/POMDP/Algorithms/GapMin.hpp>
#include <AIToolbox/POMDP/Types.hpp>

#include "Utils/TigerProblem.hpp"

BOOST_AUTO_TEST_CASE( discountedHorizon ) {
    using namespace AIToolbox::POMDP;
    GapMin gm;

    auto model = makeTigerProblem();
    Belief initialBelief(model.getS());
    initialBelief.fill(1.0 / model.getS());

    auto solution = gm(model, initialBelief);
}
