#define BOOST_TEST_MODULE POMDP_RTBSS
#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MAIN
#include <boost/test/unit_test.hpp>

#include <AIToolbox/POMDP/Algorithms/IncrementalPruning.hpp>
#include <AIToolbox/POMDP/Algorithms/RTBSS.hpp>
#include <AIToolbox/POMDP/Types.hpp>
#include "TigerProblem.hpp"

BOOST_AUTO_TEST_CASE( discountedHorizon ) {
    using namespace AIToolbox;

    auto model = makeTigerProblem();
    model.setDiscount(0.85);

    // This indicates where the tiger is.
    std::vector<POMDP::Belief> beliefs{{0.5, 0.5}, {1.0, 0.0}, {0.25, 0.75}, {0.98, 0.02}, {0.33, 0.66}};

    unsigned maxHorizon = 7;

    // Compute theoretical solution. Since the tiger problem can be actually
    // solved in multiple ways with certain discounts, I chose a discount factor
    // that seems to work, although this is in no way substantiated with theory.
    // If there's a better way to test POMCP please let me know.
    POMDP::IncrementalPruning groundTruth(maxHorizon, 0.0);
    auto solution = groundTruth(model);
    auto & vf = std::get<1>(solution);

    for ( unsigned horizon = 1; horizon <= maxHorizon; ++horizon ) {
        POMDP::RTBSS<decltype(model)> solver(model, 10.0);

        for ( auto & b : beliefs ) {
            auto a = solver.sampleAction(b, horizon);

            // We avoid using a policy so that we can also check that computed internal
            // values are correct.
            auto & vlist = vf[horizon];

            auto begin     = std::begin(vlist);
            auto bestMatch = POMDP::findBestAtBelief(std::begin(b), std::end(b), begin, std::end(vlist));

            double trueValue = std::inner_product(std::begin(b), std::end(b), std::begin(std::get<POMDP::VALUES>(*bestMatch)), 0.0);
            double trueAction = std::get<POMDP::ACTION>(*bestMatch);

            BOOST_CHECK_EQUAL(trueAction, std::get<0>(a));

            // Unfortunately it does seem that the two methods give slightly different
            // value results (they seem to be equal to around 12 digits of precision,
            // but no more). So we compare them via floats, sacrificing some precision
            // but at least checking that they are somewhat the same.
            BOOST_CHECK_EQUAL((float)trueValue, (float)std::get<1>(a));
        }
    }
}
