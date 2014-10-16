#define BOOST_TEST_MODULE POMDP_Incremental_Pruning
#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MAIN
#include <boost/test/unit_test.hpp>

#include <AIToolbox/POMDP/Algorithms/AMDP.hpp>
#include <AIToolbox/POMDP/Algorithms/IncrementalPruning.hpp>
#include <AIToolbox/POMDP/Algorithms/Utils/BeliefGenerator.hpp>
#include <AIToolbox/MDP/Algorithms/ValueIteration.hpp>
#include <AIToolbox/POMDP/Policies/Policy.hpp>
#include <AIToolbox/MDP/Policies/QGreedyPolicy.hpp>
#include <AIToolbox/POMDP/Types.hpp>
#include <AIToolbox/MDP/IO.hpp>
#include "TigerProblem.hpp"

BOOST_AUTO_TEST_CASE( discountedHorizon ) {
    using namespace AIToolbox;

    auto model = makeTigerProblem();
    model.setDiscount(0.95);

    unsigned horizon = 4;
    POMDP::IncrementalPruning ipsolver(horizon, 0.0);

    auto truth = ipsolver(model);
    POMDP::Policy truthPolicy(model.getS(), model.getA(), model.getO(), std::get<1>(truth));

    POMDP::AMDP converter(6000, 70);
    auto convertedModel = converter(model);
    auto & simplerModel = std::get<0>(convertedModel);
    auto & beliefConverter = std::get<1>(convertedModel);

    MDP::ValueIteration solver(horizon);

    auto solution = solver(simplerModel);
    MDP::QGreedyPolicy policy(std::get<2>(solution));

    // NOTE: This test is very very fragile, the point is that AMDP is
    // by definition a very approximate method of solving a POMDP, so
    // I'm not exactly sure what would be the best way to consistently
    // test its behaviour. For now, this will have to do. Trying a number
    // of random beliefs works in the majority of cases but for some
    // particular beliefs tends to fail (possibly near the edges of an
    // entropy bucket), so that's not really doable.
    std::vector<POMDP::Belief> beliefs{{0.5, 0.5}, {1.0, 0.0}, {0.25, 0.75}, {0.98, 0.02}, {0.33, 0.66}};

    for ( auto & b : beliefs )
        BOOST_CHECK_EQUAL( truthPolicy.sampleAction(b), policy.sampleAction( beliefConverter(b) ) );
}
