#define BOOST_TEST_MODULE POMDP_POMCP
#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MAIN
#include <boost/test/unit_test.hpp>

#include <AIToolbox/POMDP/Algorithms/IncrementalPruning.hpp>
#include <AIToolbox/POMDP/Algorithms/POMCP.hpp>
#include <AIToolbox/POMDP/Types.hpp>
#include <AIToolbox/POMDP/Policies/Policy.hpp>
#include <AIToolbox/POMDP/Utils.hpp>

#include <AIToolbox/Utils/Probability.hpp>

#include <AIToolbox/POMDP/Environments/TigerProblem.hpp>

BOOST_AUTO_TEST_CASE( discountedHorizon ) {
    using namespace AIToolbox;
    using namespace AIToolbox::POMDP;

    auto model = makeTigerProblem();
    model.setDiscount(0.85);

    // This indicates where the tiger is.
    Matrix2D beliefs(5, 2);
    beliefs << 0.5,     0.5,
               1.0,     0.0,
               0.25,    0.75,
               0.98,    0.02,
               0.33,    0.66;

    unsigned maxHorizon = 7;

    // Compute theoretical solution. Since the tiger problem can be actually
    // solved in multiple ways with certain discounts, I chose a discount factor
    // that seems to work, although this is in no way substantiated with theory.
    // If there's a better way to test POMCP please let me know.
    IncrementalPruning groundTruth(maxHorizon, 0.0);
    auto solution = groundTruth(model);
    auto & vf = std::get<1>(solution);
    Policy p(model.getS(), model.getA(), model.getO(), vf);

    for ( unsigned horizon = 1; horizon <= maxHorizon; ++horizon ) {
        // Again, the exploration constant has been chosen to let the solver agree with
        // the ground truth rather than not. A lower constant results in LISTEN actions
        // being swapped for OPEN actions. This still could be due to the fact that in some
        // cases listening now vs opening later really does not change much.
        // The main problem is that the high exploration constant here is used to force
        // OPEN actions in high uncertainty situations, in any case. Otherwise, LISTEN actions
        // end up being way better, since POMCP averages across actions (not very smart).
        POMCP solver(model, 1000, 10000, horizon * 10000.0);

        for ( auto i = 0; i < beliefs.rows(); ++i ) {
            auto a = solver.sampleAction(beliefs.row(i), horizon);
            auto trueA = p.sampleAction(beliefs.row(i), horizon);

            BOOST_CHECK_EQUAL( std::get<0>(trueA), a);
        }
    }
}

BOOST_AUTO_TEST_CASE( horizonOneBelief ) {
    using namespace AIToolbox;
    using namespace AIToolbox::POMDP;

    auto model = makeTigerProblem();
    model.setDiscount(0.85);

    // This indicates where the tiger is.
    Matrix2D beliefs(5, 2);
    beliefs << 0.5,     0.5,
               1.0,     0.0,
               0.25,    0.75,
               0.98,    0.02,
               0.33,    0.66;

    unsigned horizon = 1;
    unsigned count = 10000;

    POMCP solver(model, 1000, count, 10000.0);

    // We want to check that when there is an horizon of 1
    // the particle belief still gets updated so that it
    // can be used when sampling actions using an action
    // and observation.
    for ( auto i = 0; i < beliefs.rows(); ++i ) {
        auto b = beliefs.row(i);
        solver.sampleAction(b, horizon);

        auto & graph = solver.getGraph();

        unsigned particleCount = 0;
        for ( auto & a : graph.children ) {
            for ( auto & b : a.children ) {
                particleCount += b.second.belief.size();
            }
        }

        BOOST_CHECK_EQUAL( particleCount, count );
    }
}

BOOST_AUTO_TEST_CASE( sampleOneTime ) {
    using namespace AIToolbox;
    using namespace AIToolbox::POMDP;

    auto model = makeTigerProblem();
    model.setDiscount(0.85);

    // This indicates where the tiger is.
    Belief belief(2); belief.fill(0.5);

    unsigned horizon = 100;
    unsigned count = 1;

    POMCP solver(model, 1000, count, 10000.0);

    // We assure POMCP does not crash when pruning a tree
    // and the new head was a leaf (and thus did not have
    // children).

    // UCT here samples action 0, since it's
    // the first in line.
    solver.sampleAction(belief, horizon);

    auto & graph_ = solver.getGraph();
    // We find the leaf we just produced
    auto it = graph_.children[0].children.begin();
    auto o = it->first;

    // We make a,o the new head
    solver.sampleAction( 0, o, horizon-1);
}
