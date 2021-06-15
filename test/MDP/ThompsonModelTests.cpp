#define BOOST_TEST_MODULE MDP_ThompsonModel
#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MAIN
#include <boost/test/unit_test.hpp>

#include <AIToolbox/Utils/Core.hpp>
#include <AIToolbox/MDP/Experience.hpp>
#include <AIToolbox/MDP/ThompsonModel.hpp>

// #include <AIToolbox/MDP/IO.hpp>
// #include <fstream>

BOOST_AUTO_TEST_CASE( eigen_model ) {
    BOOST_CHECK(AIToolbox::MDP::IsModelEigen<AIToolbox::MDP::ThompsonModel<AIToolbox::MDP::Experience>>);
}

BOOST_AUTO_TEST_CASE( construction ) {
    using namespace AIToolbox::MDP;
    const size_t S = 10, A = 8;

    Experience exp(S,A);
    ThompsonModel model(exp, 1.0);

    for ( size_t s = 0; s < S; ++s ) {
        for ( size_t a = 0; a < A; ++a ) {
            double sum = 0.0;
            for ( size_t s1 = 0; s1 < S; ++s1 ) {
                BOOST_CHECK( 0.0 <= model.getTransitionProbability(s,a,s1));
                BOOST_CHECK( model.getTransitionProbability(s,a,s1) <= 1.0);
                sum += model.getTransitionProbability(s,a,s1);

                BOOST_CHECK_EQUAL( model.getExpectedReward(s,a,s1), 0.0 );
            }
            BOOST_CHECK( AIToolbox::checkEqualSmall(sum, 1.0) );
        }
    }
}

BOOST_AUTO_TEST_CASE( syncing ) {
    using namespace AIToolbox::MDP;
    const size_t S = 10, A = 8;

    Experience exp(S,A);
    // Single state sync
    {
        ThompsonModel model(exp, 1.0);

        exp.record(0,0,1,10);
        exp.record(0,0,2,10);
        exp.record(0,0,3,4);

        exp.record(4,1,5,10);

        // Cache T and R before sync to verify we are only changing the things
        // we want.
        auto oldT = model.getTransitionFunction();
        auto oldR = model.getRewardFunction();

        model.sync(0,0);

        // Only the row for 0, 0 should have changed.
        BOOST_CHECK_EQUAL( model.getTransitionFunction()[0].bottomRows(S-1), oldT[0].bottomRows(S-1) );
        BOOST_CHECK_EQUAL( model.getTransitionFunction()[1], oldT[1] );
        // I mean in theory these could be equal but the probability that they
        // are is vanishingly small. We only test that one value has changed
        // rather than the whole row; not sure what would be the best way to
        // check that without false positives.
        BOOST_CHECK( AIToolbox::checkDifferentSmall(model.getTransitionProbability(0, 0, 0), oldT[0](0, 0)) );

        BOOST_CHECK_EQUAL( model.getRewardFunction().row(0).tail(A-1), oldR.row(0).tail(A-1) );
        BOOST_CHECK_EQUAL( model.getRewardFunction().bottomRows(S-1), oldR.bottomRows(S-1) );
        // This sync uses the student_t sampling, check that nothing super-weird has happened.
        BOOST_CHECK( !model.getRewardFunction().hasNaN() );

        oldT = model.getTransitionFunction();
        oldR = model.getRewardFunction();

        model.sync(4,1);

        // Only the row for 4, 0 should have changed.
        BOOST_CHECK_EQUAL( model.getTransitionFunction()[0], oldT[0] );
        BOOST_CHECK_EQUAL( model.getTransitionFunction()[1].topRows(4), oldT[1].topRows(4) );
        BOOST_CHECK_EQUAL( model.getTransitionFunction()[1].bottomRows(S-5), oldT[1].bottomRows(S-5) );
        // I mean in theory these could be equal but the probability that they
        // are is vanishingly small. We only test that one value has changed
        // rather than the whole row; not sure what would be the best way to
        // check that without false positives.
        BOOST_CHECK( AIToolbox::checkDifferentSmall(model.getTransitionProbability(4, 1, 0), oldT[1](4, 0)) );

        BOOST_CHECK_EQUAL( model.getRewardFunction().row(4)[0], oldR.row(4)[0] );
        BOOST_CHECK_EQUAL( model.getRewardFunction().row(4).tail(A-2), oldR.row(4).tail(A-2) );
        BOOST_CHECK_EQUAL( model.getRewardFunction().topRows(4), oldR.topRows(4) );
        BOOST_CHECK_EQUAL( model.getRewardFunction().bottomRows(S-5), oldR.bottomRows(S-5) );
    }
}
