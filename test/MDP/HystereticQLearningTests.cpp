#define BOOST_TEST_MODULE MDP_HystereticQLearning
#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MAIN
#include <boost/test/unit_test.hpp>

#include <AIToolbox/Utils/Core.hpp>
#include <AIToolbox/MDP/Algorithms/HystereticQLearning.hpp>

BOOST_AUTO_TEST_CASE( updates ) {
    namespace mdp = AIToolbox::MDP;

    mdp::HystereticQLearning solver(6, 6, 0.9, 0.5, 0.3);

    BOOST_CHECK_EQUAL( solver.getPositiveLearningRate(), 0.5 );
    BOOST_CHECK_EQUAL( solver.getNegativeLearningRate(), 0.3 );
    {
        // State goes to itself, thus needs to consider
        // next-step value.
        solver.stepUpdateQ(0, 0, 0, 10);
        BOOST_CHECK_EQUAL( solver.getQFunction()(0, 0), 5.0 );

        solver.stepUpdateQ(0, 0, 0, 10);
        BOOST_CHECK_EQUAL( solver.getQFunction()(0, 0), 9.75 );

        // Here we go to zero to have something to compute after - using 2 as
        // s' would still get us zero since we do the max.
        solver.stepUpdateQ(2, 0, 0, -10);
        BOOST_CHECK( AIToolbox::checkEqualGeneral(solver.getQFunction()(2, 0), -0.3675 ) );

        solver.stepUpdateQ(2, 0, 0, -10);
        BOOST_CHECK( AIToolbox::checkEqualGeneral(solver.getQFunction()(2, 0), -0.62475 ) );
    }
    {
        // Here it does not, so improvement is slower.
        solver.stepUpdateQ(3, 0, 4, 10);
        BOOST_CHECK_EQUAL( solver.getQFunction()(3, 0), 5.0 );

        solver.stepUpdateQ(3, 0, 4, 10);
        BOOST_CHECK_EQUAL( solver.getQFunction()(3, 0), 7.50 );

        solver.stepUpdateQ(4, 0, 5, -10);
        BOOST_CHECK_EQUAL( solver.getQFunction()(4, 0), -3.0 );

        solver.stepUpdateQ(4, 0, 5, -10);
        BOOST_CHECK_EQUAL( solver.getQFunction()(4, 0), -5.1 );
    }
    {
        // Test that index combinations are right.
        solver.stepUpdateQ(0, 1, 1, 10);
        BOOST_CHECK_EQUAL( solver.getQFunction()(0, 1), 5.0  );
        BOOST_CHECK_EQUAL( solver.getQFunction()(1, 0), 0.0  );
        BOOST_CHECK_EQUAL( solver.getQFunction()(1, 1), 0.0  );
    }
}
