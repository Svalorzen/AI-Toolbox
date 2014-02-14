#define BOOST_TEST_MODULE MDP_RLModel
#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MAIN
#include <boost/test/unit_test.hpp>

#include <AIToolbox/MDP/QLearning.hpp>
#include <AIToolbox/MDP/Utils.hpp>

BOOST_AUTO_TEST_CASE( updates ) {
    AIToolbox::MDP::QLearning solver(5,5);
    {
        // State goes to itself, thus needs to consider
        // next-step value.
        solver.stepUpdateQ(0, 0, 0, 10);
        BOOST_CHECK_EQUAL( solver.getQFunction()[0][0], 5.0 );

        solver.stepUpdateQ(0, 0, 0, 10);
        BOOST_CHECK_EQUAL( solver.getQFunction()[0][0], 9.75 );
    }
    {
        // Here it does not, so improvement is slower.
        solver.stepUpdateQ(3, 4, 0, 10);
        BOOST_CHECK_EQUAL( solver.getQFunction()[3][0], 5.0 );

        solver.stepUpdateQ(3, 4, 0, 10);
        BOOST_CHECK_EQUAL( solver.getQFunction()[3][0], 7.50 );
    }
    {
        // Test that index combinations are right.
        solver.stepUpdateQ(0, 1, 1, 10);
        BOOST_CHECK_EQUAL( solver.getQFunction()[0][1], 5.0  );
        BOOST_CHECK_EQUAL( solver.getQFunction()[1][0], 0.0  );
        BOOST_CHECK_EQUAL( solver.getQFunction()[1][1], 0.0  );
    }
}
