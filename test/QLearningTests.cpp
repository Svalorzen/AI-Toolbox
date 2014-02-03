#define BOOST_TEST_MODULE MDP_RLModel
#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MAIN
#include <boost/test/unit_test.hpp>

#include <AIToolbox/MDP/QLearning.hpp>
#include <AIToolbox/MDP/Utils.hpp>

BOOST_AUTO_TEST_CASE( updates ) {
    AIToolbox::MDP::QFunction qfun = AIToolbox::MDP::makeQFunction(5, 5);
    
    AIToolbox::MDP::QLearning solver;

    solver.stepUpdateQ(0, 0, 0, 10, &qfun);
    BOOST_CHECK_EQUAL( qfun[0][0], 5.0 );

    solver.stepUpdateQ(0, 0, 0, 10, &qfun);
    BOOST_CHECK_EQUAL( qfun[0][0], 9.75 );
}
