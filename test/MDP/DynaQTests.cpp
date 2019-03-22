#define BOOST_TEST_MODULE MDP_DynaQ
#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MAIN
#include <boost/test/unit_test.hpp>

#include <AIToolbox/MDP/Model.hpp>
#include <AIToolbox/MDP/Algorithms/DynaQ.hpp>
#include <AIToolbox/MDP/Utils.hpp>

#include <AIToolbox/MDP/Environments/CliffProblem.hpp>

BOOST_AUTO_TEST_CASE( updates ) {
    using namespace AIToolbox::MDP;

    GridWorld grid(12, 3);

    auto model = makeCliffProblem(grid);
    model.setDiscount(0.9);

    DynaQ solver(model, 0.5);
    {
        // State goes to itself, thus needs to consider
        // next-step value.
        solver.stepUpdateQ(0, 0, 0, 10);
        BOOST_CHECK_EQUAL( solver.getQFunction()(0, 0), 5.0 );

        solver.stepUpdateQ(0, 0, 0, 10);
        BOOST_CHECK_EQUAL( solver.getQFunction()(0, 0), 9.75 );
    }
    {
        // Here it does not, so improvement is slower.
        solver.stepUpdateQ(3, 0, 4, 10);
        BOOST_CHECK_EQUAL( solver.getQFunction()(3, 0), 5.0 );

        solver.stepUpdateQ(3, 0, 4, 10);
        BOOST_CHECK_EQUAL( solver.getQFunction()(3, 0), 7.50 );
    }
    {
        // Test that index combinations are right.
        solver.stepUpdateQ(0, 1, 1, 10);
        BOOST_CHECK_EQUAL( solver.getQFunction()(0, 1), 5.0  );
        BOOST_CHECK_EQUAL( solver.getQFunction()(1, 0), 0.0  );
        BOOST_CHECK_EQUAL( solver.getQFunction()(1, 1), 0.0  );
    }
}
