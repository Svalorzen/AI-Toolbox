#define BOOST_TEST_MODULE MDP_LinearProgramming
#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MAIN
#include <boost/test/unit_test.hpp>

#include <AIToolbox/MDP/Algorithms/LinearProgramming.hpp>
#include <AIToolbox/MDP/Algorithms/ValueIteration.hpp>

#include <AIToolbox/MDP/SparseModel.hpp>
#include <AIToolbox/MDP/Environments/CornerProblem.hpp>

#include "Utils/OldMDPModel.hpp"

BOOST_AUTO_TEST_CASE( escapeToCorners ) {
    using namespace AIToolbox::MDP;

    GridWorld grid(4, 4);

    Model model = makeCornerProblem(grid);
    size_t S = model.getS(), A = model.getA();

    double tolerance = 0.0001;
    ValueIteration solver(1000000, tolerance);
    LinearProgramming solver2;

    auto [bound, vfun, qfun] = solver(model);
    auto [bound2, vfun2, qfun2] = solver2(model);
    (void)bound; (void)bound2;

    for ( size_t s = 0; s < S; ++s ) {
        BOOST_CHECK_EQUAL( vfun.actions[s], vfun2.actions[s] );

        BOOST_TEST_INFO(s << " : " << vfun.values[s] << " --- " << vfun2.values[s]);
        BOOST_CHECK( std::fabs(vfun.values[s] - vfun2.values[s]) <= tolerance );

        for ( size_t a = 0; a < A; ++a ) {
            BOOST_TEST_INFO(s << " & " << a << " : " << qfun(s, a) << " --- " << qfun2(s, a));
            BOOST_CHECK( std::fabs(qfun(s,a) - qfun2(s,a)) <= tolerance );
        }
    }
}

BOOST_AUTO_TEST_CASE( escapeToCornersSparse ) {
    using namespace AIToolbox::MDP;

    GridWorld grid(4, 4);

    Model model = makeCornerProblem(grid);
    size_t S = model.getS(), A = model.getA();

    double tolerance = 0.0001;
    ValueIteration solver(1000000, tolerance);
    LinearProgramming solver2;

    auto [bound, vfun, qfun] = solver(model);
    auto [bound2, vfun2, qfun2] = solver2(model);
    (void)bound; (void)bound2;

    for ( size_t s = 0; s < S; ++s ) {
        BOOST_CHECK_EQUAL( vfun.actions[s], vfun2.actions[s] );

        BOOST_TEST_INFO(s << " : " << vfun.values[s] << " --- " << vfun2.values[s]);
        BOOST_CHECK( std::fabs(vfun.values[s] - vfun2.values[s]) <= tolerance );

        for ( size_t a = 0; a < A; ++a ) {
            BOOST_TEST_INFO(s << " & " << a << " : " << qfun(s, a) << " --- " << qfun2(s, a));
            BOOST_CHECK( std::fabs(qfun(s,a) - qfun2(s,a)) <= tolerance );
        }
    }
}

BOOST_AUTO_TEST_CASE( escapeToCornersNonEigen ) {
    using namespace AIToolbox::MDP;

    GridWorld grid(4, 4);

    OldMDPModel model = makeCornerProblem(grid);
    size_t S = model.getS(), A = model.getA();

    double tolerance = 0.0001;
    ValueIteration solver(1000000, tolerance);
    LinearProgramming solver2;

    auto [bound, vfun, qfun] = solver(model);
    auto [bound2, vfun2, qfun2] = solver2(model);
    (void)bound; (void)bound2;

    for ( size_t s = 0; s < S; ++s ) {
        BOOST_CHECK_EQUAL( vfun.actions[s], vfun2.actions[s] );

        BOOST_TEST_INFO(s << " : " << vfun.values[s] << " --- " << vfun2.values[s]);
        BOOST_CHECK( std::fabs(vfun.values[s] - vfun2.values[s]) <= tolerance );

        for ( size_t a = 0; a < A; ++a ) {
            BOOST_TEST_INFO(s << " & " << a << " : " << qfun(s, a) << " --- " << qfun2(s, a));
            BOOST_CHECK( std::fabs(qfun(s,a) - qfun2(s,a)) <= tolerance );
        }
    }
}
