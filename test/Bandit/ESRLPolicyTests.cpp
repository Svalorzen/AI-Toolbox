#define BOOST_TEST_MODULE Factored_Game_LRPPolicy
#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MAIN
#include <boost/test/unit_test.hpp>

#include <AIToolbox/Impl/Seeder.hpp>
#include <AIToolbox/Bandit/Policies/ESRLPolicy.hpp>

double testGuessingGame(unsigned a, unsigned b, unsigned c) {
    // Normalized to 1.0
    //
    // a_11
    //
    // 0.4 0.1 0.1
    // 0.1 0.1 0.1
    // 0.1 0.1 0.1
    //
    // a_12
    //
    // 0.1 0.1 0.1
    // 0.1 0.6 0.1
    // 0.1 0.1 0.1
    //
    // a_13
    //
    // 0.1 0.1 0.1
    // 0.1 0.1 0.1
    // 0.1 0.1 0.9

    if ( a == b && b == c ) {
        if ( a == 0 ) return 0.4;
        if ( a == 1 ) return 0.6;
        if ( a == 2 ) return 0.9;
    }
    return 0.1;
}

BOOST_AUTO_TEST_CASE( guessing_game ) {
    using namespace AIToolbox::Bandit;
    constexpr size_t A = 3;

    ESRLPolicy p1(A, 0.05, 2000, 7, 100);
    ESRLPolicy p2(A, 0.05, 2000, 7, 100);
    ESRLPolicy p3(A, 0.05, 2000, 7, 100);

    std::uniform_real_distribution<double> dist(0.0, 1.0);
    AIToolbox::RandomEngine rand(AIToolbox::Impl::Seeder::getSeed());

    unsigned t = 0;
    while (!p1.isExploiting()) {
        ++t;
        const size_t a = p1.sampleAction();
        const size_t b = p2.sampleAction();
        const size_t c = p3.sampleAction();

        const auto r = dist(rand) < testGuessingGame(a,b,c);

        p1.stepUpdateP(a, r);
        p2.stepUpdateP(b, r);
        p3.stepUpdateP(c, r);
    }
    // Check convergence to Nash equilibrium
    BOOST_CHECK(p1.getActionProbability(2) > 0.9);
    BOOST_CHECK(p2.getActionProbability(2) > 0.9);
    BOOST_CHECK(p3.getActionProbability(2) > 0.9);
}
